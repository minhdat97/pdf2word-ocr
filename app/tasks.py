"""In-process task queue: submit a task, then poll it.

Production would run workers in separate processes (Celery/RQ/Arq + Redis) so
CPU-heavy conversions never share the API process; the HTTP shape stays.
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import mathocr, storage
from .analyze import analyse
from .convert import convert
from .errors import TaskError
from .settings import (
    CAREFUL_WORKERS,
    DEFAULT_OCR_LANGUAGES,
    FORMULAS,
    MAX_PENDING_TASKS,
    MAX_TASKS_PER_FILE,
    MFR_MAX_PAGES,
    TASK_TTL_SECONDS,
    TESSDATA_DIR,
    WORKERS,
)

log = logging.getLogger("pdf2word.tasks")

TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TOOLS = ("analyse-document", "pdf-to-word")
# ocr_type -> conversion mode ("solid-ocr" is accepted as another name for "ocr").
OCR_TYPES = {"": "basic", "ocr": "ocr", "solid-ocr": "ocr", "ocr-full": "ocr-full"}


class Busy(Exception):
    """A task refused for load: `code` is "busy" (the queue is full) or
    "too_many_tasks" (this file already has one running, or has had enough)."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def available_languages() -> list[str]:
    return sorted(p.stem for p in TESSDATA_DIR.glob("*.traineddata") if p.stem != "osd")


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None


@dataclass
class Task:
    task_id: str
    tool: str
    input_tokens: list[str]
    options: dict
    status: str = "queued"  # queued -> processing -> done | failed
    progress: dict = field(default_factory=lambda: {"current": 0, "total": 0})
    output_files: list[str] | None = None
    result_options: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    completed: float | None = None
    # runs with the formula model ("Đọc kỹ"): at most CAREFUL_WORKERS such tasks at once (TaskManager)
    careful: bool = field(default=False, repr=False)

    def to_json(self) -> dict:
        finished = self.status in ("done", "failed")
        return {
            "task_id": self.task_id,
            "tool": self.tool,
            "status": self.status,
            "progress": self.progress,
            "result": {
                "success": self.status == "done",
                "error_fe": self.error,
                "output_files": self.output_files,
                "options": self.result_options,
                "started_processing": _iso(self.started),
                "completed_processing": _iso(self.completed),
                "task": {"tool": self.tool, "input_tokens": self.input_tokens, "options": self.options},
            }
            if finished
            else None,
        }


def _validate(payload: dict) -> Task:
    task_id = payload.get("task_id", "")
    if not isinstance(task_id, str) or not TASK_ID_RE.match(task_id):
        raise ValueError("task_id must be 32 lowercase hex characters")
    tool = payload.get("tool")
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {', '.join(TOOLS)}")
    tokens = payload.get("input_tokens")
    if (
        not isinstance(tokens, list)
        or len(tokens) != 1
        or not isinstance(tokens[0], str)
        or not tokens[0].endswith(".pdf")
        or not storage.exists(tokens[0])
    ):
        raise ValueError("input_tokens")

    options = payload.get("options") or {}
    if tool == "pdf-to-word":
        ocr_type = options.get("ocr_type", "")
        languages = options.get("languages") or []
        if ocr_type not in OCR_TYPES or not isinstance(languages, list):
            raise ValueError("invalid_options")
        unknown = set(languages) - set(available_languages())
        if unknown:
            raise ValueError(f"invalid_options: unknown languages {sorted(unknown)}")
        if not languages:
            languages = DEFAULT_OCR_LANGUAGES
        strip = options.get("strip_watermarks", False)
        if not isinstance(strip, bool):
            raise ValueError("invalid_options: strip_watermarks must be true or false")
        careful = options.get("careful", False)  # "Đọc kỹ": formulas of scanned pages read by the formula model
        if not isinstance(careful, bool):
            raise ValueError("invalid_options: careful must be true or false")
        options = {"ocr_type": ocr_type, "languages": languages, "strip_watermarks": strip, "careful": careful}
    else:
        options = {}
    return Task(task_id=task_id, tool=tool, input_tokens=tokens, options=options)


def careful_info() -> dict:
    """What the page needs to offer "Đọc kỹ" (added to the analyse-document result): whether a careful conversion
    can use the formula model now (installed, not turned off, not broken; not when every conversion uses it,
    P2W_FORMULAS=model) and the most scanned pages a file may have for it."""
    return {"available": FORMULAS != "model" and mathocr.available(), "max_scan_pages": MFR_MAX_PAGES}


def _uses_model(task: Task) -> bool:
    """A conversion that will read formulas with the formula model: OCR asked, "Đọc kỹ" (or P2W_FORMULAS=model),
    the model usable. (A careful file with too many scanned pages is read without it but still counted: the page
    does not offer "Đọc kỹ" for such a file.)"""
    if task.tool != "pdf-to-word" or OCR_TYPES[task.options["ocr_type"]] == "basic":
        return False
    return (task.options["careful"] or FORMULAS == "model") and mathocr.available()


class TaskManager:
    """Runs tasks on `workers` threads, oldest first, except that at most `careful_workers` conversions with the
    formula model ("Đọc kỹ": up to ~2 minutes per scanned page on a 2-core server) run at once. A careful task
    that has to wait stays "queued" without holding a thread, and the tasks behind it go ahead of it meanwhile."""

    def __init__(self, workers: int = WORKERS, careful_workers: int = CAREFUL_WORKERS):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._workers = max(1, workers)
        self._careful_workers = max(1, careful_workers)
        self._queue: list[Task] = []  # submitted, not started yet, oldest first
        self._running = self._careful_running = 0
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="task")

    def submit(self, payload: dict) -> Task:
        task = _validate(payload)
        task.careful = _uses_model(task)
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError("duplicate task_id")
            unfinished = [t for t in self._tasks.values() if t.status in ("queued", "processing")]
            same_file = [t for t in self._tasks.values() if t.input_tokens == task.input_tokens]
            if len(same_file) >= MAX_TASKS_PER_FILE or any(t.input_tokens == task.input_tokens for t in unfinished):
                raise Busy("too_many_tasks")
            if len(unfinished) >= MAX_PENDING_TASKS:
                raise Busy("busy")
            self._tasks[task.task_id] = task
            self._queue.append(task)
            self._start_ready()
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _start_ready(self) -> None:
        """Hand the oldest tasks that may run to free threads (self._lock held): a careful one only while fewer
        than careful_workers careful ones run."""
        while self._running < self._workers and not self._closed:
            task = next(
                (t for t in self._queue if not t.careful or self._careful_running < self._careful_workers), None
            )
            if task is None:
                return
            try:
                self._pool.submit(self._work, task)
            except RuntimeError:  # the pool is shutting down (the service stops)
                return
            self._queue.remove(task)
            self._running += 1
            self._careful_running += task.careful

    def _work(self, task: Task) -> None:
        try:
            self._run(task)
        finally:
            with self._lock:
                self._running -= 1
                self._careful_running -= task.careful
                self._start_ready()

    def _run(self, task: Task) -> None:
        task.status, task.started = "processing", time.time()

        def progress(done: int, total: int) -> None:
            task.progress = {"current": done, "total": total}

        try:
            path = storage.path_for(task.input_tokens[0])
            if task.tool == "analyse-document":
                task.result_options = {**analyse(path), "careful": careful_info()}
            else:
                mode = OCR_TYPES[task.options["ocr_type"]]
                out, task.result_options = convert(
                    path, mode, task.options["languages"], progress, task.options["strip_watermarks"],
                    task.options["careful"],
                )
                task.output_files = [storage.store_output(out, "docx")]
            task.status = "done"
        except TaskError as exc:
            log.info("task %s failed: %s %s", task.task_id, exc.code, exc.detail)
            task.status, task.error = "failed", exc.code
        except Exception:
            log.exception("task %s crashed", task.task_id)
            task.status, task.error = "failed", "internal_server_error"
        finally:
            task.completed = time.time()

    def cleanup(self, now: float | None = None) -> None:
        now = now or time.time()
        with self._lock:
            for task_id in [
                t.task_id
                for t in self._tasks.values()
                if t.completed and now - t.completed > TASK_TTL_SECONDS
            ]:
                del self._tasks[task_id]

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self._queue.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
