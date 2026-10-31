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
    ABANDON_SECONDS,
    CAREFUL_QUEUE,
    CAREFUL_WORKERS,
    DEFAULT_OCR_LANGUAGES,
    FORMULAS,
    MAX_PENDING_TASKS,
    MAX_TASKS_PER_FILE,
    MFR_MAX_FORMULAS,
    MFR_MAX_PAGES,
    SECONDS_PER_FORMULA,
    SECONDS_PER_PICTURE_PAGE,
    SECONDS_PER_SCAN_PAGE,
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
    """A task refused for load: `code` is "busy" (the queue is full), "careful_busy" (CAREFUL_QUEUE careful
    conversions wait already) or "too_many_tasks" (this file already has one running, or has had enough)."""

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
    # "tick" counts the progress calls: news for the page within a long page (formula pictures read chunk by chunk)
    progress: dict = field(default_factory=lambda: {"current": 0, "total": 0, "tick": 0})
    output_files: list[str] | None = None
    result_options: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    completed: float | None = None
    # runs with the formula model (careful): at most CAREFUL_WORKERS such tasks at once (TaskManager)
    careful: bool = field(default=False, repr=False)
    seen: float = field(default_factory=time.time, repr=False)  # last time its page asked about it
    cancelled: bool = field(default=False, repr=False)  # its page was closed: stop at the next page

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
        careful = options.get("careful", False)  # "Chuyển thành công thức sửa được": the formula model (convert())
        if not isinstance(careful, bool):
            raise ValueError("invalid_options: careful must be true or false")
        options = {"ocr_type": ocr_type, "languages": languages, "strip_watermarks": strip, "careful": careful}
    else:
        options = {}
    return Task(task_id=task_id, tool=tool, input_tokens=tokens, options=options)


def careful_info() -> dict:
    """What the page needs to offer "Chuyển thành công thức sửa được" (added to the analyse-document result): whether
    a careful conversion can use the formula model now (installed, not turned off, not broken), the most scanned
    pages a file may have for it, the most formula pictures of text pages it reads per file, and the seconds the
    page counts per formula picture, per page holding them and per scanned page for its "khoảng X phút". (For scanned pages it is not
    offered when every OCR conversion uses the model anyway, P2W_FORMULAS=model: "scan_offered".)"""
    return {
        "available": mathocr.available(),
        "scan_offered": FORMULAS != "model",
        "max_scan_pages": MFR_MAX_PAGES,
        "max_formulas": MFR_MAX_FORMULAS,
        "seconds_per_formula": SECONDS_PER_FORMULA,
        "seconds_per_picture_page": SECONDS_PER_PICTURE_PAGE,
        "seconds_per_scan_page": SECONDS_PER_SCAN_PAGE,
    }


def _uses_model(task: Task) -> bool:
    """A conversion that may read formulas with the formula model, the model usable: careful ("Chuyển thành công
    thức sửa được": scanned pages and the formula pictures of text pages), or OCR with P2W_FORMULAS=model. (One that
    turns out not to – a file with too many scanned pages and no text page – gives its turn back as soon as
    convert() knows: TaskManager._run.)"""
    if task.tool != "pdf-to-word":
        return False
    if OCR_TYPES[task.options["ocr_type"]] == "basic":  # no OCR: only the formula pictures of its text pages
        return task.options["careful"] and MFR_MAX_FORMULAS > 0 and mathocr.available()
    return (task.options["careful"] or FORMULAS == "model") and mathocr.available()


class TaskManager:
    """Runs tasks on `workers` threads, oldest first, except that at most `careful_workers` conversions with the
    formula model (careful: up to ~2 minutes per scanned page on a 2-core server) run at once. A careful task
    that has to wait stays "queued" without holding a thread, and the tasks behind it go ahead of it meanwhile;
    at most `careful_queue` wait. A queued task whose page was closed (cancel()) or has not asked about it for
    ABANDON_SECONDS is dropped; a running one cancelled stops at its next page."""

    def __init__(self, workers: int = WORKERS, careful_workers: int = CAREFUL_WORKERS, careful_queue: int = CAREFUL_QUEUE):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._workers = max(1, workers)
        self._careful_workers = max(1, careful_workers)
        self._careful_queue = max(0, careful_queue)
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
            self._drop_abandoned()
            unfinished = [t for t in self._tasks.values() if t.status in ("queued", "processing")]
            same_file = [t for t in self._tasks.values() if t.input_tokens == task.input_tokens]
            if len(same_file) >= MAX_TASKS_PER_FILE or any(t.input_tokens == task.input_tokens for t in unfinished):
                raise Busy("too_many_tasks")
            if len(unfinished) >= MAX_PENDING_TASKS:
                raise Busy("busy")
            waiting = sum(t.careful for t in self._queue)
            if task.careful and self._careful_running + waiting >= self._careful_workers + self._careful_queue:
                raise Busy("careful_busy")
            self._tasks[task.task_id] = task
            self._queue.append(task)
            self._start_ready()
        return task

    def get(self, task_id: str) -> Task | None:
        """The task, now known to be still wanted (its page asks about it)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.seen = time.time()
            return task

    def cancel(self, task_id: str) -> bool:
        """Its page was closed: a queued task is dropped, a running one stops at its next page."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in ("done", "failed"):
                return False
            if task in self._queue:
                self._drop(task, "cancelled")
            else:
                task.cancelled = True
            return True

    def in_use(self) -> set[str]:
        """Input files of the tasks not finished yet (storage.cleanup_expired keeps them)."""
        with self._lock:
            return {tok for t in self._tasks.values() if t.status in ("queued", "processing") for tok in t.input_tokens}

    def _drop(self, task: Task, code: str) -> None:
        """A queued task that will not run (self._lock held)."""
        self._queue.remove(task)
        task.status, task.error, task.completed = "failed", code, time.time()
        log.info("task %s dropped: %s", task.task_id, code)

    def _drop_abandoned(self) -> None:
        """Queued tasks nobody asked about for ABANDON_SECONDS (self._lock held)."""
        now = time.time()
        for task in [t for t in self._queue if now - t.seen > ABANDON_SECONDS]:
            self._drop(task, "abandoned")

    def _release_careful(self, task: Task) -> None:
        """A careful task that will not use the model after all gives its turn to the next one."""
        with self._lock:
            if task.careful:
                task.careful = False
                self._careful_running -= 1
                self._start_ready()

    def _start_ready(self) -> None:
        """Hand the oldest tasks that may run to free threads (self._lock held): a careful one only while fewer
        than careful_workers careful ones run."""
        self._drop_abandoned()
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
            if task.cancelled:
                raise TaskError("cancelled")
            task.progress = {"current": done, "total": total, "tick": task.progress.get("tick", 0) + 1}

        try:
            if not storage.exists(task.input_tokens[0]):  # (kept while the task waits: storage.cleanup_expired)
                raise TaskError("input_tokens")
            path = storage.path_for(task.input_tokens[0])
            if task.tool == "analyse-document":
                task.result_options = {**analyse(path), "careful": careful_info()}
            else:
                mode = OCR_TYPES[task.options["ocr_type"]]
                out, task.result_options = convert(
                    path, mode, task.options["languages"], progress, task.options["strip_watermarks"],
                    task.options["careful"], model_unused=lambda: self._release_careful(task),
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
