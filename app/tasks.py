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

from . import storage
from .analyze import analyse
from .convert import convert
from .errors import TaskError
from .settings import DEFAULT_OCR_LANGUAGES, TASK_TTL_SECONDS, TESSDATA_DIR, WORKERS

log = logging.getLogger("pdf2word.tasks")

TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TOOLS = ("analyse-document", "pdf-to-word")
# ocr_type -> conversion mode ("solid-ocr" is accepted as another name for "ocr").
OCR_TYPES = {"": "basic", "ocr": "ocr", "solid-ocr": "ocr", "ocr-full": "ocr-full"}


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
        options = {"ocr_type": ocr_type, "languages": languages, "strip_watermarks": strip}
    else:
        options = {}
    return Task(task_id=task_id, tool=tool, input_tokens=tokens, options=options)


class TaskManager:
    def __init__(self, workers: int = WORKERS):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task")

    def submit(self, payload: dict) -> Task:
        task = _validate(payload)
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError("duplicate task_id")
            self._tasks[task.task_id] = task
        self._pool.submit(self._run, task)
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _run(self, task: Task) -> None:
        task.status, task.started = "processing", time.time()

        def progress(done: int, total: int) -> None:
            task.progress = {"current": done, "total": total}

        try:
            path = storage.path_for(task.input_tokens[0])
            if task.tool == "analyse-document":
                task.result_options = analyse(path)
            else:
                mode = OCR_TYPES[task.options["ocr_type"]]
                out, task.result_options = convert(
                    path, mode, task.options["languages"], progress, task.options["strip_watermarks"]
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
        self._pool.shutdown(wait=False, cancel_futures=True)
