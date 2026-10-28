"""HTTP API: upload, then tasks that are polled until done:

1. POST /v1/upload-url        -> presigned-style upload URL + file token
2. PUT  <upload url>          -> raw PDF bytes
3. POST /v1/tasks             -> {"tool": "analyse-document", ...}
4. GET  /v1/tasks/{task_id}   -> poll until done (text / scan / mixed); a queued task nobody polls for
                                 10 minutes is dropped (POST /v1/tasks/{task_id}/cancel drops it at once)
5. POST /v1/tasks             -> {"tool": "pdf-to-word", "options": {"ocr_type": "ocr", ...}}
6. GET  /v1/tasks/{task_id}   -> poll until done, read output_files
7. GET  /v1/files/{token}     -> download the .docx
"""

import asyncio
import contextlib
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import storage
from .settings import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_OCR_LANGUAGES,
    MAX_STORAGE_BYTES,
    MAX_UPLOAD_BYTES,
    RATE_TASKS,
    RATE_UPLOADS,
    RATE_WINDOW_SECONDS,
)
from .tasks import Busy, TaskManager, available_languages

log = logging.getLogger("pdf2word")
manager = TaskManager()

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _error(status: int, error: str) -> JSONResponse:
    return JSONResponse({"success": False, "error": error}, status_code=status)


class RateLimit:
    """At most `limit` requests per client in a sliding window."""

    def __init__(self, limit: int, window: float = RATE_WINDOW_SECONDS):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > 10000:  # forget idle clients
                for k in [k for k, h in self._hits.items() if not h or now - h[-1] > self.window]:
                    del self._hits[k]
            return True


upload_limit = RateLimit(RATE_UPLOADS)
task_limit = RateLimit(RATE_TASKS)


def _client(request: Request) -> str:
    # The server listens on 127.0.0.1 behind Cloudflare Tunnel, which sets
    # CF-Connecting-IP to the visitor's address (a visitor cannot forge it).
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


# The page loads only its own files; no framing, no third-party scripts.
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(300)
        removed = storage.cleanup_expired(keep=manager.in_use())  # not the uploads queued tasks still need
        manager.cleanup()
        if removed:
            log.info("removed %d expired files", removed)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    os.umask(0o077)  # uploads and results readable by this user only
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaner = asyncio.create_task(_cleanup_loop())
    yield
    cleaner.cancel()
    manager.shutdown()


# No public API docs (/docs, /redoc, /openapi.json): the page is the only client.
app = FastAPI(title="pdf2word-ocr", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    h = response.headers
    h["X-Content-Type-Options"] = "nosniff"
    h["Referrer-Policy"] = "no-referrer"
    h["X-Frame-Options"] = "DENY"
    h["Strict-Transport-Security"] = "max-age=31536000"
    h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/v1/"):
        # Teachers' files must not stay in Cloudflare's cache (it caches .pdf/.docx
        # by default) after the server deleted them.
        h["Cache-Control"] = "no-store, private"
        h["X-Robots-Tag"] = "noindex"
    else:
        h["Content-Security-Policy"] = CSP
        h["Cache-Control"] = "no-cache"  # the page updates as soon as it is deployed
    return response


class UploadUrlRequest(BaseModel):
    filename: str


@app.post("/v1/upload-url")
def create_upload_url(body: UploadUrlRequest, request: Request):
    if not body.filename.lower().endswith(".pdf"):
        return _error(400, "only .pdf files are accepted")
    if not upload_limit.allow(_client(request)):
        return _error(429, "rate_limited")
    if storage.used_bytes() > MAX_STORAGE_BYTES - MAX_UPLOAD_BYTES:
        log.warning("storage full")
        return _error(503, "server_full")
    return storage.presign_upload(storage.new_token("pdf"))


@app.put("/v1/files/{token}")
async def upload_file(token: str, request: Request, expires: int = 0, sig: str = ""):
    if not token.endswith(".pdf") or not storage.verify_upload(token, expires, sig):
        return _error(403, "invalid or expired upload URL")
    path = storage.path_for(token)
    if path.exists():
        return _error(409, "file already uploaded")

    part = path.with_suffix(".part")
    size = 0
    with part.open("wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                part.unlink(missing_ok=True)
                return _error(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            f.write(chunk)
    with part.open("rb") as f:
        if b"%PDF-" not in f.read(1024):
            part.unlink(missing_ok=True)
            return _error(400, "not a PDF file")
    part.rename(path)
    return {"success": True, "token": token, "size": size}


@app.get("/v1/files/{token}")
def download_file(token: str, name: str | None = None):
    if not storage.exists(token):
        return _error(404, "file not found or expired")
    ext = token.rsplit(".", 1)[1]
    # a plain file name for Content-Disposition: no path, quotes or control characters
    filename = re.sub(r'[\x00-\x1f\x7f"\\/]', "", Path(name).name)[:150] if name else token
    if not filename.lower().endswith(f".{ext}"):
        filename = f"{Path(filename).stem}.{ext}"
    return FileResponse(storage.path_for(token), media_type=MEDIA_TYPES[ext], filename=filename)


@app.post("/v1/tasks", status_code=201)
def create_task(payload: dict, request: Request):
    if not task_limit.allow(_client(request)):
        return _error(429, "rate_limited")
    try:
        task = manager.submit(payload)
    except Busy as exc:
        return _error(503 if exc.code in ("busy", "careful_busy") else 429, exc.code)
    except ValueError as exc:
        return _error(400, str(exc))
    return {"success": True, "data": {"task_id": task.task_id}}


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    task = manager.get(task_id)
    if task is None:
        return _error(404, "task not found")
    return {"success": True, "data": task.to_json()}


@app.post("/v1/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """The page was closed or reloaded (navigator.sendBeacon): its queued task is dropped, a running one stops at
    its next page, so a careful conversion nobody waits for does not hold the turn of the next one."""
    if not manager.cancel(task_id):
        return _error(404, "task not found")
    return {"success": True}


@app.get("/v1/languages")
def languages():
    return {"languages": available_languages(), "default": DEFAULT_OCR_LANGUAGES}


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
