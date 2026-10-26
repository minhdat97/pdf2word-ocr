"""HTTP API: upload, then tasks that are polled until done:

1. POST /v1/upload-url        -> presigned-style upload URL + file token
2. PUT  <upload url>          -> raw PDF bytes
3. POST /v1/tasks             -> {"tool": "analyse-document", ...}
4. GET  /v1/tasks/{task_id}   -> poll until done (text / scan / mixed)
5. POST /v1/tasks             -> {"tool": "pdf-to-word", "options": {"ocr_type": "ocr", ...}}
6. GET  /v1/tasks/{task_id}   -> poll until done, read output_files
7. GET  /v1/files/{token}     -> download the .docx
"""

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import storage
from .settings import BASE_DIR, DATA_DIR, DEFAULT_OCR_LANGUAGES, MAX_UPLOAD_BYTES
from .tasks import TaskManager, available_languages

log = logging.getLogger("pdf2word")
manager = TaskManager()

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _error(status: int, error: str) -> JSONResponse:
    return JSONResponse({"success": False, "error": error}, status_code=status)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(300)
        removed = storage.cleanup_expired()
        manager.cleanup()
        if removed:
            log.info("removed %d expired files", removed)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaner = asyncio.create_task(_cleanup_loop())
    yield
    cleaner.cancel()
    manager.shutdown()


app = FastAPI(title="pdf2word-ocr", lifespan=lifespan)


class UploadUrlRequest(BaseModel):
    filename: str


@app.post("/v1/upload-url")
def create_upload_url(body: UploadUrlRequest):
    if not body.filename.lower().endswith(".pdf"):
        return _error(400, "only .pdf files are accepted")
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
    filename = Path(name).name if name else token
    if not filename.lower().endswith(f".{ext}"):
        filename = f"{Path(filename).stem}.{ext}"
    return FileResponse(storage.path_for(token), media_type=MEDIA_TYPES[ext], filename=filename)


@app.post("/v1/tasks", status_code=201)
def create_task(payload: dict):
    try:
        task = manager.submit(payload)
    except ValueError as exc:
        return _error(400, str(exc))
    return {"success": True, "data": {"task_id": task.task_id}}


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    task = manager.get(task_id)
    if task is None:
        return _error(404, "task not found")
    return {"success": True, "data": task.to_json()}


@app.get("/v1/languages")
def languages():
    return {"languages": available_languages(), "default": DEFAULT_OCR_LANGUAGES}


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
