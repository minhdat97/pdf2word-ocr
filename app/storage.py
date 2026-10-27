"""Local file store addressed by opaque tokens, with presigned-style upload URLs.

Swap this module for S3/R2 presigned URLs in production; the HTTP contract
(get an upload URL, PUT the bytes, pass the token to a task) stays the same.
"""

import hashlib
import hmac
import re
import shutil
import time
import uuid
from pathlib import Path

from .settings import (
    DATA_DIR,
    FILE_TTL_SECONDS,
    UPLOAD_SIGNING_KEY,
    UPLOAD_URL_TTL_SECONDS,
)

TOKEN_RE = re.compile(r"^[0-9a-f]{32}\.(pdf|docx)$")


def _sign(token: str, expires: int) -> str:
    msg = f"PUT\n{token}\n{expires}".encode()
    return hmac.new(UPLOAD_SIGNING_KEY, msg, hashlib.sha256).hexdigest()


def new_token(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext}"


def presign_upload(token: str) -> dict:
    expires = int(time.time()) + UPLOAD_URL_TTL_SECONDS
    return {
        "url": f"/v1/files/{token}?expires={expires}&sig={_sign(token, expires)}",
        "token": token,
        "expires_in": UPLOAD_URL_TTL_SECONDS,
    }


def verify_upload(token: str, expires: int, sig: str) -> bool:
    if expires < time.time():
        return False
    return hmac.compare_digest(_sign(token, expires), sig)


def path_for(token: str) -> Path:
    if not TOKEN_RE.match(token):
        raise ValueError("invalid file token")
    return DATA_DIR / token


def exists(token: str) -> bool:
    try:
        return path_for(token).is_file()
    except ValueError:
        return False


def store_output(src: Path, ext: str) -> str:
    token = new_token(ext)
    shutil.move(src, path_for(token))
    return token


def used_bytes() -> int:
    return sum(f.stat().st_size for f in DATA_DIR.iterdir() if f.is_file())


def cleanup_expired(now: float | None = None) -> int:
    now = now or time.time()
    removed = 0
    for f in DATA_DIR.iterdir():
        if f.is_file() and now - f.stat().st_mtime > FILE_TTL_SECONDS:
            f.unlink(missing_ok=True)
            removed += 1
    return removed
