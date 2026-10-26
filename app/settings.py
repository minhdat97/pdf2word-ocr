import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("P2W_DATA_DIR", BASE_DIR / "data"))
TESSDATA_DIR = Path(os.environ.get("TESSDATA_PREFIX", BASE_DIR / "tessdata"))
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "tesseract")

# Signs upload URLs, like an object store's presigned PUT. A random per-process
# secret is fine for a single-instance prototype.
UPLOAD_SIGNING_KEY = os.environ.get("P2W_SIGNING_KEY", secrets.token_hex(32)).encode()
UPLOAD_URL_TTL_SECONDS = 15 * 60

FILE_TTL_SECONDS = int(os.environ.get("P2W_FILE_TTL", 60 * 60))
TASK_TTL_SECONDS = FILE_TTL_SECONDS
MAX_UPLOAD_BYTES = int(os.environ.get("P2W_MAX_UPLOAD_MB", 50)) * 1024 * 1024
MAX_PAGES = int(os.environ.get("P2W_MAX_PAGES", 50))
WORKERS = int(os.environ.get("P2W_WORKERS", 2))

OCR_DPI = 300
OCR_MAX_SIDE_PX = 4500
DEFAULT_OCR_LANGUAGES = ["vie"]
