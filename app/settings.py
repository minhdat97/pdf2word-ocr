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

# Limits against abuse (a public service on a small server). Exam PDFs are A4/B5 with at
# most a few hundred drawings per page; the limits are far above that.
MAX_PAGE_PT = 3000  # longest page side (A0 is 3370pt; renders of bigger pages exhaust memory)
MAX_DRAWINGS = 20000  # vector paths on one page; more is kept as a picture of the page
MAX_STORAGE_BYTES = int(os.environ.get("P2W_MAX_STORAGE_MB", 4096)) * 1024 * 1024  # all files in DATA_DIR
MAX_PENDING_TASKS = int(os.environ.get("P2W_MAX_PENDING", 20))  # queued or running, all users
MAX_TASKS_PER_FILE = 6  # analyse + conversions with other options
# requests per client IP in RATE_WINDOW_SECONDS (a school shares one IP: generous)
RATE_WINDOW_SECONDS = 10 * 60
RATE_UPLOADS = int(os.environ.get("P2W_RATE_UPLOADS", 40))
RATE_TASKS = int(os.environ.get("P2W_RATE_TASKS", 120))

OCR_DPI = 300
OCR_MAX_SIDE_PX = 4500
DEFAULT_OCR_LANGUAGES = ["vie"]

# Threads of each model per conversion: the cores shared by the WORKERS conversions that run at once,
# at most 2 (a 2-core server running 2 conversions: 1 each, so they do not fight over the cores).
MODEL_THREADS = max(1, min(2, (os.cpu_count() or 1) // max(1, WORKERS)))

# VietOCR re-reads Vietnamese OCR words (app/vietocr.py). On when its model files are in
# VIETOCR_DIR (scripts/fetch_models.py) unless P2W_VIETOCR=0; threads per conversion.
VIETOCR_DIR = Path(os.environ.get("P2W_VIETOCR_DIR", BASE_DIR / "models" / "vietocr"))
VIETOCR_ENABLED = os.environ.get("P2W_VIETOCR", "1") != "0"
VIETOCR_THREADS = int(os.environ.get("P2W_VIETOCR_THREADS", MODEL_THREADS))
# A file with more scanned pages than this keeps Tesseract's words (VietOCR about doubles the OCR time).
VIETOCR_MAX_PAGES = int(os.environ.get("P2W_VIETOCR_MAX_PAGES", 30))

# Formula model (app/mathocr.py, pix2text-mfr-1.5) reads formula regions of OCR pages.
# On when its files are in MFR_DIR unless P2W_MFR=0; onnxruntime threads per conversion.
MFR_DIR = Path(os.environ.get("P2W_MFR_DIR", BASE_DIR / "models" / "mfr"))
MFR_ENABLED = os.environ.get("P2W_MFR", "1") != "0"
MFR_THREADS = int(os.environ.get("P2W_MFR_THREADS", MODEL_THREADS))
# A file with more scanned pages than this is converted without formula reading (VietOCR still runs):
# formulas take about half the OCR time, and a long scan would hold a worker for too long.
MFR_MAX_PAGES = int(os.environ.get("P2W_MFR_MAX_PAGES", 10))
# Where formula regions are looked for on OCR pages (app/mathregions.py): "rules" (answer options,
# fractions/radicals/bars over letters, words with superscripts or subscripts), "off", or another
# detector registered in mathregions.DETECTORS.
MATH_DETECT = os.environ.get("P2W_MATH_DETECT", "rules").strip().lower() or "rules"
