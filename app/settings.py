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

# VietOCR re-reads Vietnamese OCR words (app/vietocr.py). On when its model files are there (scripts/fetch_models.py)
# unless P2W_VIETOCR=0; threads per conversion. Its folder (vietocr.model_dir()): P2W_VIETOCR_DIR when set, else
# VIETOCR_INT8_DIR when VIETOCR_INT8 is on and that folder has every file, else models/vietocr (float16).
VIETOCR_DIR = Path(os.environ.get("P2W_VIETOCR_DIR", BASE_DIR / "models" / "vietocr"))
VIETOCR_DIR_SET = bool(os.environ.get("P2W_VIETOCR_DIR"))
# The encoder's convolutions in int8 (scripts/quantize_vietocr.py): a default conversion of a scanned exam ~20 -> ~12 s
# per page on a 2-core ARM server by the Mac estimate, as many words right; off until the int8 speed is measured on the
# server (P2W_VIETOCR_INT8=1 turns it on). To make it the default, change the "0" on the next line to "1".
VIETOCR_INT8 = os.environ.get("P2W_VIETOCR_INT8", "0") != "0"
VIETOCR_INT8_DIR = BASE_DIR / "models" / "vietocr-int8"
VIETOCR_ENABLED = os.environ.get("P2W_VIETOCR", "1") != "0"
VIETOCR_THREADS = int(os.environ.get("P2W_VIETOCR_THREADS", MODEL_THREADS))
# A file with more scanned pages than this keeps Tesseract's words (VietOCR about doubles the OCR time).
VIETOCR_MAX_PAGES = int(os.environ.get("P2W_VIETOCR_MAX_PAGES", 30))
# Words VietOCR re-reads: "unsure" – only those whose reading it could improve (vietocr.unsure: Tesseract below its
# sure confidence, or a letter â ă ê ô with no tone mark or a grave accent), about 30 % of the words, as many words
# right on the test scans as "all" (every word) in about 40 % of the time.
VIETOCR_WORDS = os.environ.get("P2W_VIETOCR_WORDS", "unsure").strip().lower() or "unsure"

# Formula model (app/mathocr.py, pix2text-mfr-1.5) reads formula regions of OCR pages in a careful conversion
# ("Đọc kỹ", or every conversion with FORMULAS "model"). Usable when its files are in MFR_DIR unless P2W_MFR=0;
# onnxruntime threads per conversion.
MFR_DIR = Path(os.environ.get("P2W_MFR_DIR", BASE_DIR / "models" / "mfr"))
MFR_ENABLED = os.environ.get("P2W_MFR", "1") != "0"
MFR_THREADS = int(os.environ.get("P2W_MFR_THREADS", MODEL_THREADS))
# A file with more scanned pages than this reads its formulas without the model (as by default, FORMULAS "scripts"):
# with the model a page takes up to about 2 minutes on a 2-core server (35 s on average on the test scans), and a
# long scan would hold a worker for too long.
MFR_MAX_PAGES = int(os.environ.get("P2W_MFR_MAX_PAGES", 5))
# Conversions with the formula model running at once (app/tasks.py): one holds a core and ~1.5 GB for minutes, so
# the other worker stays free for default conversions; more careful ones wait in the queue (status "queued").
CAREFUL_WORKERS = int(os.environ.get("P2W_CAREFUL_WORKERS", 1))
# Where formula regions are looked for on OCR pages (app/mathregions.py): "rules" (answer options,
# fractions/radicals/bars over letters, words with superscripts or subscripts), "off", or another
# detector registered in mathregions.DETECTORS.
MATH_DETECT = os.environ.get("P2W_MATH_DETECT", "rules").strip().lower() or "rules"
# How formula regions are read by default: "scripts" (app/scriptocr.py: superscripts and subscripts of numbers and
# units read by Tesseract, no model, about a second per page on a 2-core server; fractions and roots kept as
# pictures), "model" (every conversion as a careful one) or "off". A careful conversion (convert(careful=True), task
# option "careful") uses the model where it is installed and the file has at most MFR_MAX_PAGES scanned pages, and
# "scripts" otherwise.
FORMULAS = os.environ.get("P2W_FORMULAS", "scripts").strip().lower() or "scripts"
