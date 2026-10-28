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
# Words VietOCR re-reads: "all", or "unsure" – only those whose reading it could improve (vietocr.unsure: Tesseract
# below its sure confidence, or a letter â ă ê ô with no tone mark or a grave accent), about 30 % of the words.
VIETOCR_WORDS = os.environ.get("P2W_VIETOCR_WORDS", "all").strip().lower() or "all"

# Formula model (app/mathocr.py, pix2text-mfr-1.5) reads formula regions of OCR pages.
# On when its files are in MFR_DIR unless P2W_MFR=0; onnxruntime threads per conversion.
MFR_DIR = Path(os.environ.get("P2W_MFR_DIR", BASE_DIR / "models" / "mfr"))
MFR_ENABLED = os.environ.get("P2W_MFR", "1") != "0"
MFR_THREADS = int(os.environ.get("P2W_MFR_THREADS", MODEL_THREADS))
# A file with more scanned pages than this is converted without formula reading (VietOCR still runs):
# formulas take about half the OCR time, and a long scan would hold a worker for too long.
MFR_MAX_PAGES = int(os.environ.get("P2W_MFR_MAX_PAGES", 10))
# How a region is read (app/mathocr.py read_regions; measured on VL1212, two physics scans, 30 scanned pages):
# - MFR_PADS: margins of white around the crop, in order; the third is read only when the first two disagree.
# - MFR_SAME_STOP (on): a first reading that is plain text equal to Tesseract's is kept without a second one
#   (it writes the OCR text anyway): ~20 % fewer crops, same .docx. 2: also a reading with scripts whose
#   characters are Tesseract's (3 % fewer again; not used: Tesseract does not see the scripts).
# - MFR_FIRST_STOP (off): a region whose first reading can be used in no vote is not read again (4 % fewer
#   crops, loses 5 of 206 readings).
# - MFR_HEAD (on): a region still unsettled is read once more without its last short word (a unit).
# - MFR_ENCODE_ONE (on): the encoder reads one crop per run, not a batch: same readings, ~10 % faster and half
#   the memory (the activations of one crop stay in cache); a dynamically quantized INT8 encoder needs it (it
#   scales activations over the whole batch: a crop's reading would depend on the other crops of its page).
# - MFR_SKIP_KINDS: region kinds (option, tall, inline) not read at all (every kind changes text: off).
MFR_PADS = tuple(int(p) for p in os.environ.get("P2W_MFR_PADS", "0,3,8").split(",") if p.strip())
MFR_SAME_STOP = int(os.environ.get("P2W_MFR_SAME_STOP", "1"))
MFR_FIRST_STOP = os.environ.get("P2W_MFR_FIRST_STOP", "0") == "1"
MFR_HEAD = os.environ.get("P2W_MFR_HEAD", "1") != "0"
MFR_ENCODE_ONE = os.environ.get("P2W_MFR_ENCODE_ONE", "1") != "0"
MFR_SKIP_KINDS = frozenset(k.strip() for k in os.environ.get("P2W_MFR_SKIP_KINDS", "").split(",") if k.strip())
# Where formula regions are looked for on OCR pages (app/mathregions.py): "rules" (answer options,
# fractions/radicals/bars over letters, words with superscripts or subscripts), "off", or another
# detector registered in mathregions.DETECTORS.
MATH_DETECT = os.environ.get("P2W_MATH_DETECT", "rules").strip().lower() or "rules"
# How formula regions are read: "model" (the formula model where it is installed and not skipped for the file; a
# long scan that skips it gets "scripts"), "scripts" (app/scriptocr.py: superscripts and subscripts of numbers
# and units read by Tesseract, no model; fractions and roots kept as pictures), "off".
FORMULAS = os.environ.get("P2W_FORMULAS", "model").strip().lower() or "model"
