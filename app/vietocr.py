"""Second reading of OCR words with VietOCR (vgg_transformer, Apache-2.0).

Tesseract finds the words and reads them; VietOCR, trained on Vietnamese,
re-reads each word's crop. refine_words takes its tone marks when both read the
same letters, and its whole word where Tesseract was unsure and VietOCR was not.

The model runs on onnxruntime (models/vietocr, fetched by scripts/fetch_models.py):
encoder.onnx (image -> keys and values for the decoder's cross-attention),
decoder.onnx (one step: last character + cached keys/values -> next-character
probabilities), config.json (vocabulary, image sizes, decoder shape).
Preprocessing and greedy decoding reproduce vietocr's Predictor.predict_batch:
Pillow's LANCZOS resize to 32 px high, crops of one width decoded together,
mean character probability (same text on 99.96% of 9329 word crops).
Without the model files (or with P2W_VIETOCR=0) nothing changes.
"""

import json
import logging
import math
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ocr_fix import _TONES, _toneless
from .settings import VIETOCR_DIR, VIETOCR_ENABLED, VIETOCR_THREADS

log = logging.getLogger("pdf2word.vietocr")

MODEL_FILES = ("encoder.onnx", "decoder.onnx", "config.json")
SOS, EOS = 1, 2
ENCODE_BATCH = 16  # crops per encoder run (bounds memory; decoding takes a whole width at once)

# Rule for choose() (30 scanned pages: words right 92.2% -> 96.5% at 300 dpi, 91.5% -> 95.7% at 200 dpi)
KEEP_MARKS_CONF = 90  # VietOCR may not remove a tone mark from a word Tesseract was this sure of
TESS_SURE = 92  # a word read with other letters is kept from this Tesseract confidence
VOCR_SURE = 0.5  # VietOCR's mean character probability needed to take its word
# VietOCR has no formulas in its training text: it reads "R=" as "?" and "(K)=t(°C)+273"
# as "?t(?C) % 273". Its word is only taken when it keeps these characters of Tesseract's
# (and the minus: "c.-3°C." read "C.3 ?C" would change the value).
MATH_CHARS = set("=+°%/<>()^-–")
# A word crop shows neither the line's capitals nor the end of its sentence: VietOCR reads
# "lần." as "Lần". Where Tesseract was this sure of the word, VietOCR's word keeps the case
# of the letters it shares with Tesseract's ("lần." at 86 is not "Lần"; a label "c." read at
# 54 may be "C.") and may not leave out a punctuation mark Tesseract read ("T,;" -> "T,").
CASE_CONF = 60
MARKS = set(".,;:!?")
# A full stop, comma, semicolon or colon ending Tesseract's word is put back on VietOCR's when
# VietOCR left it out; "?" and "!" there are often an exponent misread ("x.10?" is "x.10-9").
END_MARKS = set(".,;:")


# --- Pillow's LANCZOS resize (8-bit fixed point), so crops match vietocr's ---


def _lanczos(x: np.ndarray) -> np.ndarray:
    def sinc(v):
        v = v * math.pi
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v == 0, 1.0, np.sin(v) / np.where(v == 0, 1.0, v))

    return np.where((x >= -3.0) & (x < 3.0), sinc(x) * sinc(x / 3), 0.0)


@lru_cache(maxsize=1024)
def _weights(in_size: int, out_size: int) -> np.ndarray:
    """[out, in] resampling matrix, coefficients scaled by 2^22 and rounded as Pillow does."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 3.0 * filterscale
    ss = 1.0 / filterscale
    m = np.zeros((out_size, in_size))
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        k = _lanczos((np.arange(xmin, xmax) + 0.5 - center) * ss)
        total = k.sum()
        if total != 0.0:
            k = k / total
        m[xx, xmin:xmax] = np.trunc(k * (1 << 22) + np.where(k < 0, -0.5, 0.5))
    return m


def _resample(img: np.ndarray, m: np.ndarray, axis: int) -> np.ndarray:
    # float64 holds the integer sums exactly (< 2^53)
    acc = np.tensordot(img.astype(np.float64), m, axes=([axis], [1]))
    acc = np.moveaxis(acc, -1, axis)
    return np.clip((acc + (1 << 21)) // (1 << 22), 0, 255).astype(np.uint8)


def resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pillow's Image.resize((width, height), LANCZOS) of an H x W (x C) uint8 image."""
    h, w = img.shape[:2]
    # Pass order matters for the rounding; Pillow 13 goes vertical first when it shrinks
    # rows by more than twice what it changes columns (rule found by comparing outputs).
    if h > height and 2 * (w - width) < h - height:
        img = _resample(img, _weights(h, height), 0)
        h = height
    if width != w:
        img = _resample(img, _weights(w, width), 1)
    if height != h:
        img = _resample(img, _weights(h, height), 0)
    return img


# --- model -------------------------------------------------------------------


class VietOCR:
    def __init__(self, model_dir: Path, threads: int = 2):
        import onnxruntime as ort

        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.chars = ["<pad>", "<sos>", "<eos>", "*", *cfg["vocab"]]
        self.height = cfg["image_height"]
        self.min_width, self.max_width = cfg["image_min_width"], cfg["image_max_width"]
        self.max_steps = cfg["max_seq_length"]
        self.cache_shape = (cfg["layers"], cfg["heads"], cfg["head_dim"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        # idle threads sleep instead of spinning: CPU stays free for Tesseract and the other conversions
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.enable_mem_pattern = False  # input shapes change every run
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(model_dir / "encoder.onnx"), opts, providers=providers)
        self.decoder = ort.InferenceSession(str(model_dir / "decoder.onnx"), opts, providers=providers)

    def prepare(self, img: np.ndarray) -> np.ndarray:
        """Crop -> [3, 32, W] float32 in 0..1 (vietocr's process_input)."""
        h, w = img.shape[:2]
        width = math.ceil(int(self.height * float(w) / float(h)) / 10) * 10
        width = min(max(width, self.min_width), self.max_width)
        out = resize(img, width, self.height)
        if out.ndim == 2:
            out = np.repeat(out[:, :, None], 3, axis=2)
        return (out.transpose(2, 0, 1) / 255).astype(np.float32)

    def read(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Text and mean character probability of each crop (gray or RGB uint8)."""
        buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for i, img in enumerate(images):
            x = self.prepare(img)
            buckets.setdefault(x.shape[-1], []).append((i, x))
        out: list[tuple[str, float]] = [("", 0.0)] * len(images)
        for items in buckets.values():
            batch = np.stack([x for _, x in items])
            for (i, _), res in zip(items, self._translate(batch)):
                out[i] = res
        return out

    def _translate(self, batch: np.ndarray) -> list[tuple[str, float]]:
        """Greedy decoding of one width, as vietocr's translate(): all crops step together
        until each has ended, and the probability averages every character step after
        <sos> (including those after its own <eos>, as vietocr counts them)."""
        runs = [
            self.encoder.run(None, {"image": batch[i : i + ENCODE_BATCH]}) for i in range(0, len(batch), ENCODE_BATCH)
        ]
        cross_k = np.concatenate([r[0] for r in runs], axis=1)
        cross_v = np.concatenate([r[1] for r in runs], axis=1)
        n = len(batch)
        layers, heads, head_dim = self.cache_shape
        # slot 0 of the self-attention cache is a placeholder the decoder skips (no empty tensors)
        past_k = np.zeros((layers, n, heads, 1, head_dim), np.float32)
        past_v = past_k.copy()
        tokens = [np.full(n, SOS, np.int64)]
        probs = [np.ones(n)]
        ended = np.zeros(n, bool)
        while len(tokens) - 1 <= self.max_steps and not ended.all():
            feed = {
                "token": tokens[-1],
                "pos": np.array(len(tokens) - 1, np.int64),
                "past_k": past_k,
                "past_v": past_v,
                "cross_k": cross_k,
                "cross_v": cross_v,
            }
            p, past_k, past_v = self.decoder.run(None, feed)
            best = p.argmax(axis=1).astype(np.int64)
            probs.append(p[np.arange(n), best].astype(np.float64))
            tokens.append(best)
            ended |= best == EOS
        ids = np.stack(tokens, axis=1)
        chars = np.stack(probs, axis=1) * (ids > 3)
        counts = (chars > 0).sum(axis=1)
        mean = np.divide(chars.sum(axis=1), counts, out=np.zeros(n), where=counts > 0)
        out = []
        for j in range(n):
            row = ids[j, 1:].tolist()
            row = row[: row.index(EOS)] if EOS in row else row
            out.append(("".join(self.chars[k] for k in row), float(mean[j])))
        return out


_lock = threading.Lock()
_loaded = False
_reader: VietOCR | None = None


def reader() -> VietOCR | None:
    """The process's VietOCR, loaded on first use; None when disabled or not installed."""
    global _loaded, _reader
    if not _loaded:
        with _lock:
            if not _loaded:
                if VIETOCR_ENABLED and all((VIETOCR_DIR / f).is_file() for f in MODEL_FILES):
                    try:
                        _reader = VietOCR(VIETOCR_DIR, VIETOCR_THREADS)
                    except Exception:
                        log.exception("VietOCR not loaded from %s", VIETOCR_DIR)
                _loaded = True
    return _reader


# --- choosing between the two readings ----------------------------------------


def _tone_count(s: str) -> int:
    return sum(ch in _TONES for ch in unicodedata.normalize("NFD", s))


def _with_case(src: str, marks: str) -> str:
    """marks' letters in src's case (same length), else marks."""
    if len(src) != len(marks):
        return marks
    return "".join(m.upper() if s.isupper() else m.lower() if s.islower() else m for s, m in zip(src, marks))


def _base(s: str) -> str:
    """Each character as its lower-case base letter (marks and đ's stroke gone), one for one."""
    return "".join("d" if c in "đĐ" else unicodedata.normalize("NFD", c)[0].lower() for c in s)


def _blocks(a: str, b: str) -> list[tuple]:
    """Matching of two readings character by character on their base letters (difflib opcodes)."""
    return SequenceMatcher(None, _base(a), _base(b), autojunk=False).get_opcodes()


def _case_of(tess: str, vocr: str) -> str:
    """vocr with the case of each letter that lines up with the same letter in tess ("Lần" + "lân." -> "lần";
    "B.3 kpa." + "B.3kPa." -> "B.3 kPa."); its other letters as VietOCR read them."""
    out = list(vocr)
    for tag, i0, i1, j0, j1 in _blocks(tess, vocr):
        if tag == "equal":
            for s, j in zip(tess[i0:i1], range(j0, j1)):
                out[j] = out[j].upper() if s.isupper() else out[j].lower() if s.islower() else out[j]
    return "".join(out)


def _end_mark(tess: str, vocr: str) -> str:
    """vocr with Tesseract's final mark when VietOCR read the word up to it and left the mark out ("Lần" for
    "lần.", "Nan3" for "NaNs."); not when VietOCR read something else in its place ("10:3" for "10.")."""
    if tess[-1:] not in END_MARKS or not vocr or vocr[-1] in MARKS:
        return vocr
    tag, i0, i1, j0, j1 = _blocks(tess, vocr)[-1]
    if tag == "delete" or (tag == "replace" and i1 - i0 > j1 - j0 and not MARKS & set(vocr[j0:j1])):
        return vocr + tess[-1]
    return vocr


def _drops_marks(tess: str, vocr: str) -> bool:
    """VietOCR's word leaves out a punctuation mark of Tesseract's ("T,;" read "T,"), or a bracket of a pair
    ("[ĐNH]" read "ĐNH"). A mark read as another one ("0.25" as "0,25", "2026.2027" as "2026-2027") or as a
    letter or digit ("II." as "III", "NaN:" as "Nan3": which reading is right is not known) stays; so does a
    repeated mark read once ("K.." as "K.")."""
    tess, vocr = re.sub(r"([.,;:!?])\1+", r"\1", tess), re.sub(r"([.,;:!?])\1+", r"\1", vocr)
    for tag, i0, i1, j0, j1 in _blocks(tess, vocr):
        if tag == "equal" or i1 - i0 <= j1 - j0:
            continue
        if sum(c in MARKS for c in tess[i0:i1]) > sum(not c.isalnum() and not c.isspace() for c in vocr[j0:j1]):
            return True
    return any(p in tess and q in tess[tess.index(p) :] and (p not in vocr or q not in vocr) for p, q in ("[]", "{}"))


def choose(tess: str, conf: float, vocr: str, prob: float) -> str:
    """The word to keep from Tesseract's reading (confidence 0-100) and VietOCR's (probability 0-1)."""
    if not vocr.strip():
        return tess
    # "°" is not in VietOCR's vocabulary: it writes "?" ("A.8,3 ?C." for "A.83°C.").
    if "?" not in tess and 0 < tess.count("°") == vocr.count("?"):
        vocr = vocr.replace("?", "°")
    vocr = _end_mark(tess, vocr)
    if _toneless(tess) == _toneless(vocr):
        # Same letters: VietOCR's tone marks, except that it may not drop one from a word
        # Tesseract was sure of (word crops of short words lose marks: "khí" -> "khi").
        if conf < KEEP_MARKS_CONF or _tone_count(vocr) >= _tone_count(tess):
            return _with_case(tess, vocr)
        return tess
    if conf >= TESS_SURE or prob < VOCR_SURE:
        return tess
    if "?" in vocr and "?" not in tess:
        return tess
    if any(vocr.count(ch) < tess.count(ch) for ch in MATH_CHARS & set(tess)):
        return tess
    if _drops_marks(tess, vocr):
        return tess
    return _case_of(tess, vocr) if conf >= CASE_CONF else vocr


def word_crops(lines: list, gray: np.ndarray) -> list[np.ndarray]:
    """One crop per word: the word's columns, the line's rows, padded by a quarter of the line height."""
    h, w = gray.shape[:2]
    out = []
    for ln in lines:
        y0, y1 = min(wd.y0 for wd in ln.words), max(wd.y1 for wd in ln.words)
        pad = int(0.25 * (y1 - y0))
        for wd in ln.words:
            out.append(gray[max(y0 - pad, 0) : min(y1 + pad, h), max(wd.x0 - pad, 0) : min(wd.x1 + pad, w)])
    return out


def refine_words(lines: list, gray: np.ndarray, reader: VietOCR) -> int:
    """Re-read Tesseract's words (app.ocr Line/Word, before fix_lines) with VietOCR and keep
    the better reading of each. Returns how many words changed."""
    words = [wd for ln in lines for wd in ln.words]
    crops = word_crops(lines, gray)
    todo = [i for i, c in enumerate(crops) if c.shape[0] >= 2 and c.shape[1] >= 2]
    if not todo:
        return 0
    reads = reader.read([crops[i] for i in todo])
    changed = 0
    for i, (text, prob) in zip(todo, reads):
        wd = words[i]
        new = choose(wd.text, wd.conf, text, prob)
        if new != wd.text:
            wd.text = new
            changed += 1
    return changed
