"""Formula regions of OCR pages: image -> LaTeX -> Word text or equation.

Tesseract loses what is not a line of text: superscripts and decimal commas
("6,42.10^{-21} J" reads "642.107'J."), fractions and roots become garbage.
The formula model pix2text-mfr-1.5 (MIT; TrOCR: DeiT encoder + BPE decoder,
ONNX files in MFR_DIR) reads a region's crop into LaTeX (tidy() cleans it),
and decide() turns a reading into a MathDecision:

- kind "text": a one-line reading (numbers with exponents, units, v_O, °C)
  as runs of plain/superscript/subscript text;
- kind "eq": a two-dimensional formula (fraction, root, bar) for a Word
  equation, plus the plain text after it (unit, final full stop).

Precision comes first: a reading that may be wrong gives None and the caller
keeps the OCR text. A reading is dropped for low confidence, for symbols the
model writes when it cannot read the ink (\\theta \\mu \\Gamma \\ldots ...),
for leading junk or a script without base ("^{3km}"), for a subscript right
after a digit ("4_{VO}" for "4v_O"), for words it cannot spell ("phut"), and
when Tesseract's digits (number by number, outside fractions and roots too) or
first letter's case disagree with it.
read_regions() reads each region twice (crop cut to the ink, then with 3 px of
white; with 8 px when the two disagree; without its last short word, a unit or
mark, when they still do not) and keeps a decision two readings agree on; a
first reading that is plain text equal to Tesseract's ("6 kPa.", read right by
OCR) is kept without a second one, since it writes the OCR text anyway.
Before that, a reading gets what the ink tells better than the model: a
coefficient's letter read into a subscript ("4_{VO}") is written out (4v_{O})
where the ink shows a small letter on the baseline and a lowered glyph after
the digit, and a first letter whose case only its height tells (v, o, x...)
takes the case measured on the glyph. On 522 formula boxes of two physics
exams (one real scan, a native PDF and two simulated scans of it) it kept 399,
none wrong.

Only a careful conversion ("Đọc kỹ", app/convert.py) uses the model; the default
reads superscripts and subscripts without it (app/scriptocr.py).
Without the model files (or with P2W_MFR=0) read_regions() returns Nones.
decoder_model.onnx may be replaced by its two halves (scripts/split_mfr_decoder.py:
the cross-attention keys and values of a crop computed once, not at every step),
about a quarter less time per crop with the same readings.
Decoding needs only the vocabulary of tokenizer.json (byte-level BPE), and the
crops are resized as Pillow's BICUBIC does, so neither tokenizers nor Pillow is
needed; onnxruntime runs the model.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .settings import MFR_DIR, MFR_ENABLED, MFR_THREADS

log = logging.getLogger("pdf2word.mathocr")

MODEL_FILES = ("encoder_model.onnx", "decoder_model.onnx", "tokenizer.json")
SPLIT_FILES = ("cross_kv.onnx", "decoder_step.onnx")  # used in place of decoder_model.onnx when both are there
BOS, EOS, PAD = 1, 2, 0
SIZE = 384  # the encoder's input side; crops are stretched to a square
MAX_TOKENS = 200  # a reading that does not end by then is dropped
BATCH = 16  # crops per encoder/decoder run
# The encoder reads the crops of a batch one by one: the same readings, ~10 % less time and half the memory (one
# crop's activations stay in cache); a dynamically quantized INT8 encoder needs it (it scales its activations over
# the whole batch: a crop's reading would depend on the other crops of its page).
ENCODE_ONE = True
MAX_CROP_PIXELS = 2_000_000  # larger regions are not formulas (a formula at 300 dpi is < 300 px high)

MIN_CONF = 0.85  # mean token probability needed by decide() (vote() asks less of each of two readings)


# --- Pillow's BICUBIC resize (8-bit fixed point), as the model's processor does ---


def _cubic(x: np.ndarray) -> np.ndarray:
    a = -0.5
    x = np.abs(x)
    return np.where(
        x < 1.0,
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1,
        np.where(x < 2.0, (((x - 5) * x + 8) * x - 4) * a, 0.0),
    )


@lru_cache(maxsize=512)
def _weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resampling filter, coefficients scaled by 2^22 and rounded as Pillow does: the first input pixel of each output
    pixel [out] and its taps [out, k] (zero past the filter's end). Kept per output pixel, not as a dense [out, in]
    matrix: the cache stays small (≤ ~0.1 MB an entry) however large the crops are."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    rows = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        k = _cubic((np.arange(xmin, xmax) - center + 0.5) / filterscale)
        total = k.sum()
        if total != 0.0:
            k = k / total
        rows.append((xmin, np.trunc(k * (1 << 22) + np.where(k < 0, -0.5, 0.5))))
    taps = np.zeros((out_size, max(len(k) for _, k in rows)), np.int32)
    for xx, (_, k) in enumerate(rows):
        taps[xx, : len(k)] = k
    return np.array([xmin for xmin, _ in rows], np.intp), taps


def _resample(img: np.ndarray, weights: tuple[np.ndarray, np.ndarray], axis: int) -> np.ndarray:
    starts, taps = weights
    src = np.moveaxis(img, axis, -1)
    acc = np.zeros(src.shape[:-1] + (len(starts),), np.int64)  # Pillow's sums in integers: exact, in any order
    last = src.shape[-1] - 1
    for k in range(taps.shape[1]):
        acc += src[..., np.minimum(starts + k, last)] * taps[:, k].astype(np.int64)
    acc = np.moveaxis(acc, -1, axis)
    return np.clip((acc + (1 << 21)) >> 22, 0, 255).astype(np.uint8)


def resize_bicubic(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pillow's Image.resize(BICUBIC) of a uint8 grey image: horizontal pass, then vertical."""
    h, w = gray.shape
    out = gray
    if w != width:
        out = _resample(out, _weights(w, width), 1)
    if h != height:
        out = _resample(out, _weights(h, height), 0)
    return out


# --- tokenizer: byte-level BPE; decoding only needs the vocabulary ---


@lru_cache(maxsize=1)
def _byte_map() -> dict[str, int]:
    """GPT-2's bytes_to_unicode, inverted: printable stand-in character -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Vocab:
    """tokenizer.json decoder: the text of tokenizers' decode(ids, skip_special_tokens=True)."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.tokens = {i: s for s, i in data["model"]["vocab"].items()}
        self.special = {t["id"] for t in data.get("added_tokens", []) if t.get("special")}

    def decode(self, ids) -> str:
        byte_map = _byte_map()
        text = "".join(self.tokens.get(int(i), "") for i in ids if int(i) not in self.special)
        return bytes(byte_map[c] for c in text if c in byte_map).decode("utf-8", "replace")


# --- the model ---


class FormulaReader:
    """pix2text-mfr-1.5 on onnxruntime: grey crops -> (LaTeX, mean token probability).

    Greedy decoding without a key/value cache (decoder_model.onnx has none);
    a reading that does not end within MAX_TOKENS gets confidence 0.
    """

    def __init__(self, model_dir: Path, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        model_dir = Path(model_dir)
        self.encoder = ort.InferenceSession(str(model_dir / "encoder_model.onnx"), so, providers=providers)
        # decoder_model.onnx split in two (scripts/split_mfr_decoder.py): the cross-attention keys and values of a crop
        # computed once (cross_kv.onnx), not again at every step of decoder_model.onnx
        self.cross = self.decoder = None
        if all((model_dir / f).is_file() for f in SPLIT_FILES):
            self.cross = ort.InferenceSession(str(model_dir / SPLIT_FILES[0]), so, providers=providers)
            self.step = ort.InferenceSession(str(model_dir / SPLIT_FILES[1]), so, providers=providers)
        else:
            self.decoder = ort.InferenceSession(str(model_dir / "decoder_model.onnx"), so, providers=providers)
        self.vocab = Vocab(model_dir / "tokenizer.json")

    @staticmethod
    def pixels(gray: np.ndarray) -> np.ndarray:
        """[3, 384, 384] float input of a uint8 grey crop, stretched to the square."""
        img = resize_bicubic(gray, SIZE, SIZE).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return np.repeat(img[None], 3, axis=0)

    def read(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        """(LaTeX, mean token probability) of each grey crop. Crops of similar shape are decoded together."""
        order = sorted(range(len(images)), key=lambda i: images[i].shape[1] / max(images[i].shape[0], 1))
        out: list[tuple[str, float]] = [("", 0.0)] * len(images)
        for k in range(0, len(order), BATCH):
            chunk = order[k : k + BATCH]
            for i, r in zip(chunk, self._read([images[i] for i in chunk])):
                out[i] = r
        return out

    def _read(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        if not images:
            return []
        px = np.stack([self.pixels(im) for im in images]).astype(np.float32)
        if ENCODE_ONE:  # a dynamically quantized encoder scales its activations per batch: one crop at a time
            hidden = np.concatenate([self.encoder.run(None, {"pixel_values": px[k : k + 1]})[0] for k in range(len(px))])
        else:
            hidden = self.encoder.run(None, {"pixel_values": px})[0]
        n = len(images)
        seqs = [[BOS] for _ in range(n)]
        logp = np.zeros(n)
        finished = np.zeros(n, bool)
        active = np.arange(n)
        if self.cross is not None:
            names = [o.name for o in self.cross.get_outputs()]
            kv_all = dict(zip(names, self.cross.run(None, {"encoder_hidden_states": hidden})))
        feed, fed = {}, -1  # inputs of the active rows, rebuilt when a row ends
        for _ in range(MAX_TOKENS):
            ids = np.array([seqs[i] for i in active], dtype=np.int64)  # active sequences have one length
            if fed != len(active):
                whole = len(active) == n
                if self.cross is not None:
                    feed = {k: v if whole else np.ascontiguousarray(v[active]) for k, v in kv_all.items()}
                else:
                    feed = {"encoder_hidden_states": hidden if whole else hidden[active]}
                fed = len(active)
            run = self.step if self.cross is not None else self.decoder
            logits = run.run(None, {"input_ids": ids, **feed})[0][:, -1, :]
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            nxt = p.argmax(axis=1)
            for row, i in enumerate(active):
                seqs[i].append(int(nxt[row]))
                logp[i] += np.log(p[row, nxt[row]] + 1e-12)
            ended = nxt == EOS
            finished[active[ended]] = True
            active = active[~ended]
            if not len(active):
                break
        out = []
        for i in range(n):
            toks = seqs[i][1:]
            if finished[i]:
                toks = toks[:-1]
            conf = float(np.exp(logp[i] / max(len(toks), 1))) if finished[i] else 0.0
            out.append((self.vocab.decode(toks).strip(), conf))
        return out


_reader: FormulaReader | None = None
_reader_failed = False
_reader_lock = threading.Lock()


# The formula model wanted by the conversion running in this thread (convert() skips it for long scans).
_wanted: ContextVar[bool] = ContextVar("formulas_wanted", default=True)


def installed() -> bool:
    """The model's files are there and it is not turned off (P2W_MFR=0) – whether or not this conversion uses it."""
    if not MFR_ENABLED:
        return False
    have = {f for f in MODEL_FILES + SPLIT_FILES if (MFR_DIR / f).is_file()}
    return set(MODEL_FILES) <= have or {"encoder_model.onnx", "tokenizer.json", *SPLIT_FILES} <= have


def available() -> bool:
    """The conversion running in this thread can use the model: installed, not skipped, not broken (broken())."""
    return _wanted.get() and not _reader_failed and installed()


@contextmanager
def skipped(skip: bool = True) -> Iterator[None]:
    """Formula reading off for the conversion running in this thread while the block runs (other
    conversions running at the same time in other threads keep theirs)."""
    token = _wanted.set(not skip)
    try:
        yield
    finally:
        _wanted.reset(token)


def get_reader() -> FormulaReader | None:
    """The shared reader, loaded on first use; None when off, skipped for this conversion, missing or broken."""
    global _reader, _reader_failed
    if not _wanted.get():
        return None
    if _reader is not None or _reader_failed or not available():
        return _reader
    with _reader_lock:
        if _reader is None and not _reader_failed:
            try:
                _reader = FormulaReader(MFR_DIR, MFR_THREADS)
            except Exception:  # onnxruntime missing, damaged files: convert without formulas
                log.exception("formula model not loaded")
                _reader_failed = True
    return _reader


def broken() -> None:
    """Called where reading formulas failed (damaged files, a tokenizer of another revision): the formula model is
    off for the rest of the process, formulas stay OCR text."""
    global _reader, _reader_failed
    log.exception("formula reading failed: off until the service restarts")
    with _reader_lock:
        _reader, _reader_failed = None, True


# --- LaTeX: a small parser for the model's output ---
# Nodes: ["t", text] symbol (character, control symbol, control word without
# arguments; "\\ " is a real space), ["g", items] group, ["f", name, opt, args]
# command with arguments, ["s", base, sub, sup] scripts (base: node or None,
# sub/sup: items or None).

WRAPPERS = {
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathnormal", "boldsymbol", "bm", "operatorname", "mathbfit",
    "pmb", "text", "textrm", "textit", "textbf", "textup", "textnormal", "textsf", "texttt", "mbox", "hbox", "emph",
}  # fmt: skip
SWITCHES = {"rm", "bf", "it", "sf", "tt", "cal"}
SPACES = {"quad", "qquad", "enspace", "thinspace", "medspace", "thickspace", "space", "nobreakspace"}
SPACE_SYMBOLS = {"\\,", "\\;", "\\:", "\\>", "\\ ", "~"}
DROP = {
    "left", "right", "middle", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl",
    "Biggr", "bigm", "Bigm", "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits", "nolimits",
    "nonumber", "notag", "negthinspace", "negmedspace", "negthickspace", "relax", "strut", "mathstrut",
}  # fmt: skip
DROP_ARG = {"hspace", "vspace", "phantom", "hphantom", "vphantom", "label", "tag"}
ALIAS = {"dfrac": "frac", "tfrac": "frac", "cfrac": "frac", "leq": "le", "geq": "ge", "neq": "ne", "leqslant": "le",
         "geqslant": "ge", "rightarrow": "to", "varrho": "rho", "lbrace": "{", "rbrace": "}", "degree": "circ",
         "textdegree": "circ", "lvert": "|", "rvert": "|", "vert": "|", "mid": "|"}
ARITY = {"frac": 2, "binom": 2, "overset": 2, "underset": 2, "stackrel": 2, "sqrt": 1, "overline": 1, "underline": 1,
         "bar": 1, "vec": 1, "overrightarrow": 1, "overleftarrow": 1, "hat": 1, "widehat": 1, "tilde": 1, "dot": 1,
         "ddot": 1, "mathcal": 1, "mathbb": 1, "mathfrak": 1, "mathscr": 1, "boxed": 1, "cancel": 1, "underbrace": 1,
         "overbrace": 1, "xrightarrow": 1}
TWO_D = {"frac", "binom", "sqrt", "overline", "bar", "vec", "overrightarrow", "overset", "underset", "stackrel"}

# Symbols a decision may hold, with their text form. Other commands (\theta \mu \Gamma
# \psi \ldots \mathcal ...) are what the model writes for ink it cannot read.
SYMBOL_TEXT = {
    "\\alpha": "α", "\\beta": "β", "\\Delta": "Δ", "\\lambda": "λ", "\\pi": "π", "\\rho": "ρ", "\\varphi": "φ",
    "\\phi": "φ", "\\omega": "ω", "\\Omega": "Ω", "\\varepsilon": "ε", "\\epsilon": "ε",
    "\\cdot": ".", "\\times": "×", "\\pm": "±", "\\mp": "∓", "\\approx": "≈", "\\le": "≤", "\\ge": "≥", "\\ne": "≠",
    "\\to": "→", "\\Rightarrow": "⇒", "\\infty": "∞", "\\%": "%", "\\{": "{", "\\}": "}", "\\ ": " ",
    "\\sin": "sin", "\\cos": "cos", "\\tan": "tan", "\\cot": "cot", "\\log": "log", "\\ln": "ln", "\\lg": "lg",
}
GREEK = {k for k, v in SYMBOL_TEXT.items() if unicodedata.category(v[0]) == "Ll" and v.isalpha() and len(v) == 1}
GREEK |= {"\\Delta", "\\Omega"}
PLAIN = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=/,.:;()[]|°<>!%")
# words of three letters or more the model may write outside scripts (units, functions);
# others are text it cannot read ("phút" -> "phut")
WORDS = {"mol", "atm", "kPa", "MPa", "hPa", "GPa", "MeV", "keV", "GeV", "kHz", "MHz", "GHz", "min", "rad", "sin",
         "cos", "tan", "cot", "log", "lim", "max", "mmHg", "cal", "kcal", "kWh", "mAh", "cd", "lux", "dB"}


def _tokens(s: str) -> list[str]:
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
            if m:
                out.append("\\" + m.group(1))
                i += len(m.group(0))
            else:
                out.append(s[i : i + 2] if i + 1 < n else "\\")
                i += 2
            continue
        if c.isspace():
            if not out or out[-1] != " ":
                out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return out


class _Parser:
    def __init__(self, toks: list[str]):
        self.t = toks
        self.i = 0
        self.ok = True  # False: unbalanced braces or a script without argument

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def seq(self, stop: str | None = None) -> list:
        items = []
        while True:
            tok = self.peek()
            if tok is None:
                if stop is not None:
                    self.ok = False
                return items
            if tok == stop:
                self.i += 1
                return items
            if tok == "}":  # closes nothing
                self.ok = False
                self.i += 1
                continue
            if tok in ("^", "_"):
                self.i += 1
                arg = self.arg()
                if arg is None:
                    self.ok = False
                    return items
                slot = 2 if tok == "_" else 3
                last = items[-1] if items else None
                if last is not None and last[0] == "s" and last[slot] is None:
                    last[slot] = arg
                else:
                    node = ["s", items.pop() if items else None, None, None]
                    node[slot] = arg
                    items.append(node)
                continue
            node = self.atom()
            if node is not None:
                items.append(node)

    def blanks(self) -> None:
        """Skips the blanks the model puts between all tokens."""
        while self.peek() == " ":
            self.i += 1

    def arg(self) -> list | None:
        self.blanks()
        tok = self.peek()
        if tok is None or tok in ("}", "^", "_"):
            return None
        if tok == "{":
            self.i += 1
            return self.seq("}")
        node = self.atom()
        return [] if node is None else [node]

    def atom(self):
        tok = self.t[self.i]
        self.i += 1
        if tok == " ":
            return None  # the model puts a blank between all tokens; its real spaces are ~ \, \;
        if tok == "{":
            return ["g", self.seq("}")]
        if tok in SPACE_SYMBOLS:
            return ["t", "\\ "]
        if not tok.startswith("\\") or len(tok) == 1:
            return ["t", tok]
        name = tok[1:]
        if not name[0].isalpha():
            return None if name == "!" else ["t", tok]
        name = ALIAS.get(name, name)
        if name in ("{", "}", "|"):
            return ["t", "\\" + name if name != "|" else "|"]
        if name in SPACES:
            return ["t", "\\ "]
        if name in DROP:
            if name in ("left", "right", "middle"):
                self.blanks()
                if self.peek() == ".":  # "\left ." (no bracket)
                    self.i += 1
            return None
        if name in DROP_ARG:
            self.arg()
            return ["t", "\\ "] if name.startswith("hspace") else None
        if name in SWITCHES:
            return None
        if name in WRAPPERS:
            return ["g", self.arg() or []]
        if name == "circ":
            return ["t", "°"]
        if name == "sqrt":
            opt = None
            self.blanks()
            if self.peek() == "[":  # "\sqrt [ 3 ] { x }": the root's degree
                self.i += 1
                opt = self.seq("]")
            arg = self.arg()
            if arg is None:
                self.ok = False
                arg = []
            return ["f", "sqrt", opt, [arg]]
        if name in ARITY:
            args = []
            for _ in range(ARITY[name]):
                a = self.arg()
                if a is None:
                    self.ok = False
                    a = []
                args.append(a)
            return ["f", name, None, args]
        return ["t", "\\" + name]


def _simplify(items: list | None, edges: bool = True) -> list | None:
    """Drops empty and redundant groups, unwraps lowered scripts (x_{_{A}}), degree scripts -> °; one real space
    at most in a row, none at the ends of a formula or argument (edges)."""
    if items is None:
        return None
    out = []
    for node in items:
        kind = node[0]
        if kind == "g":
            out.extend(_simplify(node[1], edges=False))  # a group outside an argument prints as its content
            continue
        if kind == "f":
            node = ["f", node[1], _simplify(node[2]), [_simplify(a) for a in node[3]]]
        elif kind == "s":
            base = node[1]
            if base == ["t", "\\ "]:
                out.append(base)  # "27\ ^{o}C": the space stays, the script has no base
                base = None
            if base is not None:
                if base[0] == "g":
                    inner = _simplify(base[1])
                    if not inner:
                        base = None
                    elif len(inner) == 1:
                        base = inner[0]
                    elif all(n[0] == "t" and not n[1].startswith("\\") for n in inner):
                        out.extend(inner[:-1])  # {10}^{-21} prints as 10^{-21}
                        base = inner[-1]
                    else:
                        base = ["g", inner]
                else:
                    base = (_simplify([base]) or [None])[0]
            sub, sup = _lowered(_simplify(node[2])), _lowered(_simplify(node[3]))
            if sub is not None and not sub:
                sub = None
            if sup is not None and not sup:
                sup = None
            if sup is not None and len(sup) == 1 and sup[0] in (["t", "o"], ["t", "0"]) and _next_is(items, node, "CF"):
                sup = [["t", "°"]]  # 27^{o}C
            if sup is not None and len(sup) == 1 and sup[0] == ["t", "°"]:
                if base is not None:
                    out.append(base if sub is None else ["s", base, sub, None])
                elif sub is not None:
                    out.append(["s", None, sub, None])
                out.append(["t", "°"])
                continue
            if sub is None and sup is None:
                if base is not None:
                    out.append(base)
                continue
            node = ["s", base, sub, sup]
        out.append(node)
    tidy = []
    for node in out:
        if node == ["t", "\\ "] and (tidy[-1:] == [node] or edges and not tidy):
            continue
        tidy.append(node)
    while edges and tidy and tidy[-1] == ["t", "\\ "]:
        tidy.pop()
    return tidy


def _next_is(items, node, chars) -> bool:
    """The node after node in items is (a group starting with) one of chars."""
    k = next(i for i, n in enumerate(items) if n is node)
    nxt = items[k + 1] if k + 1 < len(items) else None
    while nxt is not None and nxt[0] == "g":
        nxt = nxt[1][0] if nxt[1] else None
    return nxt is not None and nxt[0] == "t" and nxt[1] in chars


def _lowered(items):
    """x_{_{A}} / x_{{}_{A}}: a script holding only a script of nothing is the script."""
    while items and len(items) == 1 and items[0][0] == "s" and items[0][1] is None:
        sub, sup = items[0][2], items[0][3]
        if (sub is None) == (sup is None):
            break
        items = sub if sub is not None else sup
    return items


def _ser(items) -> str:
    out = ""
    for node in items or []:
        kind = node[0]
        if kind == "t":
            piece = node[1]
        elif kind == "g":
            piece = "{" + _ser(node[1]) + "}"
        elif kind == "f":
            piece = "\\" + node[1] + ("[" + _ser(node[2]) + "]" if node[2] else "")
            piece += "".join("{" + _ser(a) + "}" for a in node[3])
        else:
            piece = "" if node[1] is None else _ser([node[1]])
            if node[2] is not None:
                piece += "_{" + _ser(node[2]) + "}"
            if node[3] is not None:
                piece += "^{" + _ser(node[3]) + "}"
        if out and re.search(r"\\[A-Za-z]+$", out) and piece[:1].isalpha():
            out += " "
        out += piece
    return out


def _parse(tex: str):
    p = _Parser(_tokens(tex))
    items = p.seq()
    return _simplify(items), p.ok


def tidy(raw: str) -> str:
    """Model LaTeX -> plain LaTeX: font wrappers and blanks between tokens gone, real spaces (~ \\, \\; \\quad)
    kept as "\\ ", ^{\\circ} -> °, \\dfrac -> \\frac, x_{_{A}} -> x_{A}, \\sqrt2 -> \\sqrt{2}, empty and redundant
    braces dropped. tidy(tidy(s)) == tidy(s)."""
    items, _ = _parse(raw or "")
    return _ser(items)


# --- decisions ---


@dataclass
class MathDecision:
    kind: str  # "text" or "eq"
    parts: list[tuple[str, str | None]] = field(default_factory=list)  # text: (text, None | "super" | "sub")
    latex: str = ""  # eq: LaTeX of the Word equation
    tail: str = ""  # eq: plain text after the equation (unit, punctuation)
    conf: float = 0.0
    source: str = ""  # tidy LaTeX the decision was made from

    def key(self) -> tuple:
        """Identity for votes: readings that differ only in spacing agree."""
        return (self.kind, re.sub(r"\\ | ", "", self.as_latex()))

    def as_latex(self) -> str:
        """The decision written back as LaTeX (for comparisons)."""
        if self.kind == "eq":
            return self.latex + self.tail
        out = ""
        for text, script in self.parts:
            out += text if script is None else ("^{%s}" if script == "super" else "_{%s}") % text
        return out


_LABEL = re.compile(r"^\s*\*?[A-D]\s*[.)]\s*")


def _walk(items):
    for node in items or []:
        yield node
        if node[0] == "g":
            yield from _walk(node[1])
        elif node[0] == "f":
            yield from _walk(node[2])
            for a in node[3]:
                yield from _walk(a)
        elif node[0] == "s":
            if node[1] is not None:
                yield from _walk([node[1]])
            yield from _walk(node[2])
            yield from _walk(node[3])


def _text(items) -> str | None:
    """Plain text of items without scripts or 2-D commands; None if they have any."""
    out = ""
    for node in items or []:
        if node[0] == "g":
            inner = _text(node[1])
            if inner is None:
                return None
            out += inner
        elif node[0] == "t":
            sym = node[1]
            if sym.startswith("\\"):
                if sym not in SYMBOL_TEXT:
                    return None
                out += SYMBOL_TEXT[sym]
            else:
                out += sym
        else:
            return None
    return out


def _parts(items) -> list[tuple[str, str | None]] | None:
    """Runs of plain / superscript / subscript text of a one-line formula; None for nested scripts or 2-D."""
    parts = []
    for node in items:
        if node[0] == "s":
            base = [] if node[1] is None else _parts([node[1]])
            if base is None:
                return None
            parts += base
            for slot, script in ((2, "sub"), (3, "super")):
                if node[slot] is not None:
                    text = _text(node[slot])
                    if text is None:
                        return None
                    parts.append((text, script))
        else:
            text = _text([node])
            if text is None:
                return None
            parts.append((text, None))
    merged = []
    for text, script in parts:
        if merged and merged[-1][1] == script:
            merged[-1] = (merged[-1][0] + text, script)
        elif text:
            merged.append((text, script))
    return merged


# Letters one reader may write for a digit the other reads ("2O0O" for 2000, "S" for 5)
_DIGIT_LOOK = {"0": "oODQ", "1": "lIiLtjf|", "2": "zZ", "4": "A", "5": "sS", "6": "bG", "7": "T", "8": "B", "9": "gq"}


def _skeleton(items, script: bool = False, out: list | None = None) -> list[tuple[str, str]]:
    """What the digit check compares of a reading, in order: ("D", d) a digit outside scripts and 2-D parts,
    ("S", c) a character of a script, ("L", c) a letter, ("G", "°") a degree sign, ("W", "") a 2-D part
    (fraction, root: Tesseract reads its ink in any order). Other signs are left out."""
    out = [] if out is None else out
    for node in items or []:
        if node[0] == "t":
            for c in SYMBOL_TEXT.get(node[1], node[1]):
                if c.isdigit() or (script and not c.isspace()):
                    out.append(("S" if script else "D", c))
                elif c.isalpha():
                    out.append(("L", c))
                elif c == "°":
                    out.append(("G", c))
        elif node[0] == "g":
            _skeleton(node[1], script, out)
        elif node[0] == "f":
            out.append(("W", ""))
        elif node[0] == "s":
            if node[1] is not None:
                _skeleton([node[1]], script, out)
            for slot in (2, 3):
                _skeleton(node[slot], True, out)
    return out


def _ocr_skeleton(ocr: str) -> list[tuple[str, str]]:
    """The OCR text as the digit check sees it: ("D", d) digits, ("L", c) letters, ("X", c) other signs and a digit
    between two letters (a letter misread: "ó6ms" for "m/s")."""
    out = []
    for k, c in enumerate(ocr):
        if c.isspace() or c in ".,;:":
            continue
        if c.isdigit() and not (0 < k < len(ocr) - 1 and ocr[k - 1].isalpha() and ocr[k + 1].isalpha()):
            out.append(("D", c))
        else:
            out.append(("L" if c.isalpha() else "X", c))
    return out


_MATH_SIGNS = set("/=+-–−()%°<>×±")  # signs Tesseract reads as they are, not for a digit


def _pairs(r: tuple[str, str], o: tuple[str, str], degree: bool) -> bool:
    (kr, cr), (ko, co) = r, o
    if kr == "D":  # a digit of the reading is Tesseract's digit, or a letter or mark Tesseract read for it
        return (ko == "D" and co == cr) or (ko == "L" and co in _DIGIT_LOOK.get(cr, "")) or (ko == "X" and co not in _MATH_SIGNS)
    if kr == "L":  # a letter is anything but a digit, unless the digit looks like it
        return ko != "D" or cr in _DIGIT_LOOK.get(co, "")
    if kr == "G":  # a degree sign Tesseract read as "0" or "9" (unless it read one as "°")
        return ko != "D" or (not degree and co in "09")
    return kr == "S"  # Tesseract reads scripts as anything


def _digits_agree(reading: list[tuple[str, str]], ocr: list[tuple[str, str]]) -> bool:
    """Every digit of the reading outside scripts is a digit Tesseract read in the same order (or a letter or sign
    it read in its place), and every digit Tesseract read is one of the reading's digits, scripts, degree signs or
    look-alike letters, or lies in a 2-D part. So a digit the model left out or added is caught ("200" for OCR
    "2000", "500 00" for "500 000", "11111" for "1111", "2ω" for "20"), while Tesseract's exponent read as digits
    ("642.107'J." for 6,42.10^{-21} J.) and its letters read for digits ("t_{2}" read "1,") are not."""
    n, m = len(reading), len(ocr)
    degree = ("X", "°") in ocr
    seen, todo = set(), [(0, 0)]
    while todo:
        i, j = todo.pop()
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if i == n and j == m:
            return True
        if i < n and reading[i][0] != "D":
            todo.append((i + 1, j))  # nothing of Tesseract's for it (an empty 2-D part)
        if j < m and ocr[j][0] != "D":
            todo.append((i, j + 1))  # a letter or sign Tesseract read in excess
        if i < n and j < m:
            if reading[i][0] == "W":
                todo.append((i, j + 1))  # Tesseract's reading of the 2-D part
            elif _pairs(reading[i], ocr[j], degree):
                todo.append((i + 1, j + 1))
    return False


def _letters(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if "a" <= c.lower() <= "z")


def _bad_case(parts, ocr: str) -> bool:
    """The first letter read by the model and by Tesseract in different case (o/O, v/V): one of them is wrong.
    Only the first: Tesseract cannot see the case of small subscripts."""
    if not parts or parts[0][1] is not None:
        return False
    mine, seen = _letters(parts[0][0])[:1], _letters(ocr)[:1]
    return bool(mine and seen) and mine != seen and mine.lower() == seen.lower()


def _junk(items) -> str | None:
    """Why a parsed reading cannot be used (any kind), or None."""
    if not items:
        return "empty"
    first = items[0]
    if first[0] == "s" and first[1] is None:
        return "script without base"  # "^{3km}": the whole crop read as an exponent
    if first[0] == "t" and first[1] in (".", ",", "'", '"', "`", ")", "]", "=", "/", ":", ";"):
        return "leading junk"
    if first[0] == "t" and first[1] == "-" and not (len(items) > 1 and items[1][0] == "t" and items[1][1].isdigit()):
        return "leading minus"
    for node in _walk(items):
        if node[0] == "t":
            sym = node[1]
            if sym.startswith("\\") and sym not in SYMBOL_TEXT:
                return f"symbol {sym}"
            if not sym.startswith("\\") and sym not in PLAIN:
                return f"character {sym!r}"
        elif node[0] == "f" and node[1] not in TWO_D:
            return f"command \\{node[1]}"
        elif node[0] == "s":
            base = node[1]
            if base is not None and base[0] == "t" and base[1] in (".", ",", "=", "+", "-", "/", ":", ";", "(", "\\ "):
                return "script after punctuation"
            if node[2] is not None and base is not None and base[0] == "t" and base[1].isdigit():
                return "subscript after a digit"  # "4_{VO}": the model's reading of "4v_O" (see coefficients())
            for slot in (2, 3):
                if node[slot] is not None and not node[slot]:
                    return "empty script"
    return None


def _unknown_word(items) -> str | None:
    """A run of three letters or more outside scripts that is not a unit or function name."""
    run = ""
    for node in (items or []) + [["t", " "]]:
        letter = None
        if node[0] == "t" and len(node[1]) == 1 and node[1].isalpha():
            letter = node[1]
        elif node[0] == "s" and node[1] is not None and node[1][0] == "t" and len(node[1][1]) == 1 and node[1][1].isalpha():
            letter = node[1][1]  # "mol^{-1}": the base letter ends the run
        if letter:
            run += letter
        if not letter or node[0] == "s":
            if len(run) >= 3 and run not in WORDS:
                return run
            run = ""
        if node[0] == "g":
            found = _unknown_word(node[1])
        elif node[0] == "f":
            found = next((w for a in node[3] for w in [_unknown_word(a)] if w), None)
        else:
            found = None
        if found:
            return found
    return None


def _split_tail(items):
    """(equation items, tail text): a unit and the final punctuation after the last 2-D part leave the equation,
    and so does a closing bracket at the end that opens nowhere in it (the sentence's: "(V=\\frac{4}{3}\\pi R^3)")."""
    last = max((i for i, n in enumerate(items) if any(m[0] == "f" and m[1] in TWO_D for m in _walk([n]))), default=-1)
    rest = _text(items[last + 1 :])
    m = re.fullmatch(r"\s*([A-Za-zΩ°%/]*)\s*([.,;:]*)", rest) if rest is not None else None
    if m:
        return items[: last + 1], (" " + m.group(1) if m.group(1) else "") + m.group(2)
    unopened = sum(n == ["t", ")"] for n in _walk(items)) - sum(n == ["t", "("] for n in _walk(items))
    k = len(items)
    while k > last + 1 and (
        items[k - 1] in (["t", "."], ["t", ","], ["t", ";"], ["t", ":"], ["t", "\\ "])
        or items[k - 1] == ["t", ")"] and unopened > 0
    ):
        unopened -= items[k - 1] == ["t", ")"]
        k -= 1
    return items[:k], (_text(items[k:]) or "").strip()


def _respace(parts: list[tuple[str, str | None]], ocr: str) -> list[tuple[str, str | None]]:
    """Spaces of a one-line reading: none before . , ; : ) or after (; a space between a number and the word after
    it where Tesseract saw one ("6kPa" + OCR "6 kPa" -> "6 kPa": the model often leaves it out)."""

    def unit_space(m):
        digit, word = m.group(1), m.group(2)
        return digit + (" " if re.search(re.escape(digit) + r"\s+" + re.escape(word), ocr) else "") + word

    out = []
    for k, (text, script) in enumerate(parts):
        if script is None:
            text = re.sub(r" +", " ", text)
            text = re.sub(r" (?=[.,;:)])|(?<=\() ", "", text)
            text = re.sub(r"(\d)([A-Za-z°Ω][A-Za-z°Ω/]*)", unit_space, text)
            m = re.match(r"([A-Za-z°Ω][A-Za-z°Ω/]*)", text)
            if m and out and out[-1][1] == "super" and out[-1][0][-1:].isdigit() and re.search(r"\s" + re.escape(m.group(1)), ocr):
                text = " " + text  # 10^{4}m/s: Tesseract's exponent digits are not the model's
            if k == 0:
                text = text.lstrip()
            if k == len(parts) - 1:
                text = text.rstrip()
        if text:
            out.append((text, script))
    return out


def _end_punct(last: str, ocr: str) -> str:
    """The final . , ; : of the OCR text when the reading left it out (the model does not always read it), if
    the character before it is the reading's last one ("6 kPa." for "6kPa"; not "x," for x_{A})."""
    ocr = ocr.strip()
    if len(ocr) >= 2 and ocr[-1] in ".,;:" and last and last[-1] not in ".,;:" and ocr[-2].lower() == last[-1].lower():
        return ocr[-1]
    return ""


def judge(tex: str, conf: float, ocr_text: str = "", min_conf: float = MIN_CONF) -> tuple[MathDecision | None, str]:
    """decide() with the reason of a rejection ("ok" when accepted)."""
    if conf < min_conf:
        return None, "confidence"
    items, ok = _parse(tex or "")
    if not ok:
        return None, "unbalanced"
    tex = _ser(items)  # tidy(tex)
    why = _junk(items)
    if why:
        return None, why
    word = _unknown_word(items)
    if word:
        return None, f"word {word}"  # text the model cannot read ("phút" -> "phut")
    ocr = _LABEL.sub("", ocr_text or "")
    nodes = list(_walk(items))
    # Tesseract drops commas and exponents and reads letters as digits ("t_2=0,405" -> "1, =0,405"), but each
    # digit it reads on the line is one of the model's (outside a fraction or root, whose ink it reads in any order)
    mine, theirs = _skeleton(items), _ocr_skeleton(ocr)
    if any(k == "D" for k, _ in mine) and any(k == "D" for k, _ in theirs) and not _digits_agree(mine, theirs):
        return None, "digits differ from OCR"
    if any(n[0] == "f" for n in nodes):
        body, tail = _split_tail(items)
        if not body:
            return None, "empty"
        tail += _end_punct(tail, ocr)
        return MathDecision("eq", latex=_ser(body), tail=tail, conf=conf, source=tex), "ok"
    parts = _parts(items)
    if parts is None:
        return None, "nested scripts"
    if not any(n[0] == "s" for n in nodes) and not any(c.isdigit() for t, _ in parts for c in t) and not any(
        n[0] == "t" and n[1] in GREEK for n in nodes
    ):
        return None, "letters only"  # OCR reads plain letters as well
    if _bad_case(parts, ocr):
        return None, "letter case"
    seen = re.sub(r"\D", "", ocr)
    if seen and not any(c.isdigit() for t, _ in parts for c in t):
        return None, "digits only in OCR"  # "0" read as \omega
    parts = _respace(parts, ocr)
    end = _end_punct(parts[-1][0], ocr)
    if end:
        parts = parts[:-1] + [(parts[-1][0] + end, None)] if parts[-1][1] is None else parts + [(end, None)]
    return MathDecision("text", parts=parts, conf=conf, source=tex), "ok"


_EQ_LABEL = re.compile(r"^\*?([A-D])\s*[.)](?:\\ |\s)*")


def split_label(decision: MathDecision | None) -> tuple[str, MathDecision] | None:
    """An equation reading that starts with an answer label ("D.\\frac{3}{2}", "C.\\ W=\\frac{m}{2v}": the region
    took in a label OCR garbled): ("D.", the equation without it), so the label can stay text; None otherwise."""
    if decision is None or decision.kind != "eq":
        return None
    m = _EQ_LABEL.match(decision.latex)
    if not m or not decision.latex[m.end() :]:
        return None
    rest = MathDecision("eq", latex=decision.latex[m.end() :], tail=decision.tail, conf=decision.conf, source=decision.source)
    return m.group(1) + ".", rest


def decide(tex: str, conf: float, ocr_text: str = "") -> MathDecision | None:
    """How a formula reading goes into Word, or None to keep the OCR text. tex: the model's LaTeX (tidy() is
    applied), conf: its mean token probability, ocr_text: Tesseract's text of the region (its digits must
    agree with a one-line reading's digits)."""
    return judge(tex, conf, ocr_text)[0]


# --- several readings of one region ---

VOTE_MIN_CONF = 0.8  # each of two agreeing readings
# what the model writes for a minus sign at the start of an exponent it cannot read: 10^{'21}, 10^{\theta^{22}}
_JUNK_EXPONENT = re.compile(r"\^\{(?:['`\"*]|\\[A-Za-z]+)\s*(?:\^\{(\d+)\}|(\d+))\}")


def repaired(tex: str) -> str | None:
    """tex with a junk exponent sign read as "-" (10^{'21} -> 10^{-21}), or None if it has none. Only a vote:
    vote() counts it for a reading that has the "-" there."""
    tex = tidy(tex)
    fixed = _JUNK_EXPONENT.sub(lambda m: "^{-" + (m.group(1) or m.group(2)) + "}", tex)
    return fixed if fixed != tex else None


# Letters whose capital is the small letter drawn larger: only the glyph's height tells their case (not k:
# its ascender is as tall as the capital).
CASE_BY_HEIGHT = set("COSUVWXZ")
# What the model writes for a coefficient, a letter at x-height and its subscript, 4v_O: 4_{VO} (or 4_{V_{O}}).
_COEFFICIENT_SUB = re.compile(r"(\d)_\{([%s])(?:_\{([^{}]+)\}|([^{}_^\\]+))\}" % "".join(sorted(CASE_BY_HEIGHT)), re.I)
_LETTER_SUB = re.compile(r"\d[A-Za-z]_\{")


def coefficients(tex: str) -> str | None:
    """tex with each coefficient whose letter the model read into a subscript written out (4_{VO} -> 4v_{O}),
    or None if it has none. Only with the ink's word for it (_letter_subscripts)."""
    tex = tidy(tex)
    fixed = _COEFFICIENT_SUB.sub(lambda m: m.group(1) + m.group(2).lower() + "_{" + (m.group(3) or m.group(4)) + "}", tex)
    return fixed if fixed != tex else None


def _letter_subscripts(img: np.ndarray) -> int:
    """How many times the ink of a one-line crop shows a tall glyph (a digit), a glyph of x-height standing on the
    baseline and a lowered glyph in a row: a coefficient, a small letter and its subscript (4v_O)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(_ink_mask(img).astype(np.uint8), connectivity=8)
    boxes = sorted((tuple(int(v) for v in st[c, :4]) for c in range(1, n) if st[c, cv2.CC_STAT_AREA] >= 4), key=lambda b: b[0])
    if len(boxes) < 3:
        return 0
    top = max(b[3] for b in boxes)
    tall = [b for b in boxes if b[3] >= 0.8 * top]
    size, base = float(np.median([b[3] for b in tall])), float(np.median([b[1] + b[3] for b in tall]))
    count = 0
    for a, b, c in zip(boxes, boxes[1:], boxes[2:]):
        near = b[0] - (a[0] + a[2]) <= 0.3 * size and c[0] - (b[0] + b[2]) <= 0.3 * size
        small = 0.5 * size <= b[3] <= 0.8 * size and abs(b[1] + b[3] - base) <= 0.08 * size
        lowered = c[1] + c[3] >= base + 0.08 * size and c[1] > b[1]
        count += a[3] >= 0.8 * top and near and small and lowered
    return count


def _ink_case(tex: str, ocr_text: str) -> str:
    """tex with its first letter in the case of the OCR text's first letter when the two differ only in case and
    the letter's case is told by its height: the OCR text's case was measured on the glyph (app/mathregions.py),
    the model sees the crop stretched to a square ("V_{H}" for v_H)."""
    tex, seen = tidy(tex), _letters(_LABEL.sub("", ocr_text or ""))[:1]
    for m in re.finditer(r"\\[A-Za-z]+|[A-Za-z]", tex):
        if m.group().startswith("\\"):
            continue
        if seen.upper() in CASE_BY_HEIGHT and m.group() != seen and m.group().lower() == seen.lower():
            return tex[: m.start()] + seen + tex[m.end() :]
        break
    return tex


def vote(readings: list[tuple[str, float]], ocr_text: str = "", min_conf: float | None = None) -> MathDecision | None:
    """The decision two readings of a region (different crops of it) agree on, or None.
    A reading whose exponent starts with junk where another reading has "-" counts for that reading."""
    min_conf = VOTE_MIN_CONF if min_conf is None else min_conf
    decisions = [judge(tex, conf, ocr_text, min_conf)[0] for tex, conf in readings]
    votes: dict[tuple, list[MathDecision]] = {}
    for d in decisions:
        if d is not None:
            votes.setdefault(d.key(), []).append(d)
    for (tex, conf), d in zip(readings, decisions):
        fixed = repaired(tex) if d is None else None
        if fixed:
            r = judge(fixed, conf, ocr_text, min_conf)[0]
            if r is not None and r.key() in votes:
                votes[r.key()].append(r)
    best = max(votes.values(), key=lambda ds: (len(ds), max(d.conf for d in ds)), default=None)
    if not best or len(best) < 2:
        return None
    # readings that agree may differ in spacing: take the one with a space between number and unit ("20 km")
    return max(best, key=lambda d: (len(re.findall(r"[\d}] [A-Za-z°Ω]", d.as_latex())) if d.kind == "text" else 0, d.conf))


def _usable(readings: list[tuple[str, float]], ocr_text: str) -> bool:
    """Some reading could count in a vote (accepted by judge() as it is or repaired)."""
    return any(judge(tex, conf, ocr_text, VOTE_MIN_CONF)[0] is not None or repaired(tex) for tex, conf in readings)


def _as_ocr(reading: tuple[str, float], ocr_text: str) -> MathDecision | None:
    """The decision of a reading that is plain text (no script, no 2-D part) equal to the OCR text but for spaces,
    else None."""
    d = judge(reading[0], reading[1], ocr_text, VOTE_MIN_CONF)[0]
    if d is None or d.kind != "text" or any(script for _, script in d.parts):
        return None
    plain, ocr = ("".join(t for t, _ in d.parts), _LABEL.sub("", ocr_text or ""))
    return d if re.sub(r"\s", "", plain) == re.sub(r"\s", "", ocr) else None


def _settled(readings: list[tuple[str, float]], ocr_text: str) -> bool:
    """True if more readings cannot change vote(): two agree, or none can be used."""
    if not _usable(readings, ocr_text):
        return True
    return vote(readings, ocr_text) is not None


def _pad(img: np.ndarray, px: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255) if px else img


# Crops of a region given to the model, tuned at 300 dpi: cut to the ink, then 3 px and 8 px of white around it.
# The third is read only when the first two do not agree.
PADS = (0, 3, 8)
HEAD_PAD = 3  # margin of the crop without its last word
# A region whose first reading is plain text equal to the OCR text ("6 kPa.") is not read again: it writes the OCR
# text anyway. About 20 % fewer crops, the same .docx on the test scans.
SAME_STOP = True
# A blank gap this wide (share of the crop's height) before a last word this narrow (share of its width): the
# unit or mark after a number or a symbol ("7,66.10^{-22} J.", "T_1;").
UNIT_GAP = 0.15
UNIT_WIDTH = 0.4
_UNIT_TAIL = re.compile(r"[A-Za-zΩ°%/]*[.,;:]*")


def _without_unit(img: np.ndarray) -> np.ndarray | None:
    """The crop up to the blank gap before its last word when that word is short (a unit, a mark), or None."""
    ink = _ink_mask(img)
    cols = ink.any(axis=0)
    w = len(cols)
    edges = np.flatnonzero(np.diff(np.concatenate(([1], cols.astype(np.int8), [1]))))
    gaps = [(a, b) for a, b in zip(edges[0::2], edges[1::2]) if a > 0 and b < w and b - a >= UNIT_GAP * img.shape[0]]
    if not gaps or w - gaps[-1][1] > UNIT_WIDTH * w:
        return None
    rows = np.flatnonzero(ink[:, : gaps[-1][0]].any(axis=1))
    return img[rows[0] : rows[-1] + 1, : gaps[-1][0]]


def _squeeze(tex: str) -> str:
    return re.sub(r"\\ | ", "", tex)


def _confirmed(readings: list[tuple[str, float]], head: tuple[str, float], ocr_text: str) -> tuple[str, float] | None:
    """A reading of the region (usable as it is, or repaired()) that a reading of its crop without the last word
    confirms: the latter is the former up to a unit or mark that Tesseract read at the end too ("T_{1}" for "T_{1};"
    with OCR "T,;"; "7,66.10^{-22}" for "7,66.10^{\\mu^{22}}\\ J." with OCR "7,66.1022J."). That reading with the
    confirming reading's confidence, or None."""
    start, ocr = _squeeze(tidy(head[0])), _squeeze(ocr_text).lower()
    if not start or start[-1] in ".,;:":
        return None
    for tex, conf in readings:
        whole = tidy(tex) if judge(tex, conf, ocr_text, VOTE_MIN_CONF)[0] else repaired(tex)
        rest = _squeeze(whole)[len(start) :] if whole and _squeeze(whole).startswith(start) else None
        if rest and _UNIT_TAIL.fullmatch(rest) and ocr.endswith(rest.lower()):
            return whole, head[1]
    return None


def read_regions(
    gray: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    ocr_texts: list[str],
    em_px: list[float] | None = None,
    reader: FormulaReader | None = None,
    case_from_ink: list[bool] | None = None,
) -> list[MathDecision | None]:
    """Decisions for formula regions of a page image (grey, uint8; boxes x0, y0, x1, y1 in its pixels;
    ocr_texts: Tesseract's text of each region; em_px: font size of each region, for the label dot;
    case_from_ink: the case of the OCR text's first letter was measured on its glyph).
    None for a region whose reading is not trusted, and for all when the model is off."""
    n = len(boxes)
    reader = reader or get_reader()
    if reader is None or not n:
        return [None] * n
    crops = [cut(gray, box, em_px[i] if em_px else None) for i, box in enumerate(boxes)]
    todo = [i for i, c in enumerate(crops) if c is not None and c.shape[0] >= 6 and c.shape[1] >= 4 and c.size <= MAX_CROP_PIXELS]
    readings: dict[int, list[tuple[str, float]]] = {i: [] for i in todo}
    agreed: dict[int, MathDecision] = {}  # SAME_STOP: one reading Tesseract's text agrees with
    shown: dict[int, int] = {}

    def checked(i: int, tex: str) -> str:
        """The reading with what the ink says: coefficients (4_{VO} -> 4v_{O}) where the ink shows as many letters
        with a subscript after a digit, the first letter's case from the glyph."""
        fixed = coefficients(tex)
        if fixed:
            if i not in shown:
                shown[i] = _letter_subscripts(crops[i])
            tex = fixed if shown[i] == len(_LETTER_SUB.findall(fixed)) else tex
        return _ink_case(tex, ocr_texts[i]) if case_from_ink and case_from_ink[i] else tex

    for k, pad in enumerate(PADS):
        if not todo:
            break
        for i, (tex, conf) in zip(todo, reader.read([_pad(crops[i], pad) for i in todo])):
            readings[i].append((checked(i, tex), conf))
        if k == 0 and SAME_STOP:
            for i in todo:
                d = _as_ocr(readings[i][0], ocr_texts[i])
                if d is not None:
                    agreed[i] = d
            todo = [i for i in todo if i not in agreed]
        if k >= 1:
            todo = [i for i in todo if not _settled(readings[i], ocr_texts[i])]
    # The crops differ in their margin only, so a misreading may come back in each: a minus starting an exponent
    # written as a symbol ("7,66.10^{\mu^{22}} J."), a thin subscript "1" beside a ";" read as "\i". Read without
    # its last short word (unit, mark), the rest is often read right; it then counts as a reading of the whole
    # region (_confirmed), so the readings that agree with it count with it (vote()).
    parts = {i: _without_unit(crops[i]) for i in todo}  # todo: unsettled, some reading usable or repairable
    again = [i for i, part in parts.items() if part is not None]
    for i, head in zip(again, reader.read([_pad(parts[i], HEAD_PAD) for i in again]) if again else []):
        confirmed = _confirmed(readings[i], head, ocr_texts[i])
        if confirmed:
            readings[i].append(confirmed)
    return [agreed[i] if i in agreed else vote(readings[i], ocr_texts[i]) if i in readings else None for i in range(n)]


# --- crops ---


def _ink_mask(img: np.ndarray) -> np.ndarray:
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img <= min(t, 180)  # Otsu's dark class (t is 0 for a pure black-and-white crop)


def cut(gray: np.ndarray, box: tuple[int, int, int, int], em: float | None = None, trim: bool = True) -> np.ndarray | None:
    """The box of a page image cut to its ink, without a small dot standing alone at its left edge (the "." of
    a label "C."). None when the box holds no ink."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    img = gray[y0:y1, x0:x1].copy()
    ink = _ink_mask(img)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    comps = sorted(range(1, n), key=lambda c: st[c, cv2.CC_STAT_LEFT])
    if not comps:
        return None
    if len(comps) >= 2:
        if not em:
            heights = [st[c, cv2.CC_STAT_HEIGHT] for c in comps if st[c, cv2.CC_STAT_AREA] >= 4]
            em = 1.4 * float(np.median(heights)) if heights else float(y1 - y0)
        c, nxt = comps[0], comps[1]
        small = st[c, cv2.CC_STAT_WIDTH] < 0.2 * em and st[c, cv2.CC_STAT_HEIGHT] < 0.2 * em
        gap = st[nxt, cv2.CC_STAT_LEFT] - (st[c, cv2.CC_STAT_LEFT] + st[c, cv2.CC_STAT_WIDTH])
        if small and gap >= 0.12 * em:
            img[lab == c] = 255
            ink[lab == c] = False
    if trim:
        ys, xs = np.nonzero(ink)
        if not len(xs):
            return None
        img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return img
