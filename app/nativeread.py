"""Formula pictures of text pages read with the formula model: "Chuyển thành công thức sửa được".

app/native.py keeps a formula as an inline picture where its text layer cannot be rebuilt (MathType saved as
vector outlines, glyphs without usable text, layouts nativemath does not handle). When the teacher asks for it
(careful conversion), those pictures are read with the formula model the way scanned pages are (mathocr.read_regions:
two readings must agree, Tesseract's text of the same place must hold the same digits, the Vietnamese-word and
label checks of app/ocr.py), at most MFR_MAX_FORMULAS per file in page order:
  - a 2-D reading (fraction, root, vector…) becomes a Word equation; its picture stays on the run, so an equation
    Word cannot take is written as the picture (docx_builder);
  - a one-line reading (x_A, 4,5.10³, 3 km) becomes text with superscripts and subscripts;
  - a one-line reading of plain text mathocr turns down ("letters only" – axes Ox, units m/s –, "word phut": the
    model has no Vietnamese letters) becomes Tesseract's text of the same place when another confident reading of
    the picture is that text letter for letter (Vietnamese marks aside: the text keeps Tesseract's) – two readers
    that never share their mistakes (Tesseract misreads MathType's italic letters: A as "4", B as "8"; such
    pictures stay pictures);
  - a reading that is not trusted, and the formulas past the budget, stay pictures.

Crops: the picture itself (only the formula's own ink, rendered at native.FORMULA_DPI), flattened on white and
stacked on one canvas so neighbours never leak into a crop. Tesseract's text: the page rendered at the same dpi,
the words at least half inside the formula's box (the setup that read the vatli test file without a wrong formula).
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from statistics import median

import cv2
import numpy as np
import pymupdf

from . import mathocr, mathregions
from .docx_builder import FONT
from .model import Run
from .ocr import (
    _base_letters,
    _converts,
    _unmarked,
    drop_light_ink,
    render_gray,
    tesseract_lines,
)
from .ocr_fix import fix_lines
from .settings import OCR_MAX_SIDE_PX

DPI = 300  # native.FORMULA_DPI: the pictures' resolution, and what mathocr's crop padding was tuned for
GAP = 24  # px of white between stacked crops
CHUNK = 20  # pictures per read_regions call: cancellation is checked (and progress told) between chunks
CANVAS_MAX = 4 * mathocr.MAX_CROP_PIXELS  # px of one chunk's canvas (a chunk ends early before it grows past)
# Tesseract reads only the bands of the page around the pictures (their lines), stacked: a dense page costs what
# its formula lines cost. Bands reach this far above and below a picture (share of its height), and no further
# than the page is wide at DPI (OCR_MAX_SIDE_PX) nor this many pixels in all: a larger page keeps its pictures.
BAND_MARGIN = 0.3
BANDS_MAX = 3 * OCR_MAX_SIDE_PX * OCR_MAX_SIDE_PX // 4
MIN_DPI = 100  # a page wider than OCR_MAX_SIDE_PX at DPI is read at a lower dpi (A3 landscape: 272), not below this
TESSERACT_SECONDS = 60  # one page's bands at most
OCR_FILE_SECONDS = 180  # all pages of a file at most (a file of many dense pages would hold the careful turn long)
# Fonts whose letters are symbols (Word draws Latin letters in Symbol as Greek): never the font of a reading
SYMBOL_FONTS = re.compile(r"symbol|wingdings|webdings|dingbat|mt\s*extra|marlett", re.I)


@dataclass
class FormulaBudget:
    """One per careful conversion of a file, shared by its text pages (page order)."""

    left: int  # formulas still to read
    languages: list[str]  # Tesseract languages for the text of each formula's place
    check: Callable[[], None] | None = None  # raises when the conversion was cancelled
    read_eq: int = 0  # became Word equations
    read_text: int = 0  # became text with superscripts / subscripts
    unsure: int = 0  # read, not trusted: kept as pictures
    over: int = 0  # past the budget: kept as pictures
    skipped: int = 0  # never tried (page turned, larger than the model reads, Tesseract time spent): kept, not charged
    failed: bool = False  # the model could not be used (missing, broken): the rest kept as pictures
    ocr_seconds: float = OCR_FILE_SECONDS  # Tesseract time left for the file: once spent, pictures are kept

    @property
    def read(self) -> int:
        return self.read_eq + self.read_text


@dataclass
class Holder:
    """Runs of one paragraph or table cell of the page (the live list), and the object whose .text follows them."""

    owner: object
    runs: list[Run]


def _reader() -> mathocr.FormulaReader | None:
    # the scanned pages of a long file skip the model (convert: mathocr.skipped); this file's text pages asked for it
    with mathocr.skipped(False):
        return mathocr.get_reader()


def _pixels(f: dict) -> float:
    """The picture's size in pixels at DPI (known before decoding it)."""
    return f["run"].image_w * f["run"].image_h * (DPI / 72) ** 2


def _crop(png: bytes) -> np.ndarray | None:
    """The picture flattened on white, grey (its alpha is the ink); None for one not decoded or larger than the model
    reads (mathocr.MAX_CROP_PIXELS)."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[0] < 6 or img.shape[1] < 4 or img.shape[0] * img.shape[1] > mathocr.MAX_CROP_PIXELS:
        return None
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if img.shape[2] == 4:  # on white: 255 - ink darkness × alpha, in integers (one byte a pixel, not floats)
        gray = (255 - (255 - gray.astype(np.uint16)) * img[:, :, 3] // 255).astype(np.uint8)
    return gray


def _stack(crops: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    width = max(c.shape[1] for c in crops) + 2 * GAP
    height = sum(c.shape[0] for c in crops) + GAP * (len(crops) + 1)
    canvas = np.full((height, width), 255, np.uint8)
    boxes = []
    y = GAP
    for c in crops:
        h, w = c.shape
        canvas[y : y + h, GAP : GAP + w] = c
        boxes.append((GAP, y, GAP + w, y + h))
        y += h + GAP
    return canvas, boxes


def _dpi(page: pymupdf.Page) -> int:
    """The dpi the bands are read at: DPI, lower for a page wider than OCR_MAX_SIDE_PX at DPI (0: too wide)."""
    dpi = min(DPI, int(OCR_MAX_SIDE_PX * 72 / max(page.rect.width, 1)))
    return dpi if dpi >= MIN_DPI else 0


def _bands(page: pymupdf.Page, boxes: list[pymupdf.Rect], dpi: int = DPI) -> list[pymupdf.Rect] | None:
    """The page's horizontal bands around the boxes (merged where they overlap), or None if they are too large to
    read (BANDS_MAX pixels in all at dpi)."""
    area = page.rect
    spans = sorted((max(area.y0, b.y0 - BAND_MARGIN * b.height), min(area.y1, b.y1 + BAND_MARGIN * b.height)) for b in boxes)
    merged: list[list[float]] = []
    for y0, y1 in spans:
        if merged and y0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])
    bands = [pymupdf.Rect(area.x0, y0, area.x1, y1) for y0, y1 in merged if y1 > y0]
    if sum(b.width * b.height for b in bands) * (dpi / 72) ** 2 > BANDS_MAX:
        return None
    return bands


def _words(page: pymupdf.Page, boxes: list[pymupdf.Rect], languages: list[str], timeout: float = TESSERACT_SECONDS) -> list | None:
    """Tesseract's words of the bands of the page around the boxes, in page pixels at DPI; None if too large."""
    dpi = _dpi(page)
    bands = _bands(page, boxes, dpi) if dpi else None
    if bands is None:
        return None
    if not bands:
        return []
    images = [render_gray(page, dpi, clip=b) for b in bands]
    width = max(img.shape[1] for img in images)
    canvas = np.full((sum(img.shape[0] for img in images) + GAP * (len(images) + 1), width), 255, np.uint8)
    slots = []  # (top in the canvas, height, top in page pixels)
    y = GAP
    for img, band in zip(images, bands, strict=True):
        canvas[y : y + img.shape[0], : img.shape[1]] = img
        slots.append((y, img.shape[0], band.y0 * dpi / 72))
        y += img.shape[0] + GAP
    drop_light_ink(canvas)
    lines = tesseract_lines(canvas, "+".join(languages) or "vie", dpi, timeout=timeout)
    fix_lines(lines)
    x0, k = page.rect.x0 * dpi / 72, DPI / dpi  # back to page pixels at DPI
    words = []
    for w in (w for ln in lines for w in ln.words):
        cy = (w.y0 + w.y1) / 2
        slot = next((s for s in slots if s[0] <= cy < s[0] + s[1]), None)
        if slot is not None:
            dy = slot[2] - slot[0]
            words.append(replace(
                w, x0=round((w.x0 + x0) * k), x1=round((w.x1 + x0) * k), y0=round((w.y0 + dy) * k),
                y1=round((w.y1 + dy) * k), em_px=w.em_px * k,
            ))
    return words


def _inside(words: list, box: pymupdf.Rect) -> list:
    x0, y0, x1, y1 = (v * DPI / 72 for v in box)
    out = []
    for w in words:
        ix = max(0, min(x1, w.x1) - max(x0, w.x0))
        iy = max(0, min(y1, w.y1) - max(y0, w.y0))
        if ix * iy >= 0.5 * max(1, (w.x1 - w.x0) * (w.y1 - w.y0)):
            out.append(w)
    return sorted(out, key=lambda w: (w.line_no, w.x0))


def _texty(r: Run) -> bool:
    return bool(r.font and not r.math and not r.image and r.text.strip() and not SYMBOL_FONTS.search(r.font))


def _font(runs: list[Run], i: int) -> str | None:
    """The text font around the picture (the nearest text run before it, else after it; not a symbol font)."""
    for r in [*reversed(runs[:i]), *runs[i + 1 :]]:
        if _texty(r):
            return r.font
    return None


def _chunks(formulas: list[dict]):
    """(formula, crop) chunks of at most CHUNK pictures and CANVAS_MAX canvas pixels, decoded one chunk at a time
    (crop None: not decoded)."""
    chunk: list = []
    width = height = 0
    for f in formulas:
        c = _crop(f["run"].image)
        if c is not None and chunk and max(width, c.shape[1]) * (height + c.shape[0]) > CANVAS_MAX:
            yield chunk
            chunk, width, height = [], 0, 0
        chunk.append((f, c))
        if c is not None:
            width, height = max(width, c.shape[1]), height + c.shape[0] + GAP
        if len(chunk) == CHUNK:
            yield chunk
            chunk, width, height = [], 0, 0
    if chunk:
        yield chunk


def _candidates(formulas: list[dict], holders: list[Holder]) -> list[dict]:
    """The formula pictures placed in the page's paragraphs and cells that a reading could replace."""
    where = {id(r) for h in holders for r in h.runs}
    return [
        f
        for f in formulas
        if f["run"] is not None and f["run"].image and not f["run"].math and not f.get("pieces")
        and id(f["run"]) in where and f["run"].image_w > 1 and f["run"].image_h > 1
    ]


def readable(page: pymupdf.Page, formulas: list[dict], holders: list[Holder]) -> int:
    """How many of the page's formula pictures a careful conversion would try to read (what the page offers)."""
    if page.rotation % 360:
        return 0
    return sum(1 for f in _candidates(formulas, holders) if _pixels(f) <= mathocr.MAX_CROP_PIXELS)


def read_pictures(page: pymupdf.Page, formulas: list[dict], holders: list[Holder], budget: FormulaBudget) -> None:
    """Read the page's formula pictures (formulas: native's formula dicts, page order) in place: a trusted reading
    replaces the picture run in its paragraph or cell; f["read"] says how ("eq" / "text")."""
    where = {id(r): h for h in holders for r in h.runs}
    todo = _candidates(formulas, holders)
    if not todo or budget.failed:
        return
    # never tried, and not charged to the budget: the boxes are the unrotated page's while its rendering is turned;
    # the file's Tesseract time is spent; larger than the model reads (a page-sized "formula" would take gigabytes)
    if page.rotation % 360 or budget.ocr_seconds <= 0:
        budget.skipped += len(todo)
        return
    keep = [f for f in todo if _pixels(f) <= mathocr.MAX_CROP_PIXELS]
    budget.skipped += len(todo) - len(keep)
    if not keep:
        return
    if budget.left <= 0:
        budget.over += len(keep)
        return
    budget.over += max(0, len(keep) - budget.left)
    keep = keep[: budget.left]
    budget.left -= len(keep)
    reader = _reader()
    if reader is None:
        budget.failed = True
        return
    if budget.check is not None:
        budget.check()
    started = time.monotonic()
    try:
        words = _words(page, [f["box"] for f in keep], budget.languages, min(TESSERACT_SECONDS, budget.ocr_seconds))
    except Exception:  # Tesseract failed or took too long: no digit check, so nothing is tried
        words = None
    budget.ocr_seconds -= time.monotonic() - started
    if words is None:
        budget.skipped += len(keep)
        return
    changed: set[int] = set()
    # the page's text font: an equation alone in its paragraph or cell has no text around it, and an equation run
    # without a font is written as Word's display equation (docx_builder._keep_inline): centred, full-size fractions
    fonts = Counter(r.font for h in holders for r in h.runs if _texty(r))
    page_font = fonts.most_common(1)[0][0] if fonts else FONT
    done = 0
    for chunk in _chunks(keep):
        if budget.check is not None:
            budget.check()
        canvas = None  # the last chunk's canvas goes before this one's is built
        missing = sum(c is None for _, c in chunk)  # not decoded, or larger than its size said
        budget.unsure += missing
        done += missing
        chunk = [(f, c) for f, c in chunk if c is not None]
        if not chunk:
            continue
        canvas, boxes = _stack([c for _, c in chunk])
        places = [_inside(words, f["box"]) for f, _ in chunk]
        texts = [" ".join(w.text for w in ws) for ws in places]
        ems = [median(w.em_px for w in ws) if ws else f["size"] * DPI / 72 for (f, _), ws in zip(chunk, places, strict=True)]
        try:
            decisions = mathocr.read_regions(canvas, boxes, texts, ems, reader)
            decisions = [
                d if d is not None else _as_tesseract(reader, canvas, box, em, text, _after(where[id(f["run"])], f["run"]))
                for d, box, em, text, (f, _) in zip(decisions, boxes, ems, texts, chunk, strict=True)
            ]
            arrows = [
                d is not None and d.kind == "eq" and _BAR.search(d.latex) is not None and _arrow_seen(reader, canvas, box, em)
                for d, box, em in zip(decisions, boxes, ems, strict=True)
            ]
        except Exception:
            mathocr.broken()
            budget.failed = True
            budget.unsure += len(keep) - done
            break
        done += len(chunk)
        for (f, _), ws, text, d, arrow in zip(chunk, places, texts, decisions, arrows, strict=True):
            if (
                d is None
                or arrow
                or (d.kind == "eq" and not _converts(d.latex))
                or mathocr.split_label(d) is not None
                or _unmarked(d, ws)
                or _risky(d, text) is not None
            ):
                budget.unsure += 1
                continue
            d = _tight(d)
            h = where[id(f["run"])]
            if _apply(f, h, d, page_font):
                changed.add(id(h.owner))
                if d.kind == "eq":
                    budget.read_eq += 1
                else:
                    budget.read_text += 1
            else:
                budget.unsure += 1
    for h in holders:
        if id(h.owner) in changed:
            h.owner.text = "".join(r.text for r in h.runs)


END_PUNCT = ".,;:"
NEXT_MARKS = ".,;:?!)]"  # marks Tesseract's last word may take from the text right after the picture
SENTENCE_MARKS = ".,;:?!"  # Tesseract may misread one as another (":" as ";", "." as ",")
PLAIN_MAX = 24  # characters of Tesseract's text a plain reading may stand for (labels, units, short words)


def _after(h: Holder, run: Run) -> str:
    """The text after the run in its paragraph or cell, from its first character that is not blank (native puts
    space and tab runs between a picture and the mark after it: "￼", " ", " ", ", suy ra"); a Greek question mark
    (PDFs extract ";" so) as ";"."""
    i = next((j for j, r in enumerate(h.runs) if r is run), len(h.runs))
    return "".join(r.text for r in h.runs[i + 1 :]).lstrip().replace("\u037e", ";")


def _as_tesseract(reader, canvas: np.ndarray, box: tuple, em: float, text: str, after: str) -> mathocr.MathDecision | None:
    """Tesseract's text of the place as the reading, when a confident reading of the picture is that same plain
    text (no script, no 2-D part), letter for letter but for Vietnamese marks and spaces; else None. A last mark of
    Tesseract's text that the text after the picture starts with is that text's ("km/h?" before "?")."""
    ocr = text.strip()
    if ocr and after[:1] and (ocr.endswith(after[:1]) and after[:1] in NEXT_MARKS
                              or ocr[-1] in SENTENCE_MARKS and after[:1] in SENTENCE_MARKS):
        ocr = ocr[:-1].rstrip()
    if not ocr or len(ocr) > PLAIN_MAX:
        return None
    crop = mathocr.cut(canvas, box, em)
    if crop is None or crop.size > mathocr.MAX_CROP_PIXELS:
        return None
    flat = "".join(ocr.split())
    readings = mathocr.plain_readings(reader, crop)
    # every reading confident and the same text ("\\mu m" read once as "um": the other reading disagrees)
    if not readings or any(p not in (flat, _base_letters(flat)) for p in readings):
        return None
    if any(p != flat for p in readings) and not _marks_ok(flat):
        return None
    return mathocr.MathDecision("text", parts=[(ocr, None)], source=flat)


# A Vietnamese syllable without its marks: initial consonants, vowels, final consonants
_SYLLABLE = re.compile(r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?[aeiouy]{1,3}(?:ch|ng|nh|[cmnpt])?")


# The marks of Vietnamese letters: grave, acute, circumflex, tilde, breve, hook above, horn, dot below
VIETNAMESE_MARKS = set("\u0300\u0301\u0302\u0303\u0306\u0309\u031b\u0323")


def _marks_ok(text: str) -> bool:
    """The Vietnamese marks of Tesseract's text may stand where the model read none: only on lowercase words of
    three letters or more that are Vietnamese syllables ("phút", "vòng", "giờ"). Tesseract also puts marks on
    letters that have none – axis and point names, capitals ("Ox" as "Ôx", "U" as "Ữ"), short letter pairs ("di" as
    "đĩ", "ac" as "ác") – and no reader checks those; nor đ (the model reads it d), nor marks Vietnamese has not (ï)."""
    for word in re.findall(r"[^\W\d_]+", text):
        if word == _base_letters(word):
            continue
        marks = {c for c in unicodedata.normalize("NFD", word) if unicodedata.combining(c)}
        if "đ" in word.lower() or not marks <= VIETNAMESE_MARKS:
            return False
        if not word.islower() or len(word) < 3 or not _SYLLABLE.fullmatch(_base_letters(word).lower()):
            return False
    return True


# Subscripts the model reads as written: digits, capitals (point names: v_{AB}), one small letter and these Latin
# abbreviations. Others are mostly Vietnamese words it spells without marks (v_ngược as v_{nguoc}, Q_tỏa as Q_{toa}).
LATIN_SUBS = {"max", "min", "tb", "ms", "ci", "tp", "tt", "ht", "hp", "th", "ph", "mt", "nt"}
# One-letter subscripts that may be a digit the model read as a letter (U₅ as U_s, l₀ as l_o, R₁ as R_l): the
# digit check pairs script characters with anything
DIGIT_LIKE = set("sSoOlIzZgqb")
# Exponents read as written: a signed number, a signed letter, a degree, primes, a star
_EXPONENT = re.compile(r"[-−+]?(?:\d+(?:[,.]\d+)?|[A-Za-z])|°|'+|\*")


_BAR = re.compile(r"\\(?:overline|bar)(?![A-Za-z])")
_ARROW = re.compile(r"\\(?:vec|overrightarrow|overleftarrow|rightarrow|to)(?![A-Za-z])")
ARROW_PADS = (8, 12)  # px of white around the crop: an arrowhead on its edge is seen with a wider margin


def _arrow_seen(reader, canvas: np.ndarray, box: tuple, em: float) -> bool:
    """Whether a reading of the crop with a wider margin has an arrow where the decision has a bar: the vector's
    arrowhead sits on the crop's edge and the two narrow readings take it for a bar (\\vec{p} as \\bar{p})."""
    crop = mathocr.cut(canvas, box, em)
    if crop is None:
        return True
    return any(_ARROW.search(tex or "") for tex, _ in reader.read([mathocr._pad(crop, px) for px in ARROW_PADS]))


def _ink_color(png: bytes) -> str | None:
    """The colour of the picture's ink ("RRGGBB") when it is not black or dark grey (a red answer, a blue step)."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    ink = img[img[:, :, 3] > 200][:, :3]
    if len(ink) < 10:
        return None
    b, g, r = (int(v) for v in np.median(ink, axis=0))
    if max(r, g, b) - min(r, g, b) < 60 and max(r, g, b) < 120:
        return None
    return f"{r:02X}{g:02X}{b:02X}"


# Units written after a number or a formula: upright text, never variables (single letters that are also common
# variables – t, x, a – are not here: "Q=\\frac{U^2}{R}t" keeps its t in the equation)
UNITS = {
    "m", "cm", "mm", "km", "dm", "nm", "μm", "s", "ms", "min", "h", "kg", "g", "mg", "N", "kN", "J", "kJ", "MJ",
    "W", "kW", "MW", "V", "kV", "mV", "A", "mA", "Ω", "kΩ", "Hz", "kHz", "MHz", "Pa", "kPa", "MPa", "atm", "mol",
    "K", "°C", "rad", "l", "L", "eV", "MeV", "C", "μC", "F", "μF", "T", "Wb", "dB", "%", "vòng", "lần",
}


def _unit(word: str) -> bool:
    return bool(word) and all(p in UNITS for p in word.split("/"))


def _leaves(items) -> str:
    return "".join(mathocr.SYMBOL_TEXT.get(n[1], n[1]) for n in mathocr._walk(items) if n[0] == "t")


def _risky(d: mathocr.MathDecision, ocr: str) -> str | None:
    """Why a reading read_regions() accepts may still be wrong here, or None. Its checks compare digits with
    Tesseract outside scripts and 2-D parts only, which the pictures of text pages need more of:
      - a subscript holding d/D (the model has no đ/Đ: W_đ, R_Đ, F_đh), one letter that may be a digit (U₅ as U_s,
        l₀ as l_o: DIGIT_LIKE) or a word
        of small letters (v_ngược, t_lên), and any script letter where Tesseract saw a Vietnamese letter;
      - an exponent that is not a signed number or letter, a degree or primes (10^{-4} rad/s read as 10^{- rad/s});
      - a lone letter whose case only its height tells as a numerator or denominator (s/t read as S/t), and a
        numerator, denominator or root without a letter or digit (\\frac{}{\\Delta l});
      - a leading "+" (k = … read as + = …);
      - a bar over a point pair (\\overline{MA}: in exams mostly the vector \\overrightarrow{MA} whose arrowhead the
        reading missed), a vector over several small letters (m\\vec{v} read as \\vec{mv})."""
    if re.match(r"\+(?!\d)", d.as_latex()):
        return "leading plus"
    items, _ = mathocr._parse(d.as_latex())
    for node in mathocr._walk(items):
        if node[0] == "f" and node[1] in ("overline", "bar"):
            text = re.sub(r"\s", "", _leaves(node[3][0]))
            if re.fullmatch(r"[A-Z]{2,}", text):  # Latin capitals (not \\overline{\\Delta L}, a mean value)
                return "bar over a point pair"  # \overrightarrow{MA} read as \overline{MA}: the arrowhead missed
        if node[0] == "f" and node[1] in ("vec", "overrightarrow"):
            text = re.sub(r"\s", "", _leaves(node[3][0]))
            if len(text) > 1 and re.search(r"[a-z]", text):
                return "vector over several small letters"  # m\vec{v} read as \vec{mv}
        if node[0] == "f" and node[1] in ("frac", "sqrt"):
            for arg in node[3]:
                text = re.sub(r"\s", "", _leaves(arg))
                if not re.search(r"[0-9A-Za-zα-ωΔΩ]", text):
                    return "empty 2-D part"  # \frac{}{\Delta l}, \frac{1}{-}: ink the picture does not hold
                if node[1] == "frac" and len(text) == 1 and text.upper() in mathocr.CASE_BY_HEIGHT:
                    return "letter case in a fraction"
        if node[0] != "s":
            continue
        for slot in (2, 3):
            if node[slot] is None:
                continue
            text = re.sub(r"\s", "", _leaves(node[slot]))
            if slot == 3:
                if not _EXPONENT.fullmatch(text):
                    return "exponent"
            elif re.search(r"[dD]", text) or text in DIGIT_LIKE:
                return "subscript d, or a letter like a digit"
            elif any(not w.isupper() and w not in LATIN_SUBS for w in re.findall(r"[A-Za-z]{2,}", text)):
                return "word in a subscript"
            if re.search(r"[A-Za-z]", text) and mathregions.vietnamese(ocr):
                return "script letters by Vietnamese ink"
    return None


def _glued(items: list) -> list:
    """Items without the spaces between two digits ("\\frac{2\\ 1}{4}": Word would show the numerator "2 1")."""
    out = []
    for k, node in enumerate(items):
        prev, nxt = (out[-1] if out else None), (items[k + 1] if k + 1 < len(items) else None)
        if node == ["t", "\\ "] and prev and nxt and prev[0] == nxt[0] == "t" and prev[1][-1:] in "0123456789," \
                and nxt[1][:1].isdigit():
            continue
        out.append(node)
    return out


def _main_line(items: list) -> list:
    """A row of an equation without the model's stray spaces: before a script (its base is the number or letter
    before it: "10\\ ^{-18}"), and between a fraction or root and the letters after it ("\\frac{1}{2}\\ at^{2}")."""
    out: list = []
    for k, node in enumerate(items):
        nxt = items[k + 1] if k + 1 < len(items) else None
        if node[0] == "s" and node[1] is None and len(out) > 1 and out[-1] == ["t", "\\ "] and out[-2][0] in ("t", "g"):
            out.pop()  # "10\\ ^{-18}": parsed as a space and a script without base
            node = ["s", out.pop(), node[2], node[3]]
        elif node == ["t", "\\ "] and out and out[-1][0] == "f" and nxt is not None and (
            (nxt[0] == "t" and nxt[1][:1].isalpha()) or (nxt[0] == "s" and nxt[1] is not None and nxt[1][1][:1].isalpha())
        ):
            continue
        if node[0] == "g":
            node = ["g", _main_line(node[1])]
        out.append(node)
    return out


# a unit after a number of an equation ("=5\\ m/s", "=45,22\\ mm"): written upright, not as variables
_EQ_UNIT = re.compile(r"(?<=\d)\\ ([A-Za-zΩμ°%]+(?:/[A-Za-zΩμ°%]+)*)(?=(?:\^\{[^{}]*\})?(?:$|[=,;.:]|\\ ))")


def _tight(d: mathocr.MathDecision) -> mathocr.MathDecision:
    """The decision without stray spaces: inside its scripts (the model's "10^{-\\,23}": superscript "- 23"),
    inside the numbers of its fractions and roots ("\\frac{2\\ 1}{4}"), before a script, after a fraction before
    letters; in text, Tesseract's spaces around "=" and after a sign ("y= ax", "q= - 4"). Units after a number of an
    equation upright (\\mathrm)."""
    if d.kind == "text":
        parts = []
        for k, (text, script) in enumerate(d.parts):
            if script is not None:
                text = re.sub(r"\s", "", text)
            else:
                text = re.sub(r"\s*=\s*", "=", text)
                text = re.sub(r"(?<=[=(])-\s+(?=\d)", "-", text)
                if k + 1 < len(d.parts) and d.parts[k + 1][1] is not None:
                    text = text.rstrip()  # "6,72.10 " before its exponent
            parts.append((text, script))
        return mathocr.MathDecision("text", parts=[p for p in parts if p[0]], conf=d.conf, source=d.source)
    items, _ = mathocr._parse(d.latex)
    items = _main_line(items)

    def tight(items):
        out = []
        for node in items or []:
            if node == ["t", "\\ "]:
                continue
            if node[0] == "g":
                node = ["g", tight(node[1])]
            out.append(node)
        return out

    for node in mathocr._walk(items):
        if node[0] == "s":
            for slot in (2, 3):
                if node[slot] is not None:
                    node[slot] = tight(node[slot])
        elif node[0] == "f" and node[1] in ("frac", "sqrt"):
            node[3][:] = [_glued(arg) for arg in node[3]]
    latex = mathocr._ser(items)
    upright = _EQ_UNIT.sub(lambda m: "\\ \\mathrm{%s}" % m.group(1) if _unit(m.group(1)) else m.group(0), latex)
    latex = upright if _converts(upright) else latex
    if not _converts(latex):
        latex = d.latex
    return mathocr.MathDecision("eq", latex=latex, tail=d.tail, conf=d.conf, source=d.source)


def _without_next_punct(text: str, after: str) -> str:
    """The reading without its final mark when the text right after the picture starts with a mark: Tesseract's
    word ran past the formula ("km/h." for the picture of "km/h", or "v2;" before ":", a mark it misread), so the
    reading took the mark the line already has."""
    if text and text[-1] in END_PUNCT and after[:1] and after[:1] in SENTENCE_MARKS:
        return text[:-1].rstrip()
    return text


def _apply(f: dict, h: Holder, d: mathocr.MathDecision, page_font: str = FONT) -> bool:
    run = f["run"]
    i = next((j for j, r in enumerate(h.runs) if r is run), None)
    if i is None:
        return False
    font = _font(h.runs, i) or page_font
    after = _after(h, run)
    color = run.color or _ink_color(run.image)  # a red answer stays red
    if d.kind == "eq":
        latex, tail = d.latex, d.tail
        # letters after the last fraction or root that are not a unit are the equation's variables (V=\\frac{1}{3}Bh)
        m = re.fullmatch(r"\s*([A-Za-z]+)(\s*[.,;:]*)", tail)
        if m and not _unit(m.group(1)):
            joined = latex + (" " if re.search(r"\\[A-Za-z]+$", latex) else "") + m.group(1)
            if _converts(joined):
                latex, tail = joined, m.group(2).strip()
        run.math = run.math_text = latex
        run.font = font  # an inline equation (docx_builder._keep_inline): a lone one is not centred by Word
        run.color = color
        tail = _without_next_punct(tail, after)
        if tail:
            h.runs.insert(i + 1, Run(tail, size=run.size, font=font, color=color))
        f["read"] = "eq"
        return True
    texts = [(text, script) for text, script in d.parts if text]
    if texts and texts[-1][1] is None:
        texts[-1] = (_without_next_punct(texts[-1][0], after), None)
    parts = [Run(text, size=run.size, font=font, color=color, script=script) for text, script in texts if text]
    if not parts:
        return False
    h.runs[i : i + 1] = parts
    f["run"] = None  # no picture left for this formula
    f["read"] = "text"
    return True
