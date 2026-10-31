"""OCR pipeline for scanned pages and for raster regions that hold text.

Per page: render -> turn upright (Tesseract OSD) and straighten (projection
profile) -> detect ruled tables (OpenCV) -> erase table rules and light
watermarks -> Tesseract hOCR (blocks, paragraphs, lines with baseline and
x_size, words) -> keep non-text ink (figures, photos, stamps) as pictures ->
paragraphs and table cells with font size, alignment, indents, spacing and
bold estimates.
"""

import logging
import os
import re
import subprocess
import tempfile
import unicodedata
from bisect import bisect_right
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from itertools import pairwise
from statistics import median

import cv2
import numpy as np
import pymupdf

from . import mathocr, mathregions, scriptocr, vietocr
from .errors import TaskError
from .layout import alignment, cell_align, looks_like_link
from .model import TAB, Cell, PageLayout, Paragraph, Picture, Run, Table
from .ocr_fix import fix_lines, fix_table
from .omml import latex_to_omml
from .settings import FORMULAS, OCR_DPI, OCR_MAX_SIDE_PX, TESSDATA_DIR, TESSERACT_CMD

log = logging.getLogger("pdf2word.ocr")

# Tesseract's per-line metrics relative to the font size in pixels, measured
# on samples/ (20pt title, 12pt body, 11pt table): x_size ~0.91em and
# x_ascenders + x_descenders ~0.445em. Averaging both damps single-line outliers.
XSIZE_PER_EM = 0.91
ASC_DESC_PER_EM = 0.445
# Stroke weight of a word = share of its ink that survives an erosion of
# ~0.11em: regular stems (~0.08em) vanish, bold stems survive. A word is bold
# at this multiple of the page median (higher for 1-2 letter words, which are
# noisy) and above an absolute floor; a paragraph or cell when most of its
# characters are.
BOLD_WEIGHT_RATIO = 6.0
BOLD_WEIGHT_RATIO_SHORT = 9.0
BOLD_WEIGHT_FLOOR = 0.08
BOLD_SHARE = 0.7
# Paragraph sizes within this share of the body size are taken as body text:
# per-line size estimates are noisy, real size changes are larger.
BODY_SIZE_TOLERANCE = 0.35
# Two Tesseract lines side by side (not overlapping horizontally) are one printed row when their word
# boxes share at least ROW_SHARE of the shorter line's height and ROW_SHARE_TALL of the taller one's, and
# their baselines are at most ROW_BASELINE_EM apart.
ROW_SHARE = 0.5
ROW_SHARE_TALL = 0.4
ROW_BASELINE_EM = 0.15
# Baseline pitch above this multiple of x_size starts a new paragraph
# (single spacing is ~1.26x, a blank line or paragraph gap adds ~0.6em).
PARAGRAPH_PITCH = 1.6
# Watermarks, background tints and scan noise are lighter than body text.
# On scans whose text is dark, everything at least this light is whitened
# (tuned on a scanned exam with a pale diagonal watermark: 170 also erased
# anti-aliased glyph edges, 215 gave the best word accuracy).
LIGHT_INK_LEVEL = 215
# Pages are straightened when skewed by at least this much (degrees); smaller
# corrections cost more in resampling blur than they gain.
MIN_DESKEW = 0.2
# Non-text ink kept as a picture must cover this share of the page and be at
# least this many inches on each side.
PICTURE_MIN_AREA = 0.01
PICTURE_MIN_SIDE_IN = 0.25
# Grey level below which a pixel counts as picture ink. Pale watermarks
# (~230) stay out; light-grey drawings keep their darker outlines.
PICTURE_INK_LEVEL = 200
HEADER_ZONE = 0.08  # top/bottom share of the page searched for headers and footers
# Text whose ink is at least this light and this grey counts as a stamp
# (black at 50% opacity scans as ~128; red and blue text keep their chroma).
FADED_LUMA = 100
FADED_CHROMA = 40
# Coloured text (blue "Câu 1:", red "[ĐNH]"): the darkest ink of a word spreads
# at least this much across R, G, B; colours of a page this close are one.
COLOUR_CHROMA = 60
COLOUR_MERGE = 60
# A table cell is shaded when its paper, against the paper around the table,
# is at least this much darker in some channel (0-255): Word's palest grey
# "Darker 5%" (F2F2F2) is 13; white cells of a scan differ by a few units.
# Paper that brightens or darkens across the table (a phone photo) raises it.
SHADE_MIN = 10
LUMA = np.array([0.299, 0.587, 0.114])


@dataclass
class Word:
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    text: str
    line_no: int
    em_px: float
    weight: float = 0.0
    bold: bool = False
    faded: bool = False  # grey ink, much lighter than body text: a stamped overlay
    color: str | None = None  # "RRGGBB" of coloured ink, None for black
    # A formula placeholder (added after fix_lines, in place of the OCR words of a formula
    # region): the formula model's reading (mathocr.MathDecision) or a FormulaPicture.
    math: object = None

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    block: int
    par: int
    baseline: float
    em_px: float
    words: list[Word] = field(default_factory=list)

    @property
    def x0(self) -> int:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> int:
        return max(w.x1 for w in self.words)


@dataclass
class FormulaPicture:
    """A formula region kept as an inline picture: PNG, size in points, depth below the baseline."""

    png: bytes
    w_pt: float
    h_pt: float
    depth_pt: float


OBJ = "\ufffc"  # text of a run that is an equation or a formula picture


@dataclass
class _Grid:
    x0: int
    y0: int
    x1: int
    y1: int
    rows: list[int]
    cols: list[int]


# --- rendering ---------------------------------------------------------------


def choose_dpi(area: pymupdf.Rect) -> int:
    long_side_in = max(area.width, area.height) / 72
    return max(72, min(OCR_DPI, int(OCR_MAX_SIDE_PX / long_side_in)))


def render_gray(page: pymupdf.Page, dpi: int, clip: pymupdf.Rect | None = None) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return img[:, : pix.width].copy()


def render_rgb(page: pymupdf.Page, dpi: int, clip: pymupdf.Rect | None = None) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return img[:, : pix.width * 3].reshape(pix.height, pix.width, 3).copy()


def _tesseract_env() -> dict:
    return {**os.environ, "TESSDATA_PREFIX": str(TESSDATA_DIR), "OMP_THREAD_LIMIT": "1"}


def upright_turns(gray: np.ndarray, dpi: int) -> int:
    """np.rot90 turns that make the page upright, from Tesseract's orientation
    detection (run at half resolution). 0 when unsure or too little text."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "osd.png")
        cv2.imwrite(path, small)
        cmd = [TESSERACT_CMD, path, "stdout", "--psm", "0", "--dpi", str(dpi // 2)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_tesseract_env(), timeout=120)
    rotate = re.search(r"Rotate: (\d+)", proc.stdout)
    conf = re.search(r"Orientation confidence: ([\d.]+)", proc.stdout)
    if proc.returncode != 0 or not rotate or not conf or float(conf.group(1)) < 2:
        return 0
    # "Rotate: 90" means the image must turn 90 degrees clockwise.
    return -int(rotate.group(1)) // 90 % 4


def skew_angle(gray: np.ndarray) -> float:
    """Rotation (degrees, counter-clockwise) that makes text lines horizontal:
    the angle maximising the variance of row ink sums."""
    h, w = gray.shape
    scale = min(1.0, 1000 / max(w, h))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sh, sw = bw.shape

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1)
        rotated = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1, dtype=np.int64)))

    best = max(np.arange(-5, 5.01, 0.5), key=score)
    return float(max(np.arange(best - 0.5, best + 0.51, 0.1), key=score))


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    white = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=white)


def drop_light_ink(gray: np.ndarray) -> bool:
    """Whiten pale watermarks and noise in place. Returns True if applied."""
    ink = gray[gray < 200]
    # Faded scans, whose text itself is light grey, are left alone.
    if ink.size < 0.002 * gray.size or np.median(ink) > 120:
        return False
    gray[gray >= LIGHT_INK_LEVEL] = 255
    return True


# --- tables ------------------------------------------------------------------


def _line_centers(coverage: np.ndarray, length: int, min_frac=0.5, gap=4) -> list[int]:
    idx = np.flatnonzero(coverage >= min_frac * length)
    if not idx.size:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)
    return [int(g.mean()) for g in groups]


def find_tables(gray: np.ndarray) -> tuple[list[_Grid], np.ndarray]:
    """Find fully ruled tables. Returns grids and a mask of their rules."""
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 50)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    rules = cv2.bitwise_or(horiz, vert)
    blobs = cv2.dilate(rules, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grids = []
    rule_mask = np.zeros_like(gray)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.15 or ch < h * 0.02:
            continue
        rows = [y + r for r in _line_centers(horiz[y : y + ch, x : x + cw].sum(axis=1) / 255, cw)]
        cols = [x + k for k in _line_centers(vert[y : y + ch, x : x + cw].sum(axis=0) / 255, ch)]
        if len(rows) < 2 or len(cols) < 2 or (len(rows) - 1) * (len(cols) - 1) < 2:
            continue
        grids.append(_Grid(x, y, x + cw, y + ch, rows, cols))
        # Only erase rules inside tables, so large glyph strokes elsewhere survive.
        rule_mask[y : y + ch, x : x + cw] = rules[y : y + ch, x : x + cw]
    rule_mask = cv2.dilate(rule_mask, np.ones((3, 3), np.uint8))
    return grids, rule_mask


# --- tesseract ---------------------------------------------------------------

_BBOX = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")
_BASELINE = re.compile(r"baseline (-?[\d.]+) (-?[\d.]+)")
_XSIZE = re.compile(r"x_size ([\d.]+)")
_ASC = re.compile(r"x_ascenders ([\d.]+)")
_DESC = re.compile(r"x_descenders ([\d.]+)")
_WCONF = re.compile(r"x_wconf (\d+)")
_LINE_CLASSES = {"ocr_line", "ocr_header", "ocr_caption", "ocr_textfloat"}


def _line_em(title: str, height: int) -> float:
    xs, asc, desc = _XSIZE.search(title), _ASC.search(title), _DESC.search(title)
    by_xsize = float(xs.group(1)) / XSIZE_PER_EM if xs else height / XSIZE_PER_EM
    if not (asc and desc) or float(asc.group(1)) + float(desc.group(1)) <= 0:
        return by_xsize
    return (by_xsize + (float(asc.group(1)) + float(desc.group(1))) / ASC_DESC_PER_EM) / 2


def _is_noise(wd: Word) -> bool:
    text = wd.text
    # Specks from scan noise come back as low-confidence punctuation.
    if wd.conf < 40 and not any(ch.isalnum() for ch in text):
        return True
    # Separators ("-----HẾT-----", "......") are content, even if repetitive;
    # a leader with letters glued on ("tên:.....ồ.ẵ.ÁcNnnnn") is not exempt.
    if re.fullmatch(r"[-._=~*]{3,}\w{0,6}[-._=~*]*|\w{0,6}[-._=~*]{3,}", text):
        return False
    # Watermark strokes over dotted leaders read as long runs of a few letters
    # ("ồ.ồẢồẢCXnnnneeeeee").
    return wd.conf < 60 and len(text) >= 6 and len(set(text.lower())) / len(text) < 0.45


class _HocrParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines: list[Line] = []
        self.block = self.par = 0
        self.word: Word | None = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if self.word is not None:
            self.depth += 1
            return
        a = dict(attrs)
        cls, title = a.get("class", ""), a.get("title", "")
        if cls == "ocr_carea":
            self.block += 1
        elif cls == "ocr_par":
            self.par += 1
        elif cls in _LINE_CLASSES:
            x0, y0, x1, y1 = map(int, _BBOX.search(title).groups())
            bl = _BASELINE.search(title)
            self.lines.append(
                Line(
                    block=self.block,
                    par=self.par,
                    baseline=y1 + (float(bl.group(2)) if bl else 0.0),
                    em_px=_line_em(title, y1 - y0),
                )
            )
        elif cls == "ocrx_word" and self.lines:
            x0, y0, x1, y1 = map(int, _BBOX.search(title).groups())
            conf = _WCONF.search(title)
            self.word = Word(
                x0, y0, x1, y1,
                conf=float(conf.group(1)) if conf else 0.0,
                text="",
                line_no=len(self.lines) - 1,
                em_px=self.lines[-1].em_px,
            )
            self.depth = 0

    def handle_endtag(self, tag):
        if self.word is None:
            return
        if self.depth:
            self.depth -= 1
            return
        wd, self.word = self.word, None
        wd.text = wd.text.strip()
        if wd.text and not _is_noise(wd):
            self.lines[-1].words.append(wd)

    def handle_data(self, data):
        if self.word is not None:
            self.word.text += data


def tesseract_lines(gray: np.ndarray, lang: str, dpi: int, timeout: float = 300) -> list[Line]:
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "page.png")
        cv2.imwrite(img_path, gray)
        cmd = [TESSERACT_CMD, img_path, "stdout", "-l", lang, "--dpi", str(dpi), "--psm", "3", "hocr"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=_tesseract_env(), timeout=timeout)
        except FileNotFoundError as exc:
            raise TaskError("internal_server_error", "tesseract not installed") from exc
    if proc.returncode != 0:
        raise TaskError("internal_server_error", proc.stderr[-500:])
    parser = _HocrParser()
    parser.feed(proc.stdout)
    for ln in parser.lines:
        for piece in _pieces(ln.words):
            # Dotted leaders under a watermark read as runs of a few letters at
            # near-zero confidence ("teen nen nen sen,").
            letters = {ch.lower() for w in piece for ch in w.text if ch.isalpha()}
            digits = any(ch.isdigit() for w in piece for ch in w.text)
            if len(piece) >= 2 and all(w.conf < 40 for w in piece) and not digits and len(letters) <= 5:
                for w in piece:
                    ln.words.remove(w)
    return [ln for ln in parser.lines if ln.words]


_ROW_LABEL = re.compile(r"^\*?([A-DĐc])[.,]")
_ROW_LETTER = {"Đ": "D", "c": "C"}


def _row_labels(ln: Line) -> list[tuple[int, str]]:
    """(x, letter) of the answer labels of a line: words starting one of its pieces (_pieces) with a label, also
    as Tesseract misreads it ("A.", "*B.", "C.6kPa.", "B,", "c.", "Đ."). A unit after a number ("1 A.",
    "2.10⁻⁶ C.") sits inside its piece and is not a label."""
    found = []
    for piece in _pieces(ln.words):
        if m := _ROW_LABEL.match(piece[0].text):
            found.append((piece[0].x0, _ROW_LETTER.get(m.group(1), m.group(1))))
    return found


def join_rows(lines: list[Line], height: int) -> None:
    """Join lines Tesseract split off a printed row ("A." alone as a block of its own, read after the rest of
    its row "B. ... C. ... D. ..."): two lines side by side (ROW_SHARE, ROW_BASELINE_EM), one of them the only
    line of its block and lying within the other's column (the line above it belongs to the other line's
    block, or both lines are blocks of one line reading A-D whole), and together one row of answer options:
    the lone line starts with a label and holds no other, and the labels of both read consecutive letters left
    to right ("A" + "BCD", "C" + "D"). Lines of two columns, also option rows of two questions side by side
    ("A. 10 m." beside "C. 25 N."), stay apart. The words of the line with fewer words join the other line in x
    order; that line keeps its place in the reading order. Lines in the header and footer bands of the page
    (HEADER_ZONE) are left alone. Words keep their line_no."""
    blocks: dict[int, list[Line]] = {}
    for ln in lines:
        blocks.setdefault(ln.block, []).append(ln)

    def rows(ln: Line) -> tuple[int, int]:
        return min(w.y0 for w in ln.words), max(w.y1 for w in ln.words)

    def within(ln: Line, other: Line) -> bool:  # ln alone in its block, in the other's column, one option row
        if len(blocks[ln.block]) != 1:
            return False
        own, theirs = _row_labels(ln), _row_labels(other)
        if len(own) != 1 or not theirs or own[0][0] != min(w.x0 for w in ln.words):
            return False
        letters = "".join(c for _, c in sorted(own + theirs))
        if letters not in "ABCD":  # consecutive: "A." beside "C." is another column
            return False
        # the line just above ln belongs to the other line's block (ln is a hole in that block, not the top of
        # another column under its own question); or the row is a block of one line too and reads A-D whole
        top = rows(ln)[0]
        above = [o for o in lines if o is not ln and o.x0 < ln.x1 and ln.x0 < o.x1 and rows(o)[1] <= top]
        nearest = max(above, key=lambda o: rows(o)[1], default=None)
        if nearest is not None and nearest.block == other.block:
            return True
        return len(blocks[other.block]) == 1 and letters == "ABCD"

    def side_by_side(a: Line, b: Line) -> bool:
        (ay0, ay1), (by0, by1) = rows(a), rows(b)
        if a.block == b.block or min(a.x1, b.x1) >= max(a.x0, b.x0):
            return False
        if min(ay0, by0) < HEADER_ZONE * height or max(ay1, by1) > (1 - HEADER_ZONE) * height:
            return False
        share = min(ay1, by1) - max(ay0, by0)
        short, tall = sorted((ay1 - ay0, by1 - by0))
        return (
            share >= ROW_SHARE * short and share >= ROW_SHARE_TALL * tall
            and abs(a.baseline - b.baseline) <= ROW_BASELINE_EM * min(a.em_px, b.em_px)
            and (within(a, b) or within(b, a))
        )

    pairs = sorted(
        ((abs(a.baseline - b.baseline), i, j) for i, a in enumerate(lines) for j, b in enumerate(lines) if i < j and side_by_side(a, b)),
    )
    gone: set[int] = set()
    for _, i, j in pairs:
        if i in gone or j in gone:
            continue
        a, b = lines[i], lines[j]
        if min(a.x1, b.x1) >= max(a.x0, b.x0):  # a line that took words already may reach the other one now
            continue
        keep, part = (a, j) if (len(a.words), a.x1 - a.x0) >= (len(b.words), b.x1 - b.x0) else (b, i)
        keep.words = sorted(keep.words + lines[part].words, key=lambda w: w.x0)
        gone.add(part)
    lines[:] = [ln for k, ln in enumerate(lines) if k not in gone]


def _measure_weight(words: list[Word], gray: np.ndarray) -> None:
    words = [w for w in words if w.math is None]  # formula placeholders stay regular
    if not words:
        return
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    body = median(w.em_px for w in words)
    for wd in words:
        box = bw[wd.y0 : wd.y1, wd.x0 : wd.x1]
        ink = int(box.sum())
        if not ink:
            continue
        # Tesseract sizes a line of short words ("A. 6 kPa.") small; a thinner
        # erosion then keeps regular stems and the line reads bold.
        k = max(2, round(0.11 * max(wd.em_px, 0.9 * body)))
        wd.weight = int(cv2.erode(box, np.ones((k, k), np.uint8)).sum()) / ink
    typical = median(w.weight for w in words)
    for wd in words:
        ratio = BOLD_WEIGHT_RATIO_SHORT if len(wd.text) <= 2 else BOLD_WEIGHT_RATIO
        wd.bold = wd.weight >= max(ratio * typical, BOLD_WEIGHT_FLOOR)


_NUMBER_WORD = re.compile(r"^[(–-]?\d+(?:[,.]\d+)*[%°]?[.,;:)]?$")
_ITEM_NUMBER = re.compile(r"^\d+[.:]$")
_LABEL_ONLY = re.compile(r"^\*?[A-D]\.$")
_ITEM_WORD = re.compile(r"^(Câu|Bài)$")


def _tidy_bold(lines: list[Line]) -> None:
    """Weight is noisy on short words: answer labels ("A.") and "Câu" of a page
    share the bold of most of them (an exam styles them alike), a number in
    plain text ("7,5", "200": Times digits have heavy strokes) is not bold
    unless most of its line is, and the number of "Câu 3:" is bold exactly
    when "Câu" is."""
    for pattern in (_LABEL_ONLY, _ITEM_WORD):
        alike = [w for ln in lines for w in ln.words if w.math is None and pattern.match(w.text)]
        if len(alike) >= 3:
            bold = sum(w.bold for w in alike) * 2 >= len(alike)
            for wd in alike:
                wd.bold = bold
    for ln in lines:
        words = sorted(ln.words, key=lambda w: w.x0)
        others = [w for w in words if w.math is None and not _NUMBER_WORD.match(w.text)]
        mostly_bold = bool(others) and _bold_share(others) >= 0.5
        for prev, wd in zip([None, *words], words):
            if wd.math is not None:
                continue
            if prev is not None and _ITEM_NUMBER.match(wd.text) and prev.text in ("Câu", "Bài", "Ví"):
                wd.bold = prev.bold
            elif _NUMBER_WORD.match(wd.text) and not mostly_bold:
                wd.bold = False


# --- layout ------------------------------------------------------------------


def _pieces(words: list[Word]) -> list[list[Word]]:
    """Words of one line split at gaps wider than 1em (word gaps are ~0.3em,
    tab gaps between answer options ~1.1em): tab stops, labels."""
    pieces: list[list[Word]] = []
    for wd in sorted(words, key=lambda w: w.x0):
        if pieces and wd.x0 - pieces[-1][-1].x1 <= 1.0 * wd.em_px:
            pieces[-1].append(wd)
        else:
            pieces.append([wd])
    return pieces


def _split_paragraphs(lines: list[Line]) -> list[list[Line]]:
    """Split one Tesseract paragraph into real ones: where the baseline pitch
    jumps, where a line ended early (the next word would have fitted), where a
    line starts indented, and around rows of tab-separated items."""
    pitches = [b.baseline - a.baseline for a, b in pairwise(lines)]
    typical = median(pitches) if len(pitches) >= 2 else None
    right = max(ln.x1 for ln in lines)
    paras, current = [], [lines[0]]
    for (prev, line), pitch in zip(pairwise(lines), pitches, strict=True):
        em = median([prev.em_px, line.em_px])
        limit = PARAGRAPH_PITCH * XSIZE_PER_EM * em
        if typical and typical > limit:  # widely spaced text, e.g. double spacing
            limit = 1.3 * typical
        first = min(line.words, key=lambda w: w.x0)
        last = max(prev.words, key=lambda w: w.x1)
        new = (
            # A sentence end followed by a capital starts anew ("... phương án." / "Câu 1:").
            (last.text.endswith((".", ":", "?", "!", ";")) and first.text[:1].isupper())
            or
            pitch > limit
            or pitch < 0
            or len(_pieces(prev.words)) > 1
            or len(_pieces(line.words)) > 1
            or line.x0 > prev.x0 + em
            or prev.x1 + 0.3 * em + (first.x1 - first.x0) < right - 0.5 * em
        )
        if new:
            paras.append(current)
            current = []
        current.append(line)
    paras.append(current)
    return paras


def _runs(lines: list[Line]) -> tuple[list[Run], list[list[Word]]]:
    """Runs of consecutive words sharing boldness ("Câu 1:" bold, the rest
    not). A single-line paragraph with wide gaps becomes tab-separated pieces;
    words hyphenated across lines are rejoined."""
    pieces_of_row = _pieces(lines[0].words) if len(lines) == 1 else []
    if len(pieces_of_row) > 1:
        rows = [pieces_of_row]
    else:
        rows = [[sorted(ln.words, key=lambda w: w.x0)] for ln in lines]
    runs: list[Run] = []
    for li, pieces in enumerate(rows):
        for pi, piece in enumerate(pieces):
            if pi:
                runs.append(Run(TAB))
            for wi, wd in enumerate(piece):
                text, sep = wd.text, "" if wi == 0 and (pi or li == 0) else " "
                if wd.math is not None:
                    parts = _formula_runs(wd.math)
                    if not parts:
                        continue
                    if sep and _plain(parts[0]):
                        parts[0].text = sep + parts[0].text
                    elif sep:
                        parts.insert(0, Run(sep, formula=True))
                    runs += parts
                    continue
                hyphen = runs and not runs[-1].formula and runs[-1].text.endswith("-")
                if wi == 0 and li and hyphen and text[:1].islower():
                    runs[-1].text, sep = runs[-1].text[:-1], ""
                bold = wd.bold
                stamp = len(pieces) > 1 and (
                    all(w.faded for w in piece) or looks_like_link(" ".join(w.text for w in piece))
                )
                if (
                    runs and runs[-1].text != TAB and not runs[-1].formula
                    and (runs[-1].bold, runs[-1].stamp, runs[-1].color) == (bold, stamp, wd.color)
                ):
                    runs[-1].text += sep + text
                else:
                    runs.append(Run(sep + text, bold=bold, stamp=stamp, color=wd.color))
    return runs, pieces_of_row if len(pieces_of_row) > 1 else []


def _plain(run: Run) -> bool:
    return not (run.script or run.math or run.image or run.field)


def _formula_runs(m) -> list[Run]:
    """Runs of a formula placeholder: text with superscripts/subscripts, a Word equation and
    the text after it (unit, full stop), or a picture. Regular weight, automatic colour."""
    if isinstance(m, FormulaPicture):
        return [Run(OBJ, image=m.png, image_w=m.w_pt, image_h=m.h_pt, image_depth=m.depth_pt, formula=True)]
    if m.kind == "eq":
        runs = [Run(OBJ, math=m.latex, math_text=m.latex, formula=True)]
        return runs + ([Run(m.tail, formula=True)] if m.tail else [])
    return [Run(text, script=script, formula=True) for text, script in m.parts if text]


def _placeholder_text(m) -> str:
    """What a placeholder word reads as for the layout rules (sentence ends, bold share)."""
    if isinstance(m, FormulaPicture):
        return OBJ
    if m.kind == "eq":
        return OBJ + m.tail
    return "".join(text for text, _ in m.parts) or OBJ


def _mark_faded(lines: list[Line], rgb: np.ndarray, gray: np.ndarray) -> None:
    """Flag words printed in light grey (translucent black, as sites stamp
    their name). Coloured text keeps its chroma and is never flagged."""
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for ln in lines:
        for wd in ln.words:
            mask = ink[wd.y0 : wd.y1, wd.x0 : wd.x1].astype(bool)
            if mask.sum() < 10:
                continue
            color = np.median(rgb[wd.y0 : wd.y1, wd.x0 : wd.x1][mask], axis=0)
            luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            wd.faded = luma >= FADED_LUMA and color.max() - color.min() < FADED_CHROMA


_LABEL_WORD = re.compile(r"^(\*?[A-D]\.) (\S.*)$")


def _label_cut(wd: Word, ink: np.ndarray, rgb: np.ndarray) -> int | None:
    """Column where an answer label glued to its option ("A.6kPa.") ends: the
    middle of the first space-wide blank (>=0.1em; the gap between "A" and its
    dot is narrower) 0.5-1.8em into the word, else where coloured ink turns
    black (a blue bold label before plain text); None when neither is seen."""
    box = ink[wd.y0 : wd.y1, wd.x0 : wd.x1].astype(bool)
    cols = box.any(axis=0)
    lo, hi = int(0.5 * wd.em_px), min(int(1.8 * wd.em_px), box.shape[1] - 1)
    start = None
    for x in range(lo, hi + 1):
        if not cols[x] and cols[:x].any():
            start = x if start is None else start
        elif start is not None:
            if x - start >= max(2, 0.1 * wd.em_px) and cols[x:].any():
                return wd.x0 + (start + x) // 2
            start = None
    px = rgb[wd.y0 : wd.y1, wd.x0 : wd.x1].astype(int)
    chroma = px.max(axis=2) - px.min(axis=2)
    coloured = [bool(box[:, x].any()) and np.median(chroma[:, x][box[:, x]]) >= COLOUR_CHROMA for x in range(box.shape[1])]
    for x in range(lo, hi):
        if cols[x] and not coloured[x] and any(coloured[:x]) and not any(coloured[x : x + int(0.3 * wd.em_px)]):
            return wd.x0 + x
    return None


def _split_labels(lines: list[Line], ink: np.ndarray, rgb: np.ndarray) -> None:
    """Split words fix_word found to be a label glued to its option ("A. 6kPa."
    from one Tesseract word) at the gap or colour change, so the label's weight
    and colour are measured on their own."""
    for ln in lines:
        words = []
        for wd in ln.words:
            m = _LABEL_WORD.match(wd.text)
            cut = _label_cut(wd, ink, rgb) if m else None
            if cut is None:
                words.append(wd)
                continue
            words += [replace(wd, x1=cut, text=m.group(1)), replace(wd, x0=cut, text=m.group(2))]
        ln.words = words


def _mark_colour(lines: list[Line], ink: np.ndarray, rgb: np.ndarray) -> None:
    """Colour of each word from its darkest ink (edges blend into the paper);
    near colours of the page are merged so a line keeps one blue."""
    found: list[tuple[Word, np.ndarray]] = []
    for ln in lines:
        for wd in ln.words:
            mask = ink[wd.y0 : wd.y1, wd.x0 : wd.x1].astype(bool)
            if wd.faded or mask.sum() < 10:
                continue
            px = rgb[wd.y0 : wd.y1, wd.x0 : wd.x1][mask].astype(float)
            luma = px @ LUMA
            core = px[luma <= np.percentile(luma, 40)]
            colour = np.median(core, axis=0)
            if colour.max() - colour.min() >= COLOUR_CHROMA:
                found.append((wd, colour))
    groups: list[list[tuple[Word, np.ndarray]]] = []
    for item in found:
        near = next((g for g in groups if np.abs(np.median([c for _, c in g], axis=0) - item[1]).max() < COLOUR_MERGE), None)
        (near.append(item) if near else groups.append([item]))
    for g in groups:
        colour = np.median([c for _, c in g], axis=0)
        for wd, _ in g:
            wd.color = "".join(f"{round(float(v)):02X}" for v in colour)


def _bold_share(words: list[Word]) -> float:
    chars = sum(len(w.text) for w in words)
    return sum(len(w.text) for w in words if w.bold) / chars if chars else 0.0


def _paragraph(lines: list[Line], s: float) -> Paragraph:
    words = [w for ln in lines for w in ln.words]
    pitches = [b.baseline - a.baseline for a, b in pairwise(lines)]
    font_pt = median(ln.em_px for ln in lines) * s
    # Glyph-height metrics go wrong when Tesseract merged a line with a
    # drawing; the last line (natural spacing) gives a width-based check.
    by_width = _line_width_size(lines[-1], s)
    if by_width and abs(font_pt - by_width) > 0.25 * by_width:
        font_pt = by_width
    runs, pieces = _runs(lines)
    stops = [(piece[0].x0 * s, "left") for piece in pieces[1:]]
    return Paragraph(
        runs=runs,
        tab_stops=stops,
        text="".join(r.text for r in runs),
        bold=_bold_share(words) >= BOLD_SHARE,
        translucent=all(w.faded for w in words),
        x0=min(w.x0 for w in words),
        y0=min(w.y0 for w in words),
        x1=max(w.x1 for w in words),
        y1=max(w.y1 for w in words),
        font_pt=font_pt,
        line_pitch_pt=median(pitches) * s if pitches else None,
        line_spans=[(ln.x0, ln.x1) for ln in lines],
        # Word sets the recognised text a little wider than the scan's ink; give
        # the longest line that much room so line breaks stay put.
        natural_right=(max(ln.x1 for ln in lines) + 0.03 * max(ln.x1 - ln.x0 for ln in lines)) * s,
    )


def _cell_text(words: list[Word]) -> str:
    by_line: dict[int, list[Word]] = {}
    for wd in words:
        by_line.setdefault(wd.line_no, []).append(wd)
    rows = sorted(by_line.values(), key=lambda ws: min(w.y0 for w in ws))
    return "\n".join(" ".join(w.text for w in sorted(ws, key=lambda w: w.x0)) for ws in rows)


def _paper_colour(pixels: np.ndarray) -> np.ndarray | None:
    """Colour of the paper among RGB pixels: their light majority, ink left out."""
    if pixels.shape[0] < 20:
        return None
    luma = pixels @ LUMA
    return np.median(pixels[luma >= np.percentile(luma, 75) - 30], axis=0)


def _shading(grid: _Grid, rgb: np.ndarray, dpi: int) -> list[list[str | None]]:
    """Fill colour ("RRGGBB") of each cell of a ruled table, None for plain paper.
    Colours are taken relative to the paper in a band around the table, so a
    yellowish or grey scan still gives white cells."""
    h, w = rgb.shape[:2]
    pt = dpi / 72
    near, far, inset = int(3 * pt), int(10 * pt), int(2 * pt)
    sides = [
        rgb[max(grid.y0 - far, 0) : max(grid.y0 - near, 0), grid.x0 : grid.x1],
        rgb[min(grid.y1 + near, h) : min(grid.y1 + far, h), grid.x0 : grid.x1],
        rgb[grid.y0 : grid.y1, max(grid.x0 - far, 0) : max(grid.x0 - near, 0)],
        rgb[grid.y0 : grid.y1, min(grid.x1 + near, w) : min(grid.x1 + far, w)],
    ]
    papers = [p for p in (_paper_colour(side.reshape(-1, 3)) for side in sides) if p is not None]
    if not papers:
        return [[None] * (len(grid.cols) - 1) for _ in range(len(grid.rows) - 1)]
    paper = np.maximum(np.median(papers, axis=0), 1)
    lumas = [float(p @ LUMA) for p in papers]
    threshold = SHADE_MIN + max(lumas) - min(lumas)
    out = []
    for y0, y1 in pairwise(grid.rows):
        row = []
        for x0, x1 in pairwise(grid.cols):
            colour = _paper_colour(rgb[y0 + inset : y1 - inset, x0 + inset : x1 - inset].reshape(-1, 3))
            fill = None
            if colour is not None:
                rel = np.clip(colour / paper * 255, 0, 255)
                if rel.max() - rel.min() <= 3:  # grey with a scanner's tint noise: F2F1F1 -> F2F2F2
                    rel[:] = rel.mean()
                if (255 - rel).max() >= threshold:
                    fill = "".join(f"{round(v):02X}" for v in rel)
            row.append(fill)
        out.append(row)
    return out


def _table(grid: _Grid, words: list[Word], s: float, fills: list[list[str | None]] | None = None) -> Table:
    nrows, ncols = len(grid.rows) - 1, len(grid.cols) - 1
    buckets: list[list[list[Word]]] = [[[] for _ in range(ncols)] for _ in range(nrows)]
    for wd in words:
        r = min(max(bisect_right(grid.rows, wd.cy) - 1, 0), nrows - 1)
        c = min(max(bisect_right(grid.cols, wd.cx) - 1, 0), ncols - 1)
        buckets[r][c].append(wd)

    def spans(ws: list[Word]) -> list[list[float]]:
        lines: dict[int, list[Word]] = {}
        for wd in ws:
            lines.setdefault(wd.line_no, []).append(wd)
        return [[min(w.x0 for w in line), max(w.x1 for w in line)] for line in lines.values()]

    cells = [
        [
            Cell(
                text=_cell_text(ws),
                bold=_bold_share(ws) >= 0.5,
                shading=fills[r][c] if fills else None,
                align=cell_align(spans(ws), grid.cols[c], grid.cols[c + 1]),
            )
            for c, ws in enumerate(row)
        ]
        for r, row in enumerate(buckets)
    ]
    fix_table(cells)
    sizes = [w.em_px for w in words]
    return Table(
        cells=cells,
        col_widths_pt=[(b - a) * s for a, b in pairwise(grid.cols)],
        row_heights_pt=[(b - a) * s for a, b in pairwise(grid.rows)],
        x0=grid.x0,
        y0=grid.y0,
        x1=grid.x1,
        y1=grid.y1,
        font_pt=median(sizes) * s if sizes else 11.0,
    )


_TIMES = pymupdf.Font("tiro")


def _base_letters(text: str) -> str:
    """Drop Vietnamese diacritics (their base letters set as wide) so the
    built-in Latin-1 Times can measure the text."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))


def _line_width_size(line: Line, s: float) -> float | None:
    """Size at which Times sets this line's text as wide as its ink."""
    words = sorted(line.words, key=lambda w: w.x0)
    text = " ".join(w.text for w in words)
    if len(text) < 15 or len(_pieces(words)) > 1 or any(w.math is not None for w in words):
        return None
    unit = _TIMES.text_length(_base_letters(text), 1)
    return (words[-1].x1 - words[0].x0) * s / unit if unit else None


def _size_from_widths(lines: list[Line], s: float) -> float | None:
    """Body size that makes Times text as wide as the scan's lines. Width is
    what decides Word's line breaks, so this beats glyph-height estimates.
    Uses lines set with natural spacing: no bold, no tab gaps, not filling the
    column (justified lines are stretched)."""
    right = max((ln.x1 for ln in lines), default=0)
    sizes = [
        size
        for ln in lines
        if ln.x1 <= right - ln.em_px and not any(w.bold for w in ln.words)
        and len(" ".join(w.text for w in ln.words)) >= 20
        and (size := _line_width_size(ln, s))
    ]
    return median(sizes) if len(sizes) >= 3 else None


def _weighted_median(pairs: list[tuple[float, int]]) -> float:
    pairs = sorted(pairs)
    half = sum(w for _, w in pairs) / 2
    acc = 0
    for value, weight in pairs:
        acc += weight
        if acc >= half:
            return value
    return pairs[-1][0]


def _snap(size: float, body: float, tolerance: float = 0.15) -> float:
    size = body if abs(size - body) <= tolerance * body else size
    return round(size * 2) / 2


def build_layout(
    lines: list[Line], grids: list[_Grid], gray: np.ndarray, dpi: int, rect: pymupdf.Rect,
    rgb: np.ndarray | None = None,
) -> PageLayout:
    s = 72 / dpi
    words = [w for ln in lines for w in ln.words]
    page = PageLayout(
        width_pt=rect.width, height_pt=rect.height, source="ocr", word_count=len(words)
    )
    page.conf_sum = sum(w.conf for w in words)
    if not words:
        return page
    _measure_weight(words, gray)
    _tidy_bold(lines)

    def table_of(wd: Word) -> int | None:
        for i, g in enumerate(grids):
            if g.x0 <= wd.cx <= g.x1 and g.y0 <= wd.cy <= g.y1:
                return i
        return None

    table_words: list[list[Word]] = [[] for _ in grids]
    groups: dict[tuple[int, int], list[Line]] = {}
    for ln in lines:
        free = []
        for wd in ln.words:
            i = table_of(wd)
            (table_words[i] if i is not None else free).append(wd)
        if free:
            groups.setdefault((ln.block, ln.par), []).append(
                Line(ln.block, ln.par, ln.baseline, ln.em_px, free)
            )

    paras = [
        _paragraph(part, s) for group in groups.values() for part in _split_paragraphs(group)
    ]
    # One to three faded characters are watermark strokes read as letters.
    paras = [p for p in paras if not (p.translucent and len(p.text.strip()) <= 3)]
    tables = [
        _table(g, ws, s, _shading(g, rgb, dpi) if rgb is not None else None)
        for g, ws in zip(grids, table_words, strict=True)
    ]

    # Body size = size of the paragraphs holding most characters, refined by
    # the line widths when both agree roughly.
    body = _weighted_median([(p.font_pt, len(p.text)) for p in paras] or [(11.0, 1)])
    by_width = _size_from_widths(lines, s)
    if by_width and abs(by_width - body) < 0.2 * body:
        body = round(by_width * 2) / 2

    left = min([p.x0 for p in paras] + [t.x0 for t in tables])
    right = max([p.x1 for p in paras] + [t.x1 for t in tables])
    for p in paras:
        p.font_pt = _snap(p.font_pt, body, BODY_SIZE_TOLERANCE)
        p.bold = p.bold or p.font_pt >= 1.2 * body
        for r in p.runs:
            if r.text != TAB:
                r.size = p.font_pt
                if not r.formula:
                    r.bold = r.bold or p.bold
        if p.tab_stops:
            continue  # tab rows stay left aligned
        p.align, indent_px = alignment(p.line_spans, left, right)
        indent = indent_px * s
        p.first_line_indent_pt = indent if abs(indent) > 3 else 0.0
    for t in tables:
        t.font_pt = _snap(t.font_pt, body, tolerance=0.05)

    # Keep Tesseract's reading order; slot each table in by vertical position.
    elements: list = list(paras)
    for t in sorted(tables, key=lambda t: t.y0):
        idx = next((i for i, e in enumerate(elements) if e.y0 > t.y0), len(elements))
        elements.insert(idx, t)

    prev_bottom = None
    for el in elements:
        if prev_bottom is not None and el.y0 > prev_bottom:
            el.space_before_pt = min(max((el.y0 - prev_bottom) * s - 0.3 * el.font_pt, 0), 72)
        prev_bottom = el.y1
        for attr in ("x0", "y0", "x1", "y1"):
            setattr(el, attr, getattr(el, attr) * s)
        if isinstance(el, Paragraph):  # header/footer candidates, as for native pages
            el.link = looks_like_link(el.text)
            middle = (el.y0 + el.y1) / 2
            if middle < HEADER_ZONE * page.height_pt:
                el.zone = "top"
            elif middle > (1 - HEADER_ZONE) * page.height_pt:
                el.zone = "bottom"
    page.elements = elements
    return page


# --- pictures ----------------------------------------------------------------


def _credible(wd: Word) -> bool:
    if wd.math is not None:
        return True  # a formula read or kept as a picture: never part of a figure
    return wd.conf >= 85 or (wd.conf >= 50 and sum(ch.isalnum() for ch in wd.text) >= 2)


def find_pictures(gray: np.ndarray, lines: list[Line], grids: list[_Grid], dpi: int) -> list[tuple]:
    """Boxes of ink that is neither recognised text nor table rules: figures,
    photos, logos, signatures, stamps. `gray` must still hold light ink."""
    h, w = gray.shape
    bw = np.where(gray < PICTURE_INK_LEVEL, 255, 0).astype(np.uint8)
    for ln in lines:
        for wd in ln.words:
            # Strokes of a drawing read as stray "words" ("|", "C") must stay
            # visible here, or the drawing vanishes with them.
            if not _credible(wd):
                continue
            pad = int(0.15 * wd.em_px)
            bw[max(wd.y0 - pad, 0) : wd.y1 + pad, max(wd.x0 - pad, 0) : wd.x1 + pad] = 0
    for g in grids:
        bw[g.y0 : g.y1, g.x0 : g.x1] = 0
    # Long thin rules (underlines, separators) are not pictures; long solid
    # shapes (bars in a drawing) survive a square opening and are kept.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 50)))
    rules = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    thick = max(3, int(0.03 * dpi))
    solid = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((thick, thick), np.uint8))
    bw[(rules > 0) & (solid == 0)] = 0
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # specks
    k = max(9, int(0.06 * dpi))
    blobs = cv2.dilate(bw, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_side = PICTURE_MIN_SIDE_IN * dpi
    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_side or ch < min_side or cw * ch < PICTURE_MIN_AREA * w * h:
            continue
        if bw[y : y + ch, x : x + cw].mean() / 255 < 0.01:
            continue
        boxes.append((x, y, x + cw, y + ch))
    return boxes


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise TaskError("internal_server_error", "PNG encoding failed")
    return buf.tobytes()


def _with_labels(box: tuple, lines: list[Line], dpi: int) -> tuple:
    """Grow a picture box over short lines (at most 3 words) right next to it:
    the labels of a diagram, which also pulls in faint strokes beside them."""
    x0, y0, x1, y1 = box
    reach = 0.3 * dpi
    for ln in lines:
        # Tesseract often glues a label onto the body line beside the figure;
        # split lines at wide gaps and judge each piece.
        for piece in _pieces(ln.words):
            if len(piece) > 3:
                continue
            lx0, ly0 = min(w.x0 for w in piece), min(w.y0 for w in piece)
            lx1, ly1 = max(w.x1 for w in piece), max(w.y1 for w in piece)
            if max(lx0 - x1, x0 - lx1, 0) > reach or max(ly0 - y1, y0 - ly1, 0) > reach:
                continue
            grown = (min(x0, lx0), min(y0, ly0), max(x1, lx1), max(y1, ly1))
            # Taking the label must not swallow any other text.
            if not any(
                grown[0] <= w.cx <= grown[2] and grown[1] <= w.cy <= grown[3]
                and not (x0 <= w.cx <= x1 and y0 <= w.cy <= y1) and w not in piece and _credible(w)
                for other in lines for w in other.words
            ):
                x0, y0, x1, y1 = grown
    return x0, y0, x1, y1


def _take_pictures(
    lines: list[Line], boxes: list[tuple], rgb: np.ndarray, s: float, dpi: int
) -> list[Picture]:
    """Turn boxes into pictures; recognised words inside a picture (labels of
    a diagram) go with it. A box that is mostly words is left as text."""
    pictures = []
    for box in boxes:
        x0, y0, x1, y1 = _with_labels(box, lines, dpi)
        inside = [
            (ln, wd) for ln in lines for wd in ln.words if x0 <= wd.cx <= x1 and y0 <= wd.cy <= y1
        ]
        word_area = sum((wd.x1 - wd.x0) * (wd.y1 - wd.y0) for _, wd in inside)
        if word_area > 0.3 * (x1 - x0) * (y1 - y0):
            continue
        for ln, wd in inside:
            ln.words.remove(wd)
        pictures.append(Picture(_png(rgb[y0:y1, x0:x1]), x0 * s, y0 * s, x1 * s, y1 * s, scanned=True))
    lines[:] = [ln for ln in lines if ln.words]
    return pictures


# --- formulas ----------------------------------------------------------------


def _converts(latex: str) -> bool:
    try:
        latex_to_omml(latex)
    except ValueError:
        return False
    return True


_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_LETTERS = re.compile(r"[^\W\d_]+")


def _unmarked(decision, words: list[Word]) -> bool:
    """The reading spells a Vietnamese word of the region without its marks ("5 và 3." read
    "5 va 3."): the model has no Vietnamese letters, so the region is text."""
    if decision.kind == "eq":
        plain = _LATEX_COMMAND.sub(" ", decision.latex) + " " + decision.tail
    else:
        plain = " ".join(text for text, script in decision.parts if script is None)
    read = {t.lower() for t in _LETTERS.findall(plain)}
    for w in words:
        if mathregions.vietnamese(w.text):
            if any(t.lower() in read for t in _LETTERS.findall(_base_letters(w.text))):
                return True
    return False


def _overlaps(w: Word, box: tuple) -> bool:
    return min(w.x1, box[2]) > max(w.x0, box[0]) and min(w.y1, box[3]) > max(w.y0, box[1])


# A garbled label glued to its option by OCR ("c.9" for "C. 9"): what follows the label's letter and mark
_GLUED_LABEL = re.compile(r"^[A-Za-zĐđ][.,:;](\S.*)$")


def _relabel(lines: list[Line], region, label) -> None:
    """An unread region whose answer label OCR garbled or lost (region.label): the label written over its glyphs,
    the region's other words kept. A word over the label only goes; one running on into the option ("c.9") keeps
    what follows the label's letter and mark; any other word over the label (or a label read well) leaves the
    line as OCR read it."""
    lx0, ly0, lx1, ly1 = label.box
    width = max(1, lx1 - lx0)
    over = [(ln, w) for ln in lines for w in ln.words
            if _overlaps(w, label.box) and min(w.x1, lx1) - max(w.x0, lx0) >= 0.5 * width]
    edits: list[tuple[Line, Word, Word | None]] = []
    for ln, w in over:
        if w.math is not None or _LABEL_ONLY.match(w.text):
            return
        if w.x1 <= lx1 + 0.3 * (region.em_px or region.line.em_px):
            edits.append((ln, w, None))
            continue
        m = _GLUED_LABEL.match(w.text)
        if m is None:
            return
        edits.append((ln, w, replace(w, x0=max(w.x0, lx1 + 1), text=m.group(1))))
    for ln, w, rest in edits:
        at = next(i for i, o in enumerate(ln.words) if o is w)
        ln.words[at : at + 1] = [rest] if rest is not None else []
    line_no = region.line.words[0].line_no if region.line.words else 0
    mark = Word(lx0, ly0, lx1, ly1, 90.0, label.text, line_no, region.em_px or region.line.em_px, color=label.color)
    at = next((i for i, w in enumerate(region.line.words) if w.x0 > lx0), len(region.line.words))
    region.line.words.insert(at, mark)


def _formula_picture(region, rgb: np.ndarray, gray: np.ndarray, s: float) -> FormulaPicture:
    """The region cut from the colour page (paper whitened where the grey page was), sized in
    points and lowered by how far it reaches below its line's baseline."""
    x0, y0, x1, y1 = region.box
    crop = rgb[y0:y1, x0:x1].copy()
    crop[gray[y0:y1, x0:x1] >= 255] = 255
    depth = max(0.0, (y1 - region.line.baseline) * s)
    return FormulaPicture(_png(crop), (x1 - x0) * s, (y1 - y0) * s, depth)


# How the conversion running in this thread reads formulas (None: settings.FORMULAS); "model" for a careful one.
_formulas: ContextVar[str | None] = ContextVar("formulas", default=None)


@contextmanager
def formulas_mode(how: str | None) -> Iterator[None]:
    """Formula regions of the conversion running in this thread are read `how` ("model", "scripts", "off"; None:
    settings.FORMULAS) while the block runs; other conversions running at the same time keep theirs."""
    token = _formulas.set(how)
    try:
        yield
    finally:
        _formulas.reset(token)


def _default_reader() -> str:
    """How formulas are read without the model: FORMULAS, "scripts" when that is "model"."""
    return FORMULAS if FORMULAS in ("scripts", "off") else "scripts"


def formula_reader() -> str:
    """How this conversion reads formula regions: "model", "scripts" or "off" (formulas_mode(), else
    settings.FORMULAS). Where the model is not installed or the conversion skips it (a long scan, app/convert.py),
    "model" falls back to the default way (_default_reader())."""
    how = _formulas.get() or FORMULAS
    return "model" if how == "model" and mathocr.available() else _default_reader()


def read_formulas(
    lines: list[Line], grids: list[_Grid], gray: np.ndarray, ink: np.ndarray, rgb: np.ndarray, dpi: int,
) -> int:
    """Read the formula regions a detector finds (app/mathregions.py) with the formula model
    (app/mathocr.py), or with Tesseract for superscripts and subscripts only (app/scriptocr.py, formula_reader()).
    Where the reading is trusted, the region's OCR words and every
    other word centred in its box (not an answer label) give way to one placeholder word on the
    region's line holding the reading; an unread "tall" region becomes a placeholder holding its
    picture; other regions keep their OCR words. A region carrying an answer label OCR garbled or
    lost (region.label) also gets that label written in place of the words over the label's glyphs,
    read or not (_relabel); so does an equation read with a label in front ("D.\\frac{3}{2}"): the label
    is written as text before it, not inside the Word equation. When the model fails to load, the page
    is read the default way (formula_reader()).
    Regions in ruled tables are skipped (table cells are plain text). Returns the number of
    placeholders. Nothing happens when formula reading is off."""
    how = formula_reader() if lines else "off"
    reader = mathocr.get_reader() if how == "model" else None
    if how == "model" and reader is None:  # the model failed to load (mathocr marks it failed): the default way
        how = _default_reader()
    if how == "off":
        return 0

    def in_table(box) -> bool:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return any(g.x0 <= cx <= g.x1 and g.y0 <= cy <= g.y1 for g in grids)

    found = mathregions.find(lines, gray, ink, rgb, dpi)
    regions = [r for r in found if any(ln is r.line for ln in lines) and not in_table(r.box)]
    if not regions:
        return 0
    if how == "scripts":
        decisions = scriptocr.read_regions(regions, gray, ink, lines, dpi)
    else:
        decisions = mathocr.read_regions(
            gray, [r.box for r in regions], [r.ocr_text for r in regions], [r.em_px for r in regions], reader,
            [r.case_from_ink for r in regions],
        )
    s = 72 / dpi
    placed = 0
    for region, decision in zip(regions, decisions, strict=True):
        x0, y0, x1, y1 = region.box = tuple(int(v) for v in region.box)
        replaced = {id(w) for w in region.words}
        label = getattr(region, "label", None)
        start = x0  # where the reading goes (after a label the region took in)

        def taken(w: Word) -> bool:
            if w.math is not None or _LABEL_ONLY.match(w.text):
                return False
            inside = x0 <= w.cx <= x1 and y0 <= w.cy <= y1
            on_label = label is not None and _overlaps(w, label.box)
            return id(w) in replaced or inside or on_label

        if decision is not None and decision.kind == "eq" and not _converts(decision.latex):
            decision = None
        if decision is not None and _unmarked(decision, [w for ln in lines for w in ln.words if taken(w)]):
            decision = None
        lead = mathocr.split_label(decision) if label is None else None
        if lead is not None:  # "D.\frac{3}{2}": the label stays text, in front of the equation
            box = mathregions.leading_label(ink, region.box, region.em_px or region.line.em_px)
            if box is None:
                decision = None
            else:
                label, decision, start = mathregions.Label(box, lead[0]), lead[1], box[2] + 1
        if decision is None and region.kind != "tall":
            if label is not None:
                _relabel(lines, region, label)
            continue
        gone: list[Word] = []
        for ln in lines:
            keep = []
            for w in ln.words:
                (gone if taken(w) else keep).append(w)
            ln.words = keep
        if decision is not None:
            payload, conf = decision, round(100 * decision.conf, 1)
        else:
            payload = _formula_picture(region, rgb, gray, s)
            conf = round(sum(w.conf for w in gone) / len(gone), 1) if gone else 0.0
        first = min(gone, key=lambda w: w.x0, default=None)
        line_no = first.line_no if first else (region.line.words[0].line_no if region.line.words else 0)
        em = region.em_px or region.line.em_px
        holder = Word(start, y0, x1, y1, conf, _placeholder_text(payload), line_no, em, math=payload)
        at = next((i for i, w in enumerate(region.line.words) if w.x0 > start), len(region.line.words))
        region.line.words.insert(at, holder)
        if label is not None and not any(_overlaps(w, label.box) for ln in lines for w in ln.words):
            lx0, ly0, lx1, ly1 = label.box
            mark = Word(lx0, ly0, lx1, ly1, 90.0, label.text, line_no, em, color=label.color)
            at = next((i for i, w in enumerate(region.line.words) if w.x0 > lx0), len(region.line.words))
            region.line.words.insert(at, mark)
        placed += 1
    lines[:] = [ln for ln in lines if ln.words]
    return placed


# --- entry points ------------------------------------------------------------


def _ocr(page: pymupdf.Page, lang: str, clip: pymupdf.Rect | None, whole_page: bool) -> PageLayout:
    dpi = choose_dpi(clip or page.rect)
    rgb = render_rgb(page, dpi, clip)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if whole_page:  # straighten scans before anything else
        turns = upright_turns(gray, dpi)
        if turns:
            rgb = np.ascontiguousarray(np.rot90(rgb, turns))
            gray = np.ascontiguousarray(np.rot90(gray, turns))
        angle = skew_angle(gray)
        if abs(angle) >= MIN_DESKEW:
            rgb, gray = rotate_image(rgb, angle), rotate_image(gray, angle)
    grids, rule_mask = find_tables(gray)
    gray[rule_mask > 0] = 255
    with_light_ink = gray.copy() if whole_page else None
    drop_light_ink(gray)
    lines = tesseract_lines(gray, lang, dpi)
    reader = vietocr.reader() if "vie" in lang.split("+") else None
    if reader is not None:
        try:
            vietocr.refine_words(lines, gray, reader)  # words change only once all are read
        except Exception:  # damaged or mismatched model files: Tesseract's words, VietOCR off
            vietocr.broken()
    fix_lines(lines)
    join_rows(lines, gray.shape[0])
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _split_labels(lines, ink, rgb)
    _mark_faded(lines, rgb, gray)
    _mark_colour(lines, ink, rgb)
    # before pictures are cut out, so formula ink is not taken for a figure
    try:
        read_formulas(lines, grids, gray, ink, rgb, dpi)  # words change only once all regions are read
    except Exception:  # damaged or mismatched model files: formula model off, this page read again the default way
        if formula_reader() == "model":
            mathocr.broken()
            try:
                read_formulas(lines, grids, gray, ink, rgb, dpi)
            except Exception:
                log.exception("formula reading failed on a page")
        else:  # the detector or the reader without the model failed on this page: its OCR text
            log.exception("formula reading failed on a page")
    formula_pictures = sum(isinstance(w.math, FormulaPicture) for ln in lines for w in ln.words)
    formulas_read = sum(isinstance(w.math, mathocr.MathDecision) for ln in lines for w in ln.words)
    s = 72 / dpi
    pictures = []
    if whole_page:
        pictures = _take_pictures(lines, find_pictures(with_light_ink, lines, grids, dpi), rgb, s, dpi)
    h, w = gray.shape
    layout = build_layout(lines, grids, gray, dpi, pymupdf.Rect(0, 0, w * s, h * s), rgb)
    layout.pictures = pictures
    layout.formula_pictures = formula_pictures
    layout.formulas_read = formulas_read
    return layout


def ocr_page(page: pymupdf.Page, languages: list[str]) -> PageLayout:
    return _ocr(page, "+".join(languages), None, whole_page=True)


def ocr_region(page: pymupdf.Page, clip: pymupdf.Rect, languages: list[str]) -> PageLayout:
    """OCR part of a page; element coordinates are returned in page space."""
    layout = _ocr(page, "+".join(languages), clip, whole_page=False)
    for el in layout.elements:
        el.x0, el.x1 = el.x0 + clip.x0, el.x1 + clip.x0
        el.y0, el.y1 = el.y0 + clip.y0, el.y1 + clip.y0
        if isinstance(el, Paragraph):
            el.natural_right = el.natural_right and el.natural_right + clip.x0
            el.tab_stops = [(x + clip.x0, kind) for x, kind in el.tab_stops]
    layout.width_pt, layout.height_pt = page.rect.width, page.rect.height
    return layout
