"""Where the formula model should look on an OCR page: formula regions.

A detector is a function

    detector(lines, gray, ink, rgb, dpi) -> list[MathRegion]

called by app/ocr.py once per OCR page (and per raster region of a native page) after
the OCR words are final (Tesseract, VietOCR, fix_lines, labels split off, faded and
colour marked) and before figures are cut out. Arguments, all in the pixels of the
rendered (upright, straightened) page image:

- lines: list[app.ocr.Line] in reading order; each has .baseline, .em_px and .words
  (app.ocr.Word: x0, y0, x1, y1, conf 0-100, text, em_px, line_no). Read only.
- gray: uint8 page image after table rules and light ink were whitened (what the
  formula model reads); ink: uint8 0/1 mask of its dark ink (Otsu); rgb: uint8
  colour image of the same page; dpi: its resolution.

Each MathRegion must carry the box to read (tight around the formula's ink; the reader
cuts it to the ink again), the OCR words it replaces, its kind, the OCR text of those
words, the font size and the line the result is written into (one of `lines`, the
same object). For every region the formula model accepts, app/ocr.py removes the
region's words and every other word whose centre lies in the box (an answer label
"A."-"D." is always kept) and writes the reading in their place on that line. A
region it does not accept keeps its OCR words, except a "tall" one, which becomes a
picture of the box. Regions in ruled tables are skipped.

A region may also carry the answer label in front of it (`label`, a Label): OCR lost it or
garbled it ("c.v=a.", a lowercase "c."), so when the region is read (or kept as a picture)
app/ocr.py writes the label's text there in place of the words over the label's glyphs.

find() runs the detector named by settings.MATH_DETECT (P2W_MATH_DETECT): "rules"
(default), "off", or any name registered in DETECTORS, and drops regions that overlap
(two copies of one box: one is kept).

The "rules" detector (rules()) works from the ink and Tesseract's words:
- answer options: the body after each label "A."-"D." of a row (option_regions); labels
  Tesseract garbled or lost are found as glyphs ("C" and a dot) at the page's label columns
  in the labels' colour or weight (_blob_labels);
- two-dimensional formulas anywhere: a fraction bar (a thin flat stroke with ink just above
  and below), a radical sign or a bar over letters, grown on ink with the neighbours of
  its band up to a gap, a label or a word read as prose (tall_regions);
- formulas in running text: a word with a superscript or subscript (a small glyph raised
  over the caps or lowered under the baseline beside its base glyph: "10^23", "mol^-1",
  "n_1", "v_H"), or a word of a text line read with low confidence that holds a decimal
  number or a math sign (inline_regions).
Options come first, then 2-D formulas outside them, then words outside both. A region's OCR
text gets the case of its first letter from the glyph's height where only the size tells
the cases apart ("Vh" -> "vh" for v_H), so the model's case is checked against the ink.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ocr as _ocr
from .mathocr import CASE_BY_HEIGHT
from .ocr_fix import syllable
from .settings import MATH_DETECT

log = logging.getLogger("pdf2word.mathregions")

KINDS = ("option", "tall", "inline")


@dataclass
class Label:
    """An answer label the reading keeps in front of it, where OCR lost or garbled it."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 of the label's glyphs
    text: str  # "C."
    color: str | None = None  # "RRGGBB" of the page's labels, None for black


@dataclass
class MathRegion:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels (x1, y1 exclusive)
    words: list = field(default_factory=list)  # app.ocr.Word objects the reading replaces
    kind: str = "inline"  # "option": an answer option's body; "tall": 2-D (fraction, root); "inline": in text
    ocr_text: str = ""  # Tesseract's text of `words`, compared with the reading's digits
    em_px: float = 0.0  # font size in pixels (the line's)
    line: object = None  # the app.ocr.Line the result is written into
    label: Label | None = None  # an answer label to write in front of the result (see above)
    case_from_ink: bool = False  # ocr_text's first letter got its case from the glyph's height (_case_by_height)


Detector = Callable[[list, np.ndarray, np.ndarray, np.ndarray, int], list[MathRegion]]


# --- the "rules" detector: bodies of answer options ---

# Growth of an option's box on ink: rightwards until a blank gap this wide (em), leftwards until
# a narrower one (never into the label), up and down to blank rows. A box still on ink this far
# above / below its line's baseline (em) has run into other lines and is dropped; a formula
# reaches about 1.7em above and 1.1em below it (a root over a fraction).
RIGHT_GAP_EM = 0.6
LEFT_GAP_EM = 0.3
MAX_UP_EM = 2.0
MAX_DOWN_EM = 1.4
# A box this many em high is two-dimensional: its OCR words are garbage (numerator and
# denominator read as separate lines), so an unread one is kept as a picture ("tall").
TALL_EM = 1.7
# A fraction bar: a run of ink at least this long (em), at most MARK_EM thick; its numerator and
# denominator start within BAR_GAP_EM of it, past a blank gap the growth on ink does not cross.
BAR_MIN_EM = 1.0
BAR_GAP_EM = 0.9
# A mark over a formula (the bar of a mean value, an arrow): at most MARK_EM thick, past a blank
# gap of at most MARK_GAP_EM.
MARK_EM = 0.15
MARK_GAP_EM = 0.3
VIETNAMESE_CONF = 80  # a word this sure of a Vietnamese letter is text: its region is not read
# OCR reads a label "C." as "c." now and then; in a line with a real label it is one too
_LABEL_LOOSE = re.compile(r"^\*?[A-Da-d]\.$")


def vietnamese(text: str) -> bool:
    """A Latin letter with a diacritic (Vietnamese tone or vowel mark) or đ/Đ."""
    for ch in text:
        if ch in "đĐ":
            return True
        if ch.isalpha() and not ch.isascii():
            base = unicodedata.normalize("NFD", ch)[0]
            if base.isascii() and base.isalpha():
                return True
    return False


def _longest_run(row: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of ink in a row of the mask."""
    padded = np.concatenate(([0], (row > 0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    if not edges.size:
        return 0, 0
    starts, ends = edges[0::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k] - starts[k])


def _beyond_gap(ink: np.ndarray, rows: range, x0: int, x1: int) -> int | None:
    """The first row of `rows` with ink in [x0, x1) after at least one blank row; None if none."""
    blank = False
    for q in rows:
        if ink[q, x0:x1].any():
            if blank:
                return q
        else:
            blank = True
    return None


def _across_bar(
    ink: np.ndarray, x0: int, y0: int, x1: int, y1: int, em: float, top: int, bottom: int,
) -> tuple[int, int]:
    """Fraction bars in the box (at most MARK_EM thick, with a run of ink at least BAR_MIN_EM
    long): the first ink past a blank gap under / over a bar's span, within BAR_GAP_EM of it,
    is its denominator / numerator; the box's bottom / top moves onto it when it lies outside."""
    reach = max(2, int(BAR_GAP_EM * em))
    r = y0
    while r < y1:
        start, length = _longest_run(ink[r, x0:x1])
        if length < BAR_MIN_EM * em:
            r += 1
            continue
        bx0, bx1 = x0 + start, x0 + start + length
        end = r
        while end + 1 < y1 and _longest_run(ink[end + 1, x0:x1])[1] >= BAR_MIN_EM * em:
            end += 1
        if end - r + 1 > max(2, MARK_EM * em):  # a solid shape, not a thin bar
            r = end + 1
            continue
        below = _beyond_gap(ink, range(end + 1, min(bottom, end + 1 + reach)), bx0, bx1)
        if below is not None and below >= y1:
            y1 = below + 1
        above = _beyond_gap(ink, range(r - 1, max(top, r - reach) - 1, -1), bx0, bx1)
        if above is not None and above < y0:
            y0 = above
        r = end + 1
    return y0, y1


def _mark_above(ink: np.ndarray, x0: int, y0: int, x1: int, em: float, top: int) -> int:
    """A thin mark over the box past a small blank gap (the bar of a mean value, an arrow, a dot):
    at most MARK_EM thick, within MARK_GAP_EM, blank above it and not running past the box's
    columns. Returns the box's new top."""
    h, w = ink.shape
    q = _beyond_gap(ink, range(y0 - 1, max(top, y0 - 1 - int(MARK_GAP_EM * em)) - 1, -1), x0, x1)
    if q is None:
        return y0
    first = q
    while first - 1 >= top and ink[first - 1, x0:x1].any():
        first -= 1
        if q - first + 1 > max(2, MARK_EM * em):
            return y0  # thick: a line of text above
    if first <= top:
        return y0
    rows = ink[first : q + 1]
    if (x0 > 0 and rows[:, x0 - 1].any()) or (x1 < w and rows[:, x1].any()):
        return y0
    return first


def grow(
    ink: np.ndarray, box: tuple[int, int, int, int], em: float, left_limit: int,
    right_limit: int | None = None, top_limit: int | None = None, bottom_limit: int | None = None,
    left_gap_em: float = LEFT_GAP_EM,
) -> tuple[tuple[int, int, int, int], bool]:
    """Grow a box on ink until it is stable: up and down while the row next to it has ink (and
    across the gap between a fraction bar and its numerator or denominator, and up to a thin mark
    just over the box), right
    until RIGHT_GAP_EM of blank columns, left until left_gap_em (LEFT_GAP_EM) of blank columns (never
    reaching left_limit). Returns the box and whether it settled within the other limits (False: it is
    still on ink at right_limit, top_limit or bottom_limit)."""
    h, w = ink.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1), False
    top = 0 if top_limit is None else max(0, min(int(top_limit), y0))
    bottom = h if bottom_limit is None else min(h, max(int(bottom_limit), y1))
    gap, lgap = max(2, int(RIGHT_GAP_EM * em)), max(2, int(left_gap_em * em))
    right = w - gap if right_limit is None else min(w - gap, max(int(right_limit), x1))
    while True:
        before = (x0, y0, x1, y1)
        while y0 > top and ink[y0 - 1, x0:x1].any():
            y0 -= 1
        while y1 < bottom and ink[y1, x0:x1].any():
            y1 += 1
        while x1 < right and ink[y0:y1, x1 : x1 + gap].any():
            x1 += 1
        while x0 - lgap > left_limit and ink[y0:y1, x0 - lgap : x0].any():
            x0 -= 1
        if (x0, y0, x1, y1) == before:
            y0, y1 = _across_bar(ink, x0, y0, x1, y1, em, top, bottom)
            y0 = _mark_above(ink, x0, y0, x1, em, top)
            if (x0, y0, x1, y1) == before:
                break
    settled = not (
        (y0 == top and top > 0 and ink[y0 - 1, x0:x1].any())
        or (y1 == bottom and bottom < h and ink[y1, x0:x1].any())
        or (right_limit is not None and x1 >= right and ink[y0:y1, x1 : x1 + gap].any())
    )
    return (x0, y0, x1, y1), settled


def tight(ink: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """The box shrunk to the ink inside it; None when it holds none."""
    x0, y0, x1, y1 = box
    part = ink[y0:y1, x0:x1]
    rows, cols = np.flatnonzero(part.any(axis=1)), np.flatnonzero(part.any(axis=0))
    if not rows.size:
        return None
    return x0 + int(cols[0]), y0 + int(rows[0]), x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1


def _centred(w, box) -> bool:
    return box[0] <= w.cx <= box[2] and box[1] <= w.cy <= box[3]


def _overlap_any(box, boxes) -> bool:
    return any(_overlap(box, b) for b in boxes)


def _body_em(lines: list) -> float:
    """Font size of body text: Tesseract sizes a line merged with a fraction's parts far too big."""
    return float(np.median([w.em_px for ln in lines for w in ln.words]))


# --- page geometry shared by the rules ---

XHEIGHT_EM = 0.47  # x-height of a serif text face (Times) in em, when a line has too few letters to measure


class _Page:
    """Ink components of a page (8-connected), looked up by their centre column, and the text
    metrics of its lines (baseline and x-height measured on their glyphs)."""

    def __init__(self, lines: list, ink: np.ndarray, rgb: np.ndarray | None):
        self.lines, self.ink, self.rgb = lines, ink, rgb
        _, self.lab, st, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(ink, np.uint8), connectivity=8)
        self.st = st.astype(np.int64)
        twice_cx = 2 * self.st[:, 0] + self.st[:, 2]
        self.order = np.argsort(twice_cx[1:], kind="stable") + 1
        self.keys = twice_cx[self.order]
        self.body = _body_em(lines)
        self._metrics: dict[int, tuple[float, float]] = {}
        xhs = [self.metrics(ln)[1] for ln in lines if ln.words]
        self.xh = float(np.median(xhs)) if xhs else XHEIGHT_EM * self.body

    def near(self, x0: float, x1: float) -> np.ndarray:
        """Components whose centre column lies in [x0, x1]."""
        lo = np.searchsorted(self.keys, 2 * x0, "left")
        hi = np.searchsorted(self.keys, 2 * x1, "right")
        return self.order[lo:hi]

    def box(self, c: int) -> tuple[int, int, int, int]:
        x, y, w, h = (int(v) for v in self.st[c, :4])
        return x, y, x + w, y + h

    def em(self, ln) -> float:
        return min(ln.em_px, self.body)

    def metrics(self, ln) -> tuple[float, float]:
        """(baseline, x-height) of a line: the median bottom of its letter-sized glyphs near
        Tesseract's baseline and the median height of the x-height letters standing on it;
        Tesseract's baseline and XHEIGHT_EM when there are too few glyphs."""
        key = id(ln)
        if key not in self._metrics:
            em = self.em(ln)
            base, xh = float(ln.baseline), XHEIGHT_EM * em
            tops, bottoms = [], []
            for w in ln.words:
                for c in self.near(w.x0, w.x1):
                    _, y, _, h, _ = self.st[c]
                    if 0.3 * em <= h <= 0.85 * em and abs(y + h - ln.baseline) <= 0.3 * em:
                        tops.append(int(y))
                        bottoms.append(int(y + h))
            if len(bottoms) >= 3:
                base = float(np.median(bottoms))
                hs = [b - t for t, b in zip(tops, bottoms) if abs(b - base) <= 0.08 * em and 0.32 * em <= b - t <= 0.58 * em]
                if len(hs) >= 3:
                    xh = float(np.median(hs))
            self._metrics[key] = (base, xh)
        return self._metrics[key]

    def union(self, comps) -> tuple[int, int, int, int]:
        boxes = [self.box(c) for c in comps]
        return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)

    def weight(self, c: int, em: float) -> float:
        """Stroke weight of a glyph: the share of its ink an erosion of ~0.11em keeps (as app/ocr.py
        measures bold words)."""
        x0, y0, x1, y1 = self.box(c)
        k = max(2, round(0.11 * em))
        mask = np.pad((self.lab[y0:y1, x0:x1] == c).astype(np.uint8), k)  # paper around it erodes its edges
        total = int(mask.sum())
        return int(cv2.erode(mask, np.ones((k, k), np.uint8)).sum()) / total if total else 0.0

    def colour(self, c: int) -> np.ndarray | None:
        """Colour of a glyph's darkest ink (as app/ocr.py measures a word's colour); None without a colour page."""
        if self.rgb is None:
            return None
        x0, y0, x1, y1 = self.box(c)
        mask = self.lab[y0:y1, x0:x1] == c
        px = self.rgb[y0:y1, x0:x1][mask].astype(float)
        if not len(px):
            return None
        luma = px @ _ocr.LUMA
        return np.median(px[luma <= np.percentile(luma, 40)], axis=0)


# Words that end a formula's growth: read with confidence as prose (a Vietnamese syllable, a word with marks,
# or a longer word of letters that is no unit), answer labels and "Câu"/"Bài".
PROSE_CONF = 80
UNITS = {
    "m", "cm", "mm", "dm", "km", "nm", "g", "kg", "mg", "s", "ms", "h", "min", "Pa", "kPa", "MPa", "hPa", "atm", "J",
    "kJ", "eV", "MeV", "keV", "W", "kW", "MW", "V", "mV", "kV", "A", "mA", "N", "kN", "K", "Hz", "kHz", "MHz", "mol",
    "rad", "T", "Wb", "F", "H", "C", "L", "ml", "mL", "cal", "kcal", "kWh", "dB", "lx", "Ω",
}


def _prose(w) -> bool:
    if _LABEL_LOOSE.match(w.text) or _ocr._ITEM_WORD.match(w.text):
        return True
    if w.conf < PROSE_CONF or w.math is not None:
        return False
    if vietnamese(w.text):
        return True
    core = w.text.strip(".,;:!?()[]\"'“”…")
    if not core.isalpha() or len(core) < 2 or core in UNITS:
        return False
    return syllable(core)[0] or len(core) >= 3


# --- answer labels Tesseract garbled or lost ---

LABEL_COLUMN_EM = 0.5  # a label lies this close to a column where the page's labels stand
LABEL_SIZE = 0.25  # its letter's height within this share of the labels' letter height
LABEL_WEIGHT = 0.6  # at least this share of the labels' stroke weight (when they are not coloured)
LABEL_GAP_EM = 0.6  # and blank paper this wide before it
_LABEL_START = re.compile(r"^\*?([A-Da-d])[.,]")
_GARBLED_LABEL = re.compile(r"^\*?\S[.,]\s*")


@dataclass
class _Style:
    height: float  # the labels' letter height, px
    weight: float  # their median stroke weight
    color: str | None  # their colour when most share one
    columns: list[tuple[float, dict[str, int]]]  # x of a column of labels and the letters seen there


def _letter(page: _Page, w) -> int | None:
    """The letter glyph of a label word: its tallest component."""
    comps = [c for c in page.near(w.x0, w.x1) if w.y0 - 2 <= page.st[c, 1] + page.st[c, 3] / 2 <= w.y1 + 2]
    return max(comps, key=lambda c: page.st[c, 3], default=None)


def _label_style(page: _Page) -> _Style | None:
    """How the page's answer labels look, from those Tesseract read ("A."-"D." alone): None with fewer than 4."""
    words = [w for ln in page.lines for w in ln.words if _ocr._LABEL_ONLY.match(w.text)]
    letters = [(w, c) for w in words for c in [_letter(page, w)] if c is not None]
    if len(letters) < 4:
        return None
    heights = [int(page.st[c, 3]) for _, c in letters]
    weights = [page.weight(c, page.body) for _, c in letters]
    colors = [w.color for w, _ in letters]
    top = max(set(colors), key=colors.count)
    color = top if top is not None and colors.count(top) * 3 >= 2 * len(colors) else None
    columns: list[tuple[float, dict[str, int]]] = []
    for w, _ in sorted(letters, key=lambda t: t[0].x0):
        letter = w.text.lstrip("*")[0]
        if columns and w.x0 - columns[-1][0] <= LABEL_COLUMN_EM * page.body:
            columns[-1][1][letter] = columns[-1][1].get(letter, 0) + 1
        else:
            columns.append((float(w.x0), {letter: 1}))
    return _Style(float(np.median(heights)), float(np.median(weights)), color, columns)


def _blob_at(page: _Page, style: _Style, x: float, base: float, em: float) -> tuple[int, int, int, int] | None:
    """Box of a label drawn at column x of a row with this baseline: a letter as high as the labels' standing
    on the baseline, then a dot, in the labels' colour (or as bold as they are)."""
    glyphs = sorted(
        (c for c in page.near(x - LABEL_COLUMN_EM * em, x + 1.5 * em)
         if abs(page.st[c, 1] + page.st[c, 3] - base) <= 0.2 * em and page.st[c, 3] >= 0.3 * em),
        key=lambda c: page.st[c, 0],
    )
    if not glyphs:
        return None
    c = glyphs[0]
    lx0, ly0, lx1, ly1 = page.box(c)
    if abs(lx0 - x) > LABEL_COLUMN_EM * em or abs((ly1 - ly0) - style.height) > LABEL_SIZE * style.height:
        return None
    if not 0.3 * em <= lx1 - lx0 <= 1.1 * em:
        return None
    before = page.ink[max(0, int(base - 0.8 * em)) : int(base + 0.2 * em), max(0, int(lx0 - LABEL_GAP_EM * em)) : lx0]
    if before.any():
        return None  # a label starts a row or follows a tab gap: "4." of "4,14.10" does not
    dot = None
    for d in page.near(lx1, lx1 + 0.5 * em):
        dx0, dy0, dx1, dy1 = page.box(d)
        if d != c and 0 <= dx0 - lx1 <= 0.35 * em and dx1 - dx0 <= 0.25 * em and dy1 - dy0 <= 0.25 * em \
                and abs(dy1 - ly1) <= 0.12 * em:
            dot = (dx0, dy0, dx1, dy1)
            break
    if dot is None:
        return None
    colour = page.colour(c) if style.color else None
    if colour is not None:
        want = np.array([int(style.color[k : k + 2], 16) for k in (0, 2, 4)], float)
        if np.abs(colour - want).max() >= _ocr.COLOUR_MERGE:
            return None
    elif page.weight(c, page.body) < LABEL_WEIGHT * style.weight:
        return None
    return lx0, min(ly0, dot[1]), dot[2], max(ly1, dot[3])


def _blob_labels(page: _Page, style: _Style | None) -> dict[int, list[tuple]]:
    """Labels of option rows that Tesseract garbled ("c.v=a.", "p.v=—.") or lost, found as glyphs at the
    page's label columns: {id(line): [(box, letter or None, word starting at the label or None)]}. Rows
    looked at: lines with a label read, or whose first word or a tab-separated piece starts at a column."""
    out: dict[int, list[tuple]] = {}
    if style is None:
        return out
    for ln in page.lines:
        em = page.em(ln)
        base, _ = page.metrics(ln)
        pieces = _ocr._pieces(ln.words)
        heads = [p[0] for p in pieces]
        cols = [x for x, _ in style.columns]
        at_col = [h for h in heads if any(abs(h.x0 - x) <= LABEL_COLUMN_EM * em for x in cols)]
        read = [h for h in heads if _LABEL_LOOSE.match(h.text)]
        if not (read or (at_col and (len(pieces) >= 2 or at_col[0] is heads[0]))):
            continue
        found = []
        for x, _ in style.columns:
            if not ln.x0 - em <= x <= ln.x1:
                continue
            if any(abs(h.x0 - x) <= LABEL_COLUMN_EM * em for h in read):
                continue  # a label read there
            box = _blob_at(page, style, x, base, em)
            if box is None:
                continue
            word = next((w for w in ln.words if w.x0 <= box[0] + 2 and w.x1 >= box[2] - 2), None)
            starts = word is not None and abs(word.x0 - box[0]) <= 0.3 * em
            m = _LABEL_START.match(word.text) if starts else None
            found.append((box, m.group(1).upper() if m else None, word if starts else None))
        if found:
            out[id(ln)] = found
    return out


def _letters_of(row: list[tuple[float, str | None]], style: _Style, em: float) -> list[str | None]:
    """Letters of a row's labels (x, letter read or None) from those read: consecutive from left to right; a
    label alone takes the letter its column always has on the page."""
    known = [(i, t) for i, (_, t) in enumerate(row) if t]
    out = []
    for i, (x, t) in enumerate(row):
        if t is None and known:
            j, a = min(known, key=lambda k: abs(k[0] - i))
            guess = chr(ord(a) + i - j)
            t = guess if "A" <= guess <= "D" else None
        if t is None:
            col = min(style.columns, key=lambda c: abs(c[0] - x))
            total = sum(col[1].values())
            letter, n = max(col[1].items(), key=lambda kv: kv[1])
            if abs(col[0] - x) <= LABEL_COLUMN_EM * em and total >= 2 and n == total:
                t = letter
        out.append(t)
    return out


# --- the options of answer rows ---


def option_regions(
    lines: list, gray: np.ndarray, ink: np.ndarray, rgb: np.ndarray | None, dpi: int, page: _Page | None = None,
) -> list[MathRegion]:
    """Bodies of answer options. In a line of several pieces (tab-separated), or a line that is
    only a label, each piece starting with a label "A."-"D." (or "a."-"d." beside such a label)
    gives a region: from the first word after the label (for a lone label, from just right of
    it) grown on ink, at most to 1em before the next label. Labels Tesseract garbled or lost
    (_blob_labels) count as well; such a region, and one after a lowercase label, carries the
    label to write. A region holding a word read with confidence as Vietnamese text (any line's
    word centred in it) is text, not a formula, and is skipped; so is one whose growth did not
    settle (it ran into the next option or other lines). A region at least TALL_EM high is
    "tall", others "option"."""
    if not lines:
        return []
    page = page or _Page(lines, ink, rgb)
    style = _label_style(page)
    blobs = _blob_labels(page, style)
    out = []
    for ln in lines:
        pieces = _ocr._pieces(ln.words)
        labelled = [k for k, piece in enumerate(pieces) if _LABEL_LOOSE.match(piece[0].text)]
        entries = []  # (x0, box, words after the label, label word, Label to write or None, letter)
        if any(_ocr._LABEL_ONLY.match(pieces[k][0].text) for k in labelled) and not (len(pieces) < 2 and len(pieces[0]) > 1):
            for k in labelled:
                lab = pieces[k][0]
                box = (lab.x0, lab.y0, lab.x1, lab.y1)
                entries.append([lab.x0, box, pieces[k][1:], lab, None, lab.text.lstrip("*")[0].upper()])
        if id(ln) in blobs and (len(pieces) >= 2 or len(ln.words) == 1):
            for box, letter, word in blobs[id(ln)]:
                entries.append([box[0], box, [], word, None, letter])
        if not entries:
            continue
        entries.sort(key=lambda e: e[0])
        em = page.em(ln)
        if style is not None:
            for e, letter in zip(entries, _letters_of([(e[0], e[5]) for e in entries], style, em)):
                e[5] = letter
        for n, (_, lbox, rest, word, _, letter) in enumerate(entries):
            blob = word is None or not _LABEL_LOOSE.match(word.text)
            if blob and letter is None:
                continue  # a label without a letter to write: its text would be lost
            nxt = entries[n + 1][0] if n + 1 < len(entries) else None
            # the box ends more than 1em before the next option, so the two stay apart (_pieces)
            right_limit = int(nxt - em) - 1 if nxt is not None else None
            if rest:
                seed = (rest[0].x0, min(w.y0 for w in rest), max(w.x1 for w in rest), max(w.y1 for w in rest))
            else:
                seed = (lbox[2] + 3, lbox[1], lbox[2] + 4, lbox[3])
            box, settled = grow(
                ink, seed, em, left_limit=lbox[2] + 2, right_limit=right_limit,
                top_limit=ln.baseline - MAX_UP_EM * em, bottom_limit=ln.baseline + MAX_DOWN_EM * em,
            )
            box = tight(ink, box) if settled else None
            if box is None:
                continue
            inside = [w for other in lines for w in other.words if w is not word and _centred(w, box)]
            if any(vietnamese(w.text) and w.conf >= VIETNAMESE_CONF for w in rest + inside):
                continue
            words = list(rest)
            if blob:  # the words of its row over the label and the option
                words = sorted((w for w in ln.words if w.x1 > lbox[0] and w.x0 < box[2] and lbox[1] <= w.cy <= lbox[3]),
                               key=lambda w: w.x0)
                if any(w.x0 < lbox[0] - 0.3 * em or w.x1 > box[2] + 0.3 * em for w in words):
                    continue  # a word runs on into the next option ("C. 273 K. D. 373"): its text would be lost
                words = [w for w in words if w is word or w.x1 > box[0]]
            elif not words:  # a lone label: the words read on its row stand for the option
                words = sorted((w for w in inside if lbox[1] <= w.cy <= lbox[3]), key=lambda w: w.x0)
            texts = [w.text for w in words]
            fix = None
            if blob:
                if word is not None and texts and words[0] is word:  # the label read as one character and a dot
                    texts[0] = _GARBLED_LABEL.sub("", texts[0])
                fix = Label(tuple(int(v) for v in lbox), letter + ".", style.color if style else None)
            elif word.text != word.text.upper():  # "c." beside real labels
                fix = Label((word.x0, word.y0, word.x1, word.y1), word.text.upper(), word.color)
            kind = "tall" if box[3] - box[1] >= TALL_EM * em else "option"
            out.append(MathRegion(box, words, kind, " ".join(t for t in texts if t), em, ln, fix))
    return out


# --- two-dimensional formulas: fraction bars, radicals, bars over letters ---

BAR_THIN_XH = 0.3  # a bar is at most this thick (x-heights; 3 px at least)
BAR_LONG_XH = 0.85  # and at least this long
BAR_REACH_XH = 1.0  # a fraction's numerator and denominator start within this distance of the bar
OVER_REACH_XH = 0.7  # the letters under a bar over them start within this distance
ROOT_HIGH_XH = 1.5  # a radical sign is at least this high


def _near_ink(page: _Page, c: int, y0: int, y1: int) -> tuple[float, int, list[int]]:
    """Other components with ink in rows [y0, y1) over a component's columns: the share of its columns
    their spans cover, the height of the tallest, and the components."""
    x0, _, x1, _ = page.box(c)
    part = page.lab[max(0, y0) : max(0, y1), x0:x1]
    ids = [int(o) for o in np.unique(part) if o != 0 and o != c] if part.size else []
    if not ids:
        return 0.0, 0, []
    covered = np.zeros(x1 - x0, bool)
    for o in ids:
        ox0, _, ox1, _ = page.box(o)
        covered[max(ox0, x0) - x0 : min(ox1, x1) - x0] = True
    return float(covered.mean()), int(page.st[ids, 3].max()), ids


def _radical(page: _Page, c: int) -> bool:
    """A radical sign: a long stroke along its top that reaches its right end, nothing in the top-left corner
    and a tick in its lower-left part."""
    x0, y0, x1, y1 = page.box(c)
    m = page.lab[y0:y1, x0:x1] == c
    h, w = m.shape
    start, length = _longest_run(m[: max(2, h // 8)].any(axis=0))
    if length < 0.45 * w or start + length < w - max(2, w // 10):
        return False
    corner = max(1, w // 5)
    return not m[: int(0.35 * h), :corner].any() and bool(m[int(0.35 * h) :, :corner].any())


def _seeds(page: _Page) -> list[tuple[str, tuple[int, int, int, int], float]]:
    """2-D seeds of a page: (kind, box of the seed and the glyphs it holds, where its line's baseline should
    be). Kinds: "bar" a fraction bar (thin, flat, long, glyphs just above and below it; the baseline lies
    0.27em under it), "over" a bar over letters (letters just under it, nothing over it), "root" a radical
    sign (it ends 0.12em under the baseline)."""
    xh = page.xh
    st = page.st
    w, h, area = st[:, 2], st[:, 3], st[:, 4]
    thin = (h <= max(3, BAR_THIN_XH * xh)) & (w >= BAR_LONG_XH * xh) & (w <= 20 * xh) & (area >= 0.6 * w * h)
    tall = (h >= ROOT_HIGH_XH * xh) & (h <= 6 * xh) & (w >= xh) & (w <= 25 * xh) & (area <= 0.3 * w * h)
    thin[0] = tall[0] = False
    out = []
    for c in np.flatnonzero(thin):
        c = int(c)
        y, ch = int(st[c, 1]), int(st[c, 3])
        above, top_h, ups = _near_ink(page, c, y - int(BAR_REACH_XH * xh), y)
        below, low_h, downs = _near_ink(page, c, y + ch, y + ch + int(BAR_REACH_XH * xh))
        if above >= 0.25 and below >= 0.25 and top_h >= 0.5 * xh and low_h >= 0.5 * xh:
            out.append(("bar", page.union([c, *ups, *downs]), y + ch / 2 + 0.27 * page.body))
            continue
        under, under_h, unders = _near_ink(page, c, y + ch, y + ch + int(OVER_REACH_XH * xh))
        if above == 0.0 and under >= 0.4 and under_h >= 0.8 * xh:
            box = page.union([c, *unders])
            out.append(("over", box, float(box[3])))
    for c in np.flatnonzero(tall):
        if _radical(page, int(c)):
            box = page.box(int(c))
            out.append(("root", box, box[3] - 0.12 * page.body))
    return out


def tall_regions(page: _Page, taken: list[MathRegion]) -> list[MathRegion]:
    """Two-dimensional formulas outside the regions taken: each seed (_seeds) grown on ink (up and down
    across the bar, sideways over gaps up to RIGHT_GAP_EM) where words read as prose and labels are
    blanked, so "v_cqp =" or a unit joins it but "tăng" and "lần" do not. A region whose growth did not
    settle, or that holds a prose word, is dropped. The line it is written into is the one whose
    baseline lies where the seed's sits (a fraction bar 0.27em over it). All are "tall": an unread one
    becomes a picture."""
    seeds = [s for s in _seeds(page) if not _overlap_any(s[1], [r.box for r in taken])]
    if not seeds:
        return []
    stops = [w for ln in page.lines for w in ln.words if _prose(w)]
    free = page.ink.copy()
    for w in stops:
        free[w.y0 : w.y1, w.x0 : w.x1] = 0
    for r in taken:
        free[r.box[1] : r.box[3], r.box[0] : r.box[2]] = 0
    em = page.body
    out = []
    done: list[tuple] = []
    for kind, (sx0, sy0, sx1, sy1), want in seeds:
        if any(b[0] <= (sx0 + sx1) / 2 <= b[2] and b[1] <= (sy0 + sy1) / 2 <= b[3] for b in done):
            continue  # part of a formula grown already
        up, down = {"bar": (0.6, 0.6), "over": (0.3, 0.5), "root": (0.5, 0.5)}[kind]
        box, settled = grow(free, (sx0, sy0, sx1, sy1), em, left_limit=-1, top_limit=sy0 - up * em,
                            bottom_limit=sy1 + down * em, left_gap_em=RIGHT_GAP_EM)
        box = tight(free, box) if settled else None
        if box is None:
            continue
        done.append(box)
        near = [ln for ln in page.lines if box[1] - 0.2 * em <= ln.baseline <= box[3] + 0.5 * em
                and any(w.x1 >= box[0] - 3 * em and w.x0 <= box[2] + 3 * em for w in ln.words)]
        if not near:
            continue
        line = min(near, key=lambda ln: abs(ln.baseline - want))
        inside = [w for ln in page.lines for w in ln.words if _centred(w, box)]
        if any(_prose(w) for w in inside):
            continue
        words = sorted(inside, key=lambda w: w.x0)
        out.append(MathRegion(box, words, "tall", " ".join(w.text for w in words), page.em(line), line))
    return out



# --- formulas in running text: words with superscripts or subscripts ---

# Heights in x-heights (xh) from the line's baseline, measured on Times: caps reach 1.47xh, superscript digits
# 1.83xh and end 0.9xh above the baseline; subscript digits start 0.3xh up and end 0.55xh under it, small
# capitals start 0.75xh up and end 0.2xh under it; a comma is 0.55xh high, an accent 0.35xh.
SUP_TOP = 1.65  # a superscript reaches at least this high (over the caps; a degree sign does not)
SUP_BOTTOM = 0.55  # and ends at least this far over the baseline
SUB_TOP = 0.85  # a subscript starts at most this high (under the x-height letters' tops)
SUB_BOTTOM = 0.12  # and ends at least this far under the baseline
SUP_HIGH = 2.3  # and not higher than this (a glyph of the line above)
SUP_SIZE = (0.55, 1.15)  # height of a superscript
SUB_SIZE = (0.76, 1.25)  # of a subscript (a comma, bold or not, or the foot of a semicolon is lower)
BASE_GAP = 1.4  # the glyph a script belongs to ends at most this far left of it
MARK_GAP = 0.35  # a small glyph this close over (under) another one is its accent (dot below), not a script
LOW_CONF = 50  # a word of a text line read with less confidence holding a number is read by the model too


def _base_glyph(page: _Page, c: int, base: float, xh: float, scripts: list[int]) -> tuple[int, bool] | None:
    """The glyph a script candidate belongs to: the nearest component ending left of its centre within
    BASE_GAP that stands on the baseline at least a letter high, or a script found before it (True)."""
    x, _, cw, _, _ = (int(v) for v in page.st[c])
    best, right = None, -1
    for o in page.near(x - (BASE_GAP + 3) * xh, x + cw / 2):
        if o == c:
            continue
        ox, oy, ow, oh, _ = (int(v) for v in page.st[o])
        if ox >= x or ox + ow > x + cw // 2 + 1 or ox + ow < x - BASE_GAP * xh:
            continue
        on_line = 0.75 * xh <= oh <= 2.4 * xh and abs(oy + oh - base) <= 0.25 * xh
        if (on_line or o in scripts) and ox + ow > right:
            best, right = (int(o), o in scripts), ox + ow
    return best


def _marked(page: _Page, c: int, xh: float, below: bool) -> bool:
    """Other ink within MARK_GAP under (or over) the middle of a glyph: it is an accent of that one."""
    x, y, cw, ch, _ = (int(v) for v in page.st[c])
    a = x + cw // 5
    b = max(a + 1, x + cw - cw // 5)
    gap = max(2, int(MARK_GAP * xh))
    rows = page.lab[y + ch : y + ch + gap, a:b] if below else page.lab[max(0, y - gap) : y, a:b]
    return bool(((rows != 0) & (rows != c)).any())


def _scripts(page: _Page, w, base: float, xh: float) -> tuple[list[int], list[int]]:
    """Components of a word in its line's band, and those that are superscripts or subscripts: small glyphs
    raised over the caps or lowered under the baseline, higher or lower than the glyph they follow, that
    are not an accent of a glyph just under (over) them."""
    comps = []
    for c in page.near(w.x0 - 1, w.x1 + 1):
        _, y, cw, ch, _ = (int(v) for v in page.st[c])
        if y >= base - 2.6 * xh and y + ch <= base + 1.5 * xh and ch <= 2.4 * xh and cw <= 6 * xh:
            comps.append(int(c))
    found: list[int] = []
    for c in sorted(comps, key=lambda c: page.st[c, 0]):
        _, y, _, ch, _ = (int(v) for v in page.st[c])
        top, bottom = y - base, y + ch - base
        sup = -SUP_HIGH * xh <= top <= -SUP_TOP * xh and bottom <= -SUP_BOTTOM * xh and SUP_SIZE[0] * xh <= ch <= SUP_SIZE[1] * xh
        sub = top >= -SUB_TOP * xh and bottom >= SUB_BOTTOM * xh and SUB_SIZE[0] * xh <= ch <= SUB_SIZE[1] * xh
        if not (sup or sub):
            continue
        b = _base_glyph(page, c, base, xh, found)
        if b is None:
            continue
        if not b[1]:
            _, by0, _, by1 = page.box(b[0])
            if sup and not (y <= by0 - 0.15 * xh and y + ch <= by1 - 0.5 * xh):
                continue
            if sub and not (y >= by0 + 0.15 * xh and y + ch >= by1 + 0.1 * xh):
                continue
        if _marked(page, c, xh, below=sup):
            continue
        found.append(c)
    return comps, found


_MATH_SIGN = re.compile(r"[=+±×÷/°^'\"]|\d[,.]\d")


def _low_conf_number(w) -> bool:
    """A word read with low confidence holding a digit and a decimal mark or a math sign ("2,5.10°", "J/(mol.K)"
    with a digit): Tesseract garbles exponents and symbols. Plain integers stay OCR's."""
    return (w.conf < LOW_CONF and any(ch.isdigit() for ch in w.text) and bool(_MATH_SIGN.search(w.text))
            and not vietnamese(w.text))


def inline_regions(page: _Page, taken: list[MathRegion]) -> list[MathRegion]:
    """Words of running text read by the formula model: those with a superscript or subscript (_scripts), with
    a word that is only scripts joined to the word before it, and words read with low confidence that hold a
    digit. A region is the word's ink, tight; words read as prose, labels and words in the regions taken
    are left alone."""
    boxes = [r.box for r in taken]
    out = []
    for ln in page.lines:
        base, xh = page.metrics(ln)
        text_line = sum(_prose(w) for w in ln.words) >= 2  # not the labels of a figure
        words = sorted((w for w in ln.words if w.math is None), key=lambda w: w.x0)
        groups: list[list[tuple]] = []
        prev = None
        for w in words:
            if _LABEL_LOOSE.match(w.text) or (vietnamese(w.text) and w.conf >= VIETNAMESE_CONF):
                prev = None
                continue
            comps, scripts = _scripts(page, w, base, xh)
            if not comps:
                prev = None
                continue
            item = (w, comps)
            if scripts and len(scripts) == len(comps) and prev is not None and w.x0 - prev[0].x1 <= 0.6 * xh:
                if groups and groups[-1][-1] is prev:
                    groups[-1].append(item)
                else:
                    groups.append([prev, item])
            elif scripts or (text_line and _low_conf_number(w)):
                groups.append([item])
            prev = item
        em = page.em(ln)
        for group in groups:
            box = tight(page.ink, page.union([c for _, comps in group for c in comps]))
            if box is None or _overlap_any(box, boxes):
                continue
            ws = [w for w, _ in group]
            out.append(MathRegion(box, ws, "inline", " ".join(w.text for w in ws), em, ln))
    return out


def _case_by_height(page: _Page, region: MathRegion) -> None:
    """Tesseract's case of a region's first letter set from its glyph when the text starts with a letter
    whose two cases differ in size only ("Vh" for v_H, "XA" for x_A): mathocr refuses a reading whose first
    letter's case differs from the OCR text's, or, measured so (case_from_ink), gives it the glyph's case."""
    text = region.ocr_text
    if not text or text[0].upper() not in CASE_BY_HEIGHT or not region.words:
        return
    _, xh = page.metrics(region.line)
    x0 = region.box[0]
    glyphs = [c for c in page.near(x0 - 1, x0 + 1.5 * xh) if page.st[c, 3] >= 0.6 * xh and page.st[c, 0] >= x0
              and region.box[1] <= page.st[c, 1] + page.st[c, 3] / 2 <= region.box[3]]
    if not glyphs:
        return
    h = int(page.st[min(glyphs, key=lambda c: page.st[c, 0]), 3])
    if h <= 1.2 * xh:
        region.ocr_text, region.case_from_ink = text[0].lower() + text[1:], True
    elif h >= 1.35 * xh:
        region.ocr_text, region.case_from_ink = text[0].upper() + text[1:], True


def rules(lines: list, gray: np.ndarray, ink: np.ndarray, rgb: np.ndarray | None, dpi: int) -> list[MathRegion]:
    """The "rules" detector: answer options, then 2-D formulas outside them, then words with scripts outside both."""
    if not lines:
        return []
    page = _Page(lines, ink, rgb)
    options = option_regions(lines, gray, ink, rgb, dpi, page)
    tall = tall_regions(page, options)
    inline = inline_regions(page, options + tall)
    for region in options + tall + inline:
        _case_by_height(page, region)
    return options + tall + inline


DETECTORS: dict[str, Detector] = {"rules": rules}


# --- dispatch ---


def _overlap(a: tuple, b: tuple) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def dedupe(regions: list[MathRegion]) -> list[MathRegion]:
    """Regions in order without overlaps: a box found twice is kept once; boxes that overlap
    otherwise are all dropped (each crop would hold part of the other formula)."""
    unique: list[MathRegion] = []
    for r in regions:
        if not any(tuple(r.box) == tuple(u.box) for u in unique):
            unique.append(r)
    return [r for r in unique if not any(o is not r and _overlap(r.box, o.box) for o in unique)]


def find(
    lines: list, gray: np.ndarray, ink: np.ndarray, rgb: np.ndarray, dpi: int, method: str | None = None,
) -> list[MathRegion]:
    """Formula regions of a page by the detector `method` (default settings.MATH_DETECT)."""
    method = MATH_DETECT if method is None else method
    if method == "off" or not lines:
        return []
    detector = DETECTORS.get(method)
    if detector is None:
        log.warning("unknown formula detector %r: no formula regions", method)
        return []
    regions = [r for r in detector(lines, gray, ink, rgb, dpi) if r.box[2] > r.box[0] and r.box[3] > r.box[1]]
    return dedupe(regions)
