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

find() runs the detector named by settings.MATH_DETECT (P2W_MATH_DETECT): "rules"
(default), "off", or any name registered in DETECTORS, and drops regions that overlap
(two copies of one box: one is kept).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import ocr as _ocr
from .settings import MATH_DETECT

log = logging.getLogger("pdf2word.mathregions")

KINDS = ("option", "tall", "inline")


@dataclass
class MathRegion:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels (x1, y1 exclusive)
    words: list = field(default_factory=list)  # app.ocr.Word objects the reading replaces
    kind: str = "inline"  # "option": an answer option's body; "tall": 2-D (fraction, root); "inline": in text
    ocr_text: str = ""  # Tesseract's text of `words`, compared with the reading's digits
    em_px: float = 0.0  # font size in pixels (the line's)
    line: object = None  # the app.ocr.Line the result is written into


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
) -> tuple[tuple[int, int, int, int], bool]:
    """Grow a box on ink until it is stable: up and down while the row next to it has ink (and
    across the gap between a fraction bar and its numerator or denominator, and up to a thin mark
    just over the box), right
    until RIGHT_GAP_EM of blank columns, left until LEFT_GAP_EM of blank columns (never reaching
    left_limit). Returns the box and whether it settled within the other limits (False: it is
    still on ink at right_limit, top_limit or bottom_limit)."""
    h, w = ink.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1), False
    top = 0 if top_limit is None else max(0, min(int(top_limit), y0))
    bottom = h if bottom_limit is None else min(h, max(int(bottom_limit), y1))
    gap, lgap = max(2, int(RIGHT_GAP_EM * em)), max(2, int(LEFT_GAP_EM * em))
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


def option_regions(lines: list, gray: np.ndarray, ink: np.ndarray, rgb: np.ndarray, dpi: int) -> list[MathRegion]:
    """Bodies of answer options. In a line of several pieces (tab-separated), or a line that is
    only a label, each piece starting with a label "A."-"D." (or "a."-"d." beside such a label)
    gives a region: from the first word after the label (for a lone label, from just right of
    it) grown on ink, at most to 1em before the next label. A region holding a word read with
    confidence as Vietnamese text (any line's word centred in it) is text, not a formula, and is
    skipped; so is one whose growth did not settle (it ran into the next option or other lines).
    A region at least TALL_EM high is "tall", others "option"."""
    if not lines:
        return []
    out = []
    # Tesseract sizes a line merged with a fraction's parts far too big: no more than body text
    body = float(np.median([w.em_px for ln in lines for w in ln.words]))
    for ln in lines:
        pieces = _ocr._pieces(ln.words)
        labelled = [k for k, piece in enumerate(pieces) if _LABEL_LOOSE.match(piece[0].text)]
        if not any(_ocr._LABEL_ONLY.match(pieces[k][0].text) for k in labelled):
            continue
        if len(pieces) < 2 and len(pieces[0]) > 1:
            continue
        em = min(ln.em_px, body)
        for n, k in enumerate(labelled):
            label, rest = pieces[k][0], pieces[k][1:]
            nxt = pieces[labelled[n + 1]][0].x0 if n + 1 < len(labelled) else None
            # the box ends more than 1em before the next option, so the two stay apart (_pieces)
            right_limit = int(nxt - em) - 1 if nxt is not None else None
            if rest:
                seed = (rest[0].x0, min(w.y0 for w in rest), max(w.x1 for w in rest), max(w.y1 for w in rest))
            else:
                seed = (label.x1 + 3, label.y0, label.x1 + 4, label.y1)
            box, settled = grow(
                ink, seed, em, left_limit=label.x1 + 2, right_limit=right_limit,
                top_limit=ln.baseline - MAX_UP_EM * em, bottom_limit=ln.baseline + MAX_DOWN_EM * em,
            )
            box = tight(ink, box) if settled else None
            if box is None:
                continue
            inside = [w for other in lines for w in other.words if w is not label and _centred(w, box)]
            if any(vietnamese(w.text) and w.conf >= VIETNAMESE_CONF for w in rest + inside):
                continue
            words = list(rest)
            if not words:  # a lone label: the words read on its row stand for the option
                words = sorted((w for w in inside if label.y0 <= w.cy <= label.y1), key=lambda w: w.x0)
            kind = "tall" if box[3] - box[1] >= TALL_EM * em else "option"
            out.append(MathRegion(box, words, kind, " ".join(w.text for w in words), em, ln))
    return out


DETECTORS: dict[str, Detector] = {"rules": option_regions}


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
