"""Superscripts and subscripts of formula regions without the formula model.

Most formulas of exam scans are a number or a unit with an exponent or a subscript ("6,42.10⁻²¹ J.", "mol⁻¹",
"g/m³", "4v_O = 9v_H", "n₁"). Tesseract reads them flat or garbles them ("642.107'J."); the formula model reads
them well but costs about 2 s a region on a small server. This reader needs neither the model nor its time:

1. Geometry (app/mathregions.py) splits a region's glyphs into runs on the line and runs raised over the caps or
   lowered under the baseline (_segments); a short flat stroke at the middle of a raised run's first glyph, just
   before it, is its minus sign. A run of "." "," ";" ":" only is read from its glyphs.
2. The other runs are set side by side on one line, scripts scaled to the text's size and put on the baseline, a
   wide gap between runs (_canvas), and Tesseract reads that line (psm 7, digits, Latin letters and a few signs
   only), twice: as it is and VOTE_SCALE larger. All canvases of a page go to one Tesseract process (a list of
   images), so the process starts once a page.
3. Tesseract's words are given back to the runs, and each character to a glyph of its run (glyphs one above the
   other are one: "=", ";", "i"; touching letters may share one: "m/"). The glyphs check and complete the reading
   (_char): as many characters as glyphs, "," and "." of a number from the glyph (going under the baseline or
   not; Tesseract often drops a small comma), a script's "-" where the geometry found a minus and nowhere else, a
   digit only for a glyph standing on the line at digit height, nothing raised read on the line, "0" and "O" by
   the glyph's width, the case of letters whose cases differ in size only (o, v, x...) by height (in a script by
   SCRIPT_CASE_XH, or against a digit or capital of the same script; y/Y and p/P only against one: the canvas
   shows neither), no "D" or "Q" (a Đ, a 0 or an O), no "t" "f" "j" "L" for a bare stem (a 1), a closed triangle
   read "A" or "4" is a Δ; each character but "," and "." at least MIN_CONF sure; a word after a script is a unit.
4. The two readings must agree. Tesseract misreads a lone letter at one size and not at the other ("Tr" for "T",
   "I." for "J."): one reading still counts when they differ in letters only (same digits and scripts) and the
   page's OCR text has its letters.
   A script letter that can stand for a digit or a Vietnamese letter Tesseract cannot read here (O for 0 or Đ, b
   or d for đ, t for 1, B for 8...: _UNSURE) must be in the page's OCR text too, after the letter it is put on
   ("R_{O}" for R_Đ is refused: OCR read "Rp").
5. The runs are written as LaTeX (scripts in ^{} and _{}) and mathocr.judge() decides as for a model reading
   (Tesseract's digits of the region agree, no unknown word, no subscript after a digit...): a MathDecision of
   kind "text", written as superscript and subscript runs.

A region whose reading fails a check keeps its OCR words (None). Regions without a raised or lowered glyph are
left to OCR (their text is Tesseract's anyway), and two-dimensional ones (fractions, roots, bars: kind "tall")
are not read here: app/ocr.py keeps them as pictures.
"""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import mathocr
from . import mathregions as MR
from .settings import TESSDATA_DIR, TESSERACT_CMD

log = logging.getLogger("pdf2word.scriptocr")

# What Tesseract may read on a canvas (no Vietnamese: a region with Vietnamese text is not a formula). The space
# must be listed: without it Tesseract's LSTM reads a whitelisted line as one word.
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+=,.;:/()' "
# Tesseract model for the canvases (P2W_SCRIPTS_LANG): "eng" (tessdata_best, in tessdata/); a tessdata_fast English
# model copied there under another name (e.g. engfast) reads them in about half the time.
LANG = os.environ.get("P2W_SCRIPTS_LANG", "eng").strip() or "eng"
VOTE_SCALE = 1.5  # each canvas is read again this much larger; the readings must agree (0: one reading)
LETTERS_FROM_OCR = False  # also ask the page's OCR text to hold the reading's letters (_letters_seen)
MIN_CONF = 60  # Tesseract's confidence (0-100) needed for each word of a reading
GAP_XH = 2.5  # blank between two runs on the canvas (x-heights): wider than any space, so a word never spans two
PAD_XH = 0.8  # paper around the canvas
SPACE_XH = 0.4  # a gap this wide between a script and the run after it is a space ("10^{-21} J.")
# A minus starting an exponent: at most this thick (x-heights; 3 px at least), at least this long and at most this
# far left of the exponent's first glyph, its middle between that glyph's top and bottom.
MINUS_THICK_XH = 0.3
MINUS_LONG_XH = (0.25, 1.2)
MINUS_GAP_XH = 0.6
RAISED_XH = 0.3  # a glyph of a run on the line ending this far over the baseline is raised
DIGIT_XH = 1.15  # a digit on the line is at least this high
DOT_XH = 0.7  # "." and "," are at most this high
SPECK_XH2 = 0.02  # ink of noise: at most this area (x-height squared)
LABEL_DOT_XH = 0.4  # a dot at a region's left edge, this small and this far from the rest, is its label's
LABEL_GAP_XH = 0.25
COMMA_HIGH_XH = 0.38  # and is at least this high (a full stop is a smaller dot)
COMMA_XH = 0.15  # a comma ends at least this far under the baseline, a full stop above it
ROUND_LETTER = 0.78  # "O" is at least this wide for its height, "0" at most ROUND_DIGIT
ROUND_DIGIT = 0.7
LOWER_XH, UPPER_XH = 1.2, 1.35  # o/O, v/V...: a glyph at most / at least this high (x-heights)
SUPER = re.compile(r"[-+]?\d{1,3}|[-+]?[A-Za-z]")  # exponents: "-21", "3", "-1", "n"
SUB = re.compile(r"\d{1,2}|[A-Za-z]{1,3}|[A-Za-z]\d")  # indices: "1", "A", "O", "cqp", "A1"
_ROUND = set("0OoDQ")  # round glyphs Tesseract reads as one another; only "0", "O", "o" are kept (by width)
_ONE = set("lIi|")
_STEM = set("tfjL")  # in a script, letters a bare stem is not (a "1" without its flag and foot)
STEM_WIDTH = 0.35  # a glyph narrower than this for its height is a stem
# Script letters that may stand for something the canvas cannot show: a digit (mathocr._DIGIT_LOOK) or "Đ"/"đ"
# (the canvas has no Vietnamese: R_Đ reads R_O, U_Đ reads U_b, F_đh reads F_dh). The page's OCR must hold them.
_UNSURE = set("".join(mathocr._DIGIT_LOOK.values())) - {"|"} | set("Ddbp")
# Script letters whose two cases differ in size only (the canvas scales every script to a capital's height) and
# the script heights (x-heights of the text) that tell them apart: at most the first is small, at least the second
# a capital (scripts at ~50-80 % of the text: capitals measured 0.85-0.96, small letters 0.55-0.75).
_SCRIPT_CASE = mathocr.CASE_BY_HEIGHT
SCRIPT_CASE_XH = (0.7, 0.84)
_DESCENDER_CASE = set("PY")  # p/P, y/Y: alike once put on the baseline; told only by another glyph of the script
_DESCENDS = set("gjpqyJQ")
_RAISED = set("-=+'")  # glyphs of a run on the line that do not reach down to it


@dataclass
class _Seg:
    role: str  # "base", "super" or "sub"
    comps: list[int]  # page components, left to right
    minus: int | None = None  # the minus sign's component (super only)
    span: tuple[int, int] = (0, 0)  # columns on the canvas
    words: list[tuple[str, float]] = field(default_factory=list)  # Tesseract's words (text, conf)
    marks: str | None = None  # a run of punctuation only: its text, from the glyphs (not on the canvas)


class _Span:
    """The columns of a region, as mathregions._scripts looks at a word's."""

    def __init__(self, x0: int, x1: int):
        self.x0, self.x1 = x0, x1


def _segments(page, region) -> list[_Seg] | None:
    """Runs of a region's glyphs on the line / raised / lowered, left to right; None when the region holds ink
    outside the line's band or has no script."""
    base, xh = page.metrics(region.line)
    x0, y0, x1, y1 = region.box
    comps, scripts = MR._scripts(page, _Span(x0, x1), base, xh)
    inside = {int(c) for c in page.near(x0, x1) if y0 <= page.st[c, 1] + page.st[c, 3] / 2 <= y1
              and not _speck(page, c, xh)}
    comps = [int(c) for c in comps if int(c) in inside]
    scripts = [c for c in scripts if int(c) in inside]
    if not scripts or len(comps) != len(inside):
        return None  # nothing raised or lowered, or ink outside the band (a tall glyph, another line)
    found = set(scripts)
    order = sorted(comps, key=lambda c: (page.st[c, 0], page.st[c, 1]))
    role = {}
    for c in order:
        y, h = int(page.st[c, 1]), int(page.st[c, 3])
        role[c] = ("super" if y + h - base <= -MR.SUP_BOTTOM * xh else "sub") if c in found else "base"
    segs: list[_Seg] = []
    for c in order:
        if segs and segs[-1].role == role[c]:
            segs[-1].comps.append(c)
        else:
            segs.append(_Seg(role[c], [c]))
    for k, seg in enumerate(segs):
        if seg.role != "super" or k == 0 or segs[k - 1].role != "base":
            continue
        m, first = segs[k - 1].comps[-1], seg.comps[0]
        mx, my, mw, mh, _ = (int(v) for v in page.st[m])
        fx, fy, _, fh, _ = (int(v) for v in page.st[first])
        if (mh <= max(3, MINUS_THICK_XH * xh) and MINUS_LONG_XH[0] * xh <= mw <= MINUS_LONG_XH[1] * xh and mw >= 1.1 * mh
                and fy <= my + mh / 2 <= fy + fh and 0 <= fx - (mx + mw) <= MINUS_GAP_XH * xh):
            segs[k - 1].comps.pop()
            seg.comps.insert(0, m)
            seg.minus = m
    segs = [s for s in segs if s.comps]
    # a script and the scripts after it must not share columns (x_1^2 stacked: not one line)
    for a, b in zip(segs, segs[1:]):
        if page.st[b.comps[0], 0] < page.st[a.comps[-1], 0] + page.st[a.comps[-1], 2] - 1 and "base" not in (a.role, b.role):
            return None
    first = _units(page, segs[0].comps) if segs[0].role == "base" else []
    if len(first) >= 2 and len(first[0]) == 1:  # the dot of a label "C." the region took in (as mathocr.cut())
        c, nxt = first[0][0], first[1]
        if (page.st[c, 2] <= LABEL_DOT_XH * xh and page.st[c, 3] <= LABEL_DOT_XH * xh
                and min(page.st[o, 0] for o in nxt) - (page.st[c, 0] + page.st[c, 2]) >= LABEL_GAP_XH * xh):
            segs[0].comps.remove(c)
    for k, seg in enumerate(segs):
        units = _units(page, seg.comps)
        if seg.role == "base" and all(page.union(u)[3] - page.union(u)[1] <= DOT_XH * xh for u in units):
            marks = [_mark(page, u, base, xh) for u in units]
            if k == 0 or None in marks:
                return None
            seg.marks = "".join(marks)
    if all(s.role == "base" for s in segs):
        return None
    last = max(k for k, s in enumerate(segs) if s.role != "base")
    for k in range(last + 1, len(segs)):  # "2,408.10^{23} phân tử.": the words after the number stay OCR's
        a, b = page.union(segs[k].comps)[0], page.union(segs[k].comps)[2]
        words = [w for w in region.words if a <= w.cx <= b]
        if any(MR.vietnamese(w.text) and w.conf >= MR.VIETNAMESE_CONF and not any(c.isdigit() for c in w.text)
               for w in words):
            cut = page.union(segs[k].comps)[0]
            if any(w.x0 < cut < w.x1 - 2 for w in region.words):
                return None  # an OCR word runs across the cut
            return segs[:k]
    return segs


def _speck(page, c: int, xh: float) -> bool:
    """Scan noise: a few pixels of ink, far smaller than a full stop."""
    return page.st[c, 4] < SPECK_XH2 * xh * xh and max(page.st[c, 2], page.st[c, 3]) <= 0.2 * xh


def _mark(page, unit: list[int], base: float, xh: float) -> str | None:
    """A punctuation glyph told by its shape: "." a dot on the baseline, "," a mark going under it, ":" and ";" a
    dot over either; None for anything else."""
    parts = sorted(unit, key=lambda c: page.st[c, 1])
    low = page.box(parts[-1])
    w, h = low[2] - low[0], low[3] - low[1]
    if w > DOT_XH * xh:
        return None
    if low[3] >= base + COMMA_XH * xh and low[1] <= base and h >= COMMA_HIGH_XH * xh:
        mark = ","
    elif abs(low[3] - base) <= 0.2 * xh and h <= 0.4 * xh and w <= 2 * h:
        mark = "."
    else:
        return None
    if len(parts) == 1:
        return mark
    if len(parts) == 2:
        x0, y0, x1, y1 = page.box(parts[0])
        if y1 - y0 <= 0.4 * xh and x1 - x0 <= 0.4 * xh and base - 1.2 * xh <= (y0 + y1) / 2 <= base - 0.35 * xh:
            return ":" if mark == "." else ";"
    return None


def _units(page, comps: list[int]) -> list[list[int]]:
    """Glyphs of a run: components one above the other that share most of their columns (the bars of "=", the
    dots of ":", the dot of "i") are one glyph; left to right."""
    out: list[list[int]] = []
    for c in sorted(comps, key=lambda c: page.st[c, 0]):
        x, y, w, h = (int(v) for v in page.st[c, :4])
        if out:
            for o in out[-1]:
                ox, oy, ow, oh = (int(v) for v in page.st[o, :4])
                share = min(x + w, ox + ow) - max(x, ox)
                apart = y >= oy + oh or oy >= y + h
                if apart and share >= 0.6 * min(w, ow):
                    out[-1].append(c)
                    break
            else:
                out.append([c])
            continue
        out.append([c])
    return out


def _canvas(page, gray: np.ndarray, segs: list[_Seg], base: float, xh: float) -> np.ndarray:
    """The runs on one line: each run's own ink (other glyphs whitened) at its place for runs on the line, scripts
    scaled to the text's digit height with their glyphs' bottom on the baseline; GAP_XH of paper between runs."""
    pad, gap = max(4, int(PAD_XH * xh)), max(6, int(GAP_XH * xh))
    above, below = int(2.2 * xh), int(1.2 * xh)  # room over and under the baseline
    height = pad + above + below + pad
    line_y = pad + above
    pieces = []
    segs = [seg for seg in segs if seg.marks is None]
    for seg in segs:
        bx0, by0, bx1, by1 = page.union(seg.comps)
        mask = np.isin(page.lab[by0:by1, bx0:bx1], seg.comps).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        img = np.where(mask > 0, gray[by0:by1, bx0:bx1], 255).astype(np.uint8)
        if seg.role == "base":
            top = int(round(line_y - (base - by0)))
        else:
            glyphs = [c for c in seg.comps if c != seg.minus] or seg.comps
            bottom = max(int(page.st[c, 1] + page.st[c, 3]) for c in glyphs)
            tall = max(int(page.st[c, 3]) for c in glyphs)
            scale = float(np.clip(1.45 * xh / max(tall, 1), 1.0, 2.2))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            top = int(round(line_y - (bottom - by0) * scale))
        pieces.append((img, top))
    width = pad + sum(img.shape[1] for img, _ in pieces) + gap * (len(pieces) - 1) + pad
    lo = min(0, min(top for _, top in pieces))
    hi = max(height, max(top + img.shape[0] for img, top in pieces))
    canvas = np.full((hi - lo, width), 255, np.uint8)
    x = pad
    for seg, (img, top) in zip(segs, pieces):
        h, w = img.shape
        canvas[top - lo : top - lo + h, x : x + w] = np.minimum(canvas[top - lo : top - lo + h, x : x + w], img)
        seg.span = (x, x + w)
        x += w + gap
    return canvas


_PAGE = re.compile(r"class='ocr_page'")
_WORD = re.compile(r"class='ocrx_word'[^>]*title='bbox (\d+) \d+ (\d+) \d+")
_CHAR = re.compile(r"class='ocrx_cinfo' title='x_bboxes [^;]*; x_conf ([\d.]+)'>([^<]*)</span>")


def _tesseract(images: list[np.ndarray], dpi: int) -> list[list[tuple[str, float, int, int]]] | None:
    """Words (text, confidence, left, right) Tesseract reads on each image as one line, in one process. A word's
    confidence is its least sure character's other than "," and "." (the glyphs tell those): Tesseract's word
    confidence drops to 0 for a comma it could not place in an otherwise sure number."""
    if not images:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(tmp, f"{i:04d}.png")
            cv2.imwrite(path, img)
            paths.append(path)
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w", encoding="utf-8") as fh:
            fh.write("\n".join(paths) + "\n")
        cmd = [TESSERACT_CMD, listing, "stdout", "-l", LANG, "--psm", "7", "--dpi", str(dpi),
               "-c", "tessedit_char_whitelist=" + CHARS, "-c", "hocr_char_boxes=1", "hocr"]
        env = {**os.environ, "TESSDATA_PREFIX": str(TESSDATA_DIR), "OMP_THREAD_LIMIT": "1"}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        except (OSError, subprocess.SubprocessError):
            log.exception("tesseract failed on script crops")
            return None
    if proc.returncode != 0:
        log.warning("tesseract failed on script crops: %s", proc.stderr[-300:])
        return None
    out: list[list[tuple[str, float, int, int]]] = []
    word = None
    for line in proc.stdout.splitlines():
        if _PAGE.search(line):
            out.append([])
            continue
        m = _WORD.search(line)
        if m and out:
            word = [out[-1], "", 100.0, int(m.group(1)), int(m.group(2))]
            out[-1].append(word)
            continue
        m = _CHAR.search(line)
        if m and word is not None:
            ch = html.unescape(m.group(2))
            word[1] += ch
            if ch not in ",." and ch.strip():
                word[2] = min(word[2], float(m.group(1)))
    if len(out) != len(images):
        log.warning("tesseract read %d of %d script crops", len(out), len(images))
        return None
    return [[(text.strip(), conf, left, right) for _, text, conf, left, right in words if text.strip()] for words in out]


def _triangle(page, c: int) -> bool:
    """A Δ: a glyph whose hole is a large part of its box and reaches down to its base (an "A" or a "4" has a
    hole ending half way)."""
    x0, y0, x1, y1 = page.box(c)
    paper = np.pad(page.lab[y0:y1, x0:x1] != c, 1, constant_values=True).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    holes = [k for k in range(1, n) if k != lab[0, 0]]
    if not holes:
        return False
    k = max(holes, key=lambda k: st[k, cv2.CC_STAT_AREA])
    return st[k, cv2.CC_STAT_AREA] >= MR.DELTA_HOLE * (x1 - x0) * (y1 - y0) and st[k, 1] - 1 + st[k, 3] >= MR.DELTA_BASE * (y1 - y0)


def _char(page, ch: str, unit: list[int], role: str, base: float, xh: float, minus: int | None) -> str | None:
    """A character Tesseract read for a glyph, checked (and completed) by the glyph; None when they disagree."""
    x0, y0, x1, y1 = page.union(unit)
    w, h = x1 - x0, y1 - y0
    is_minus = minus is not None and minus in unit
    if role != "base":
        if (ch == "-") != is_minus:
            return None
        if ch in "DQ":
            return None  # a Đ, an O or a 0 (Tesseract reads Đ as D or O here)
        if ch in _ROUND:
            return "o" if w >= ROUND_LETTER * h else "0" if w <= ROUND_DIGIT * h else None  # (o/O: _script_case)
        if ch in _ONE:
            return None  # "1", "l" or "I": the glyph does not tell
        if ch in _STEM and w < STEM_WIDTH * h:
            return None  # a "1" that lost its flag and foot
        return ch if ch.isalnum() or ch in "-+" else None
    on_line = abs(y1 - base) <= 0.25 * xh
    if y1 < base - RAISED_XH * xh and ch not in _RAISED:
        return None  # a raised glyph the geometry did not take for a script ("2" of 10^{23} read "°")
    if ch in "DQ":
        return None  # a Đ, an O or a 0
    if ch in _ROUND:
        if not on_line:
            return None
        if h <= LOWER_XH * xh:
            return "o"
        if h >= DIGIT_XH * xh:
            return "O" if w >= ROUND_LETTER * h else "0" if w <= ROUND_DIGIT * h else None
        return None
    if ch.isdigit():
        if not (h >= DIGIT_XH * xh and on_line):
            return None
        if ch == "4" and len(unit) == 1 and _triangle(page, unit[0]):
            return "\\Delta "
        return ch
    if ch in ",.;:":  # (in a run on the line "," and "." are the glyphs' already, see _latex)
        return ch if _mark(page, unit, base, xh) == ch else None
    if ch.isalpha() and ch not in _DESCENDS and y1 >= base + COMMA_XH * xh:
        return None  # a letter that stands on the line read for one that goes under it ("I." for "J.")
    if ch == "A" and len(unit) == 1 and _triangle(page, unit[0]):
        return "\\Delta "
    if ch.isalpha() and ch.upper() in mathocr.CASE_BY_HEIGHT:
        if h <= LOWER_XH * xh:
            ch = ch.lower()
        elif h >= UPPER_XH * xh:
            ch = ch.upper()
    return ch if _shape(ch, unit, w, h, y1 - base, xh) else None


_TALL_REF = set("ABEFGHKMNRT") | set("bdhk")  # script letters as high as a digit, case shown by their shape


def _script_case(page, units: list[list[int]], chars: list[str], xh: float) -> list[str] | None:
    """The case of a script's letters the canvas cannot show (_SCRIPT_CASE, _DESCENDER_CASE): against a digit or a
    capital of the same script (as high, standing on the same line), else by SCRIPT_CASE_XH for the ones that differ
    in size; None when a case is not told."""
    boxes = [page.union(u) for u in units]
    refs = [b for b, c in zip(boxes, chars) if c.isdigit() or c in _TALL_REF]
    out = []
    for c, (x0, y0, x1, y1) in zip(chars, boxes):
        up = c.upper()
        if not c.isalpha() or (up not in _SCRIPT_CASE and up not in _DESCENDER_CASE):
            out.append(c)
            continue
        h, small = y1 - y0, None
        if refs:
            rh = max(b[3] - b[1] for b in refs)
            rb = sorted(b[3] for b in refs)[len(refs) // 2]
            if up in _DESCENDER_CASE:
                small = True if y1 >= rb + 0.15 * rh else False if abs(y1 - rb) <= 0.08 * rh and h >= 0.85 * rh else None
            else:
                small = True if h <= 0.8 * rh else False if h >= 0.9 * rh else None
        elif up in _SCRIPT_CASE:
            small = True if h <= SCRIPT_CASE_XH[0] * xh else False if h >= SCRIPT_CASE_XH[1] * xh else None
        if small is None:
            return None
        out.append(c.lower() if small else up)
    return out


_SMALL = set("acemnorsuvwxz")  # letters of x-height
_TALL = set("bdfhklt") | set("ABCDEFGHIKLMNOPRSTUVWXYZ")


def _shape(ch: str, unit: list[int], w: int, h: int, below: float, xh: float) -> bool:
    """The glyph has the outline of the character on the line: height of an x-height letter, of a tall one or a
    capital, a descender, the long stroke of "/" or a bracket, the bar of "-", the two bars of "="."""
    if ch in _SMALL:
        return h <= LOWER_XH * xh and below <= 0.25 * xh
    if ch in _TALL:
        return h >= (1.05 if ch == "t" else DIGIT_XH) * xh and below <= 0.25 * xh
    if ch in "gpqy":
        return below >= 0.2 * xh
    if ch in "/()":
        return h >= 1.2 * xh and w <= 0.9 * xh
    if ch == "-":
        return h <= max(3, MINUS_THICK_XH * xh) and w >= 1.5 * h
    if ch == "=":
        return len(unit) == 2
    return True


# Ink width of a character (x-heights), to share a glyph of touching characters ("m/" read as one) between them.
_NARROW, _WIDE = set("ijltfr1IJ'.,;:!|/()-"), set("mwMW")


def _width(ch: str) -> float:
    return 0.6 if ch in _NARROW else 1.7 if ch in _WIDE else 1.0


def _align(chars: str, widths: list[float]) -> list[str] | None:
    """The characters of a run shared out among its glyphs, left to right: one each when there are as many,
    otherwise touching characters share a glyph (a glyph as wide as their widths); None when there are fewer
    characters than glyphs or no sharing fits."""
    n, m = len(chars), len(widths)
    if n == m:
        return list(chars)
    if n < m or not m:
        return None
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(m + 1)]
    back = [[0] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for u in range(1, m + 1):
        for e in range(u, n - (m - u) + 1):
            for k in range(1, min(4, e - u + 2)):
                prev = best[u - 1][e - k]
                if prev == inf:
                    continue
                cost = prev + abs(widths[u - 1] - sum(_width(c) for c in chars[e - k : e])) + 0.3 * (k - 1)
                if cost < best[u][e]:
                    best[u][e], back[u][e] = cost, k
    if best[m][n] == inf:
        return None
    out, e = [], n
    for u in range(m, 0, -1):
        k = back[u][e]
        out.append(chars[e - k : e])
        e -= k
    out.reverse()
    if any(len(g) > 1 and (abs(w - sum(_width(c) for c in g)) > 0.8 or w < 0.85 * sum(_width(c) for c in g))
           for g, w in zip(out, widths)):
        return None  # a glyph too narrow for the characters given to it ("Tr" for "T")
    return out


def _space(before: str, after: str) -> bool:
    """A gap between two characters of a run may be a space: not inside a number ("6,42.10", "0,5") nor before
    punctuation."""
    if after in ".,;:)" or before == "(":
        return False
    return not (before[-1:].isdigit() or before[-1:] in ",.") or not after[:1].isdigit()


_SLASHED = re.compile(r"[A-Za-z]+(?:/[A-Za-z]+)+")
_MORE_UNITS = {"s", "h", "g", "l", "lit"}  # with mathregions.UNITS and mathocr.WORDS (read when used: the modules
# import each other)


def _unit(text: str) -> bool:
    """Text after a script that starts with a letter is a unit ("J.", "m/s.", "kPa", "mol.K"): its words of
    letters, split at "/" "." "(" ")", are units, then punctuation."""
    m = re.match(r"([A-Za-zΩ/.()]+?)([.,;:]*)$", text.replace("\\ ", " ").split(" ")[0])
    if not m:
        return False
    words = [w for w in re.split(r"[/.()]", m.group(1)) if w]
    return bool(words) and all(w in MR.UNITS or w in mathocr.WORDS or w in _MORE_UNITS for w in words)


def _latex(page, segs: list[_Seg], base: float, xh: float) -> tuple[str, float] | str:
    """LaTeX of a region's runs as read, checked glyph by glyph (_char), and its lowest word confidence; the
    reason when a check fails. Spaces are where the glyphs leave a word gap (SPACE_XH, _space), whatever
    Tesseract wrote."""
    tex, conf, last, prev = "", 100.0, None, ""
    for seg in segs:
        units = _units(page, seg.comps)
        boxes = [page.union(u) for u in units]
        gap = boxes[0][0] - last if last is not None else 0
        last = boxes[-1][2]
        if seg.marks is not None:
            tex += seg.marks
            prev = seg.marks
            continue
        if not seg.words:
            return f"no word on a {seg.role} run"
        conf = min(conf, *(c for _, c in seg.words))
        read = "".join(t for t, _ in seg.words).replace(" ", "")
        if seg.minus is not None and not read.startswith("-") and len(read) == len(units) - 1:
            read = "-" + read  # Tesseract dropped the exponent's minus the geometry found
        # "," and "." of a run on the line are told by their glyphs (Tesseract drops a small comma or reads it as a
        # full stop: "821.10" for "8,21.10"); Tesseract's characters go to the other glyphs
        marks = [_mark(page, u, base, xh) if seg.role == "base" and b[3] - b[1] <= DOT_XH * xh else None
                 for u, b in zip(units, boxes)]
        marks = [m if m in (",", ".") else None for m in marks]
        if seg.role == "base":
            read = "".join(c for c in read if c not in ",.")
        glyphs = [b for b, m in zip(boxes, marks) if m is None]
        shared = _align(read, [(b[2] - b[0]) / xh for b in glyphs])
        if shared is None:
            return f"{read!r}: {len(read)} characters for {len(glyphs)} glyphs"
        it = iter(shared)
        shared = [m if m is not None else next(it) for m in marks]
        gots = []
        for k, (chars, unit) in enumerate(zip(shared, units)):
            if marks[k] is not None:
                got = chars
            elif len(chars) > 1:  # a "/" touching a letter ("m/"), as Tesseract read them
                if seg.role != "base" or len(chars) != 2 or "/" not in chars or not chars.replace("/", "").isalpha():
                    return f"{chars!r} on one glyph"
                got = chars
            else:
                got = _char(page, chars, unit, seg.role, base, xh, seg.minus)
                if got is None:
                    return f"glyph {chars!r} ({seg.role})"
            gots.append(got)
        if seg.role != "base":
            cased = _script_case(page, units, gots, xh)
            if cased is None:
                return f"case of {''.join(gots)!r} ({seg.role})"
            gots = cased
        out = ""
        for k, (got, box) in enumerate(zip(gots, boxes)):
            if k and seg.role == "base" and box[0] - boxes[k - 1][2] >= SPACE_XH * xh and _space(out, got):
                out += "\\ "
            out += got
        if seg.role == "base":
            if prev == "}" and out[:1].isalpha() and not _unit(out):
                return f"no unit: {out!r}"
            if any(not _unit(t) for t in _SLASHED.findall(out)):
                return f"no unit: {out!r}"  # "g/em" for "g/cm"
            if gap >= SPACE_XH * xh and _space(prev, out):
                tex += "\\ "
            tex += out
        else:
            if not (SUPER if seg.role == "super" else SUB).fullmatch(out):
                return f"script {out!r}"
            tex += ("^{%s}" if seg.role == "super" else "_{%s}") % out
        prev = out if seg.role == "base" else "}"
    return tex, conf


def _reading(page, segs: list[_Seg], words: list, base: float, xh: float, region) -> tuple:
    """One canvas reading: (LaTeX and confidence, or why it cannot be used; what Tesseract read, run by run (role,
    text), or None when its words do not fall on the runs)."""
    segs = [replace(seg, words=[]) for seg in segs]
    runs = [seg for seg in segs if seg.marks is None]
    if len(words) == len(runs):  # one word a run (Tesseract's boxes are loose: the order tells)
        for seg, (text, conf, _, _) in zip(runs, words):
            seg.words.append((text, conf))
    elif any(not _assign(segs, word) for word in words):
        return "a word between runs: " + " ".join(w[0] for w in words), None
    scripts = tuple((seg.role, "".join(t for t, _ in seg.words)) for seg in runs)
    result = _latex(page, segs, base, xh)
    if isinstance(result, str):
        return result, scripts
    tex, conf = result
    if conf < MIN_CONF:
        return f"confidence {conf:.0f}: {tex}", scripts
    if LETTERS_FROM_OCR and not _letters_seen(tex, region.ocr_text):
        return f"letters not in OCR: {tex}", scripts
    return (tex, conf), scripts


def read_regions(regions: list, gray: np.ndarray, ink: np.ndarray, lines: list, dpi: int,
                 reasons: dict | None = None) -> list:
    """MathDecisions (kind "text") for the regions whose scripts read and check, None for the others (and for
    "tall" regions, regions without a script, or when Tesseract fails). Each canvas is read twice, as it is and
    scaled by VOTE_SCALE (Tesseract misreads a glyph at one size and not at the other: "I." for "J."); the two
    readings must agree. reasons: filled with why each region with scripts got None ("ok" when read)."""
    out: list = [None] * len(regions)
    why = reasons if reasons is not None else {}
    page = next((r.page for r in regions if getattr(r, "page", None) is not None), None)
    jobs, images = [], []
    for i, region in enumerate(regions):
        if region.kind == "tall":
            continue
        if page is None:
            page = MR._Page(lines, ink, None)
        segs = _segments(page, region)
        if not segs:
            continue
        base, xh = page.metrics(region.line)
        canvas = _canvas(page, gray, segs, base, xh)
        jobs.append((i, segs, base, xh))
        images.append(canvas)
        if VOTE_SCALE:
            images.append(cv2.resize(canvas, None, fx=VOTE_SCALE, fy=VOTE_SCALE, interpolation=cv2.INTER_CUBIC))
    read = _tesseract(images, dpi)
    if not read:
        return out
    per = 2 if VOTE_SCALE else 1
    for k, (i, segs, base, xh) in enumerate(jobs):
        region = regions[i]
        tries = [_reading(page, segs, read[per * k], base, xh, region)]
        if VOTE_SCALE:
            scaled = [replace(seg, span=(int(seg.span[0] * VOTE_SCALE), int(seg.span[1] * VOTE_SCALE))) for seg in segs]
            tries.append(_reading(page, scaled, read[per * k + 1], base, xh, region))
        readings = [r for r, _ in tries]
        valid = [r for r in readings if not isinstance(r, str)]
        if len(valid) == len(readings) and len({t for t, _ in valid}) == 1:
            first = (valid[0][0], min(c for _, c in valid))
        else:
            # Tesseract misreads a lone letter at one size and not at the other ("Tr" for "T", "I." for "J."): a
            # reading counts when the two differ in letters only and the page's OCR text holds its letters
            alike = _letters_only(tries)
            first = next((r for r in valid if alike and _letters_seen(r[0], region.ocr_text)), None)
            if first is None:
                why[i] = "readings differ: " + " | ".join(r if isinstance(r, str) else r[0] for r in readings)
                continue
        tex, conf = first
        if not _letters_seen(_free_letters(tex), region.ocr_text):
            why[i] = f"letters not in OCR: {tex}"
            continue
        if not _script_letters_seen(tex, region.ocr_text):
            why[i] = f"script letters not in OCR: {tex}"
            continue
        right = max(page.union(seg.comps)[2] for seg in segs)
        kept = [w for w in region.words if w.cx < right]
        ocr_text = " ".join(w.text for w in kept) if len(kept) < len(region.words) else region.ocr_text
        decision, reason = mathocr.judge(tex, conf / 100, ocr_text, min_conf=0.0)
        if decision is not None and decision.kind == "text":
            if len(kept) < len(region.words):  # the words after the runs read keep their OCR text
                region.box = (region.box[0], region.box[1], right, region.box[3])
                region.words, region.ocr_text = kept, ocr_text
            out[i] = decision
            why[i] = "ok"
        else:
            why[i] = f"{reason}: {tex}"
    return out


def _free_letters(tex: str) -> str:
    """The letters of a reading no check of this reader vouches for: not a script's, not the letter a script is
    put on (two readings agree on it), not the unit after a script (_unit). The page's OCR text must hold them
    ("l g/em^{3}" for "1 g/cm^{3}" is refused: OCR read "1 g/cm")."""
    tex = re.sub(r"\}(\\ )?[A-Za-zΩ/.()]+", "}", tex)  # a unit after a script
    tex = re.sub(r"[A-Za-z](?=[_^]\{)", "", tex)  # the letter a script is put on
    return re.sub(r"[_^]\{[^{}]*\}|\\[A-Za-z]+", "", tex)


def _letters_only(tries: list[tuple]) -> bool:
    """Two canvas readings that may differ in letters only: Tesseract's words fell on the runs both times, the
    scripts read the same, every run has the same digits, and where both readings are valid they are the same with
    letters (and "/", read "l" at one size: "g/cm", "glem") left out ("6,47.10" against "6,42.10" is a digit
    misread at one size: no reading)."""
    runs = [r for _, r in tries]
    if any(r is None for r in runs):
        return False
    if len({tuple(t for role, t in r if role != "base") for r in runs}) != 1:
        return False
    if len({tuple(re.sub(r"\D", "", t) for _, t in r) for r in runs}) != 1:
        return False
    valid = [t for t, _ in tries if not isinstance(t, str)]
    return len({re.sub(r"[A-Za-z/]", "?", tex) for tex, _ in valid}) <= 1


def _script_letters_seen(tex: str, ocr_text: str) -> bool:
    """The letters of a script that may stand for a digit or a "Đ"/"đ" (_UNSURE) appear in Tesseract's text of the
    region, after the letter the script is put on (any case, without marks; "đ" is not "d"): T_{t} for T_1 is
    refused when OCR read "T,", R_{O} for R_Đ when it read "Rp"."""
    seen = mathocr._letters(ocr_text).lower()
    pos = 0
    for m in re.finditer(r"([A-Za-z]?)[_^]\{([^{}]*)\}", tex):
        on, script = m.group(1).lower(), [c for c in m.group(2) if c.isalpha()]
        if not any(c in _UNSURE for c in script):
            continue
        if on and (at := seen.find(on, pos)) >= 0:
            pos = at + 1
        for c in script:
            at = seen.find(c.lower(), pos)
            if at < 0:
                if c in _UNSURE:
                    return False
                continue
            pos = at + 1
    return True


def _letters_seen(tex: str, ocr_text: str) -> bool:
    """The letters of the reading outside scripts appear in Tesseract's text of the region, in order (any case,
    without marks): a letter the canvas reading has and the page's OCR did not see is a misreading ("I." for "J.")."""
    plain = re.sub(r"[_^]\{[^{}]*\}|\\[A-Za-z]+", "", tex)
    mine, seen = mathocr._letters(plain).lower(), iter(mathocr._letters(ocr_text).lower())
    return all(c in seen for c in mine)


def _assign(segs: list[_Seg], word: tuple[str, float, int, int]) -> bool:
    """Gives a word Tesseract read on a canvas to the run it starts on (Tesseract's word boxes run on into the gap
    after the word, their left edge is right); False when it starts in a gap."""
    text, conf, left, _ = word
    for seg in segs:
        a, b = seg.span
        if seg.marks is None and a - 3 <= left < b:
            seg.words.append((text, conf))
            return True
    return False
