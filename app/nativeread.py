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

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np
import pymupdf

from . import mathocr
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

DPI = 300  # native.FORMULA_DPI: the pictures' resolution, and what mathocr's crop padding was tuned for
GAP = 24  # px of white between stacked crops
CHUNK = 20  # pictures per read_regions call: cancellation is checked between chunks


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
    failed: bool = False  # the model could not be used (missing, broken): the rest kept as pictures

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


def _crop(png: bytes) -> np.ndarray | None:
    """The picture flattened on white, grey (its alpha is the ink)."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[0] < 6 or img.shape[1] < 4:
        return None
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def _words(page: pymupdf.Page, languages: list[str]) -> list:
    gray = render_gray(page, DPI)
    drop_light_ink(gray)
    lines = tesseract_lines(gray, "+".join(languages) or "vie", DPI)
    fix_lines(lines)
    return [w for ln in lines for w in ln.words]


def _inside(words: list, box: pymupdf.Rect) -> list:
    x0, y0, x1, y1 = (v * DPI / 72 for v in box)
    out = []
    for w in words:
        ix = max(0, min(x1, w.x1) - max(x0, w.x0))
        iy = max(0, min(y1, w.y1) - max(y0, w.y0))
        if ix * iy >= 0.5 * max(1, (w.x1 - w.x0) * (w.y1 - w.y0)):
            out.append(w)
    return sorted(out, key=lambda w: (w.line_no, w.x0))


def _font(runs: list[Run], i: int) -> str | None:
    """The text font around the picture (the nearest run before it, else after it)."""
    for r in [*reversed(runs[:i]), *runs[i + 1 :]]:
        if r.font and not r.math and not r.image and r.text.strip():
            return r.font
    return None


def read_pictures(page: pymupdf.Page, formulas: list[dict], holders: list[Holder], budget: FormulaBudget) -> None:
    """Read the page's formula pictures (formulas: native's formula dicts, page order) in place: a trusted reading
    replaces the picture run in its paragraph or cell; f["read"] says how ("eq" / "text")."""
    where = {id(r): h for h in holders for r in h.runs}
    todo = [
        f
        for f in formulas
        if f["run"] is not None and f["run"].image and not f["run"].math and not f.get("pieces")
        and id(f["run"]) in where and f["run"].image_w > 1 and f["run"].image_h > 1
    ]
    if not todo:
        return
    if budget.failed or budget.left <= 0:
        budget.over += 0 if budget.failed else len(todo)
        return
    budget.over += max(0, len(todo) - budget.left)
    todo = todo[: budget.left]
    budget.left -= len(todo)
    reader = _reader()
    if reader is None:
        budget.failed = True
        return
    crops = [_crop(f["run"].image) for f in todo]
    keep = [(f, c) for f, c in zip(todo, crops, strict=True) if c is not None]
    budget.unsure += len(todo) - len(keep)
    if not keep:
        return
    try:
        words = _words(page, budget.languages)
    except Exception:  # Tesseract failed: no digit check, so nothing is trusted
        budget.unsure += len(keep)
        return
    changed: set[int] = set()
    for k in range(0, len(keep), CHUNK):
        if budget.check is not None:
            budget.check()
        chunk = keep[k : k + CHUNK]
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
        except Exception:
            mathocr.broken()
            budget.failed = True
            budget.unsure += len(keep) - k
            break
        for (f, _), ws, d in zip(chunk, places, decisions, strict=True):
            if (
                d is None
                or (d.kind == "eq" and not _converts(d.latex))
                or mathocr.split_label(d) is not None
                or _unmarked(d, ws)
            ):
                budget.unsure += 1
                continue
            h = where[id(f["run"])]
            if _apply(f, h, d):
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
PLAIN_MAX = 24  # characters of Tesseract's text a plain reading may stand for (labels, units, short words)


def _after(h: Holder, run: Run) -> str:
    """The text right after the run in its paragraph or cell."""
    i = next((j for j, r in enumerate(h.runs) if r is run), len(h.runs))
    return h.runs[i + 1].text if i + 1 < len(h.runs) else ""


def _as_tesseract(reader, canvas: np.ndarray, box: tuple, em: float, text: str, after: str) -> mathocr.MathDecision | None:
    """Tesseract's text of the place as the reading, when a confident reading of the picture is that same plain
    text (no script, no 2-D part), letter for letter but for Vietnamese marks and spaces; else None. A last mark of
    Tesseract's text that the text after the picture starts with is that text's ("km/h?" before "?")."""
    ocr = text.strip()
    if ocr and after[:1] in NEXT_MARKS and after[:1] and ocr.endswith(after[:1]):
        ocr = ocr[:-1].rstrip()
    if not ocr or len(ocr) > PLAIN_MAX:
        return None
    crop = mathocr.cut(canvas, box, em)
    if crop is None:
        return None
    flat = "".join(ocr.split())
    if not any(p in (flat, _base_letters(flat)) for p in mathocr.plain_readings(reader, crop) if p):
        return None
    return mathocr.MathDecision("text", parts=[(ocr, None)], source=flat)


def _without_next_punct(text: str, after: str) -> str:
    """The reading without its final mark when the text right after the picture starts with it: Tesseract's word
    ran past the formula ("km/h." for the picture of "km/h"), so the reading took the mark the line already has."""
    if text and text[-1] in END_PUNCT and after.startswith(text[-1]):
        return text[:-1].rstrip()
    return text


def _apply(f: dict, h: Holder, d: mathocr.MathDecision) -> bool:
    run = f["run"]
    i = next((j for j, r in enumerate(h.runs) if r is run), None)
    if i is None:
        return False
    font = _font(h.runs, i)
    after = h.runs[i + 1].text if i + 1 < len(h.runs) else ""
    if d.kind == "eq":
        run.math = run.math_text = d.latex
        run.font = font  # an inline equation (docx_builder._keep_inline): a lone one is not centred by Word
        tail = _without_next_punct(d.tail, after)
        if tail:
            h.runs.insert(i + 1, Run(tail, size=run.size, font=font, color=run.color))
        f["read"] = "eq"
        return True
    texts = [(text, script) for text, script in d.parts if text]
    if texts and texts[-1][1] is None:
        texts[-1] = (_without_next_punct(texts[-1][0], after), None)
    parts = [Run(text, size=run.size, font=font, color=run.color, script=script) for text, script in texts if text]
    if not parts:
        return False
    h.runs[i : i + 1] = parts
    f["run"] = None  # no picture left for this formula
    f["read"] = "text"
    return True
