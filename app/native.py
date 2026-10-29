"""Layout from a PDF's own text layer: for born-digital files the stored text
is kept rather than re-recognised.

- Spans become runs with font, size, bold, italic, underline, colour, highlight
  and super/subscript; VNI/TCVN3 text is converted to Unicode. Underline and
  highlight are decided per character (Word underlines just "C." of "C. 273 K."
  as a thin filled bar, and paints a highlight as a filled box behind the text).
- Items on one baseline (answer options A. B. C. D.) become tab stops.
- Text blocks set side by side become a borderless one-row table.
- Ruled tables (PyMuPDF find_tables) become Word tables.
- Word equations (Cambria Math: math letters, scripts, fraction bars) become
  Word equations again, rebuilt from the glyph positions (app/nativemath.py).
  So do MathType objects that Word for Windows exports as text: each is drawn
  by a form XObject of its own (its EMF picture) in Times New Roman and Symbol,
  which gives the formula's glyphs and strokes whole (_form_frames).
  Formulas it cannot read with certainty, and those whose glyphs carry no
  usable text (MathType pictures exported by Word: subset fonts read as
  "!\"#$"), are cut out as inline pictures at their place in the line, with
  the bars and radicals drawn around them, on a clear ground (no watermark
  from behind).
- Raster images and vector drawings become floating pictures; small labels
  over a figure are rendered into it, large drawings (watermarks) go behind
  the text.
- In OCR mode, large raster images that hold text are OCR'd instead.
"""

import bisect
import math
import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import pymupdf

from . import nativemath
from .layout import alignment, cell_align, looks_like_link
from .model import LINE_BREAK, TAB, Cell, Columns, PageLayout, Paragraph, Picture, Run, Table
from .ocr import ocr_region
from .vnfont import decode_span

FLAG_SUPERSCRIPT, FLAG_ITALIC, FLAG_BOLD = 1, 2, 16
FIGURE_DPI = 220
BACKGROUND_DPI = 150
HEADER_ZONE = 0.08  # top/bottom share of the page searched for headers and footers
DESCENT = 0.22  # depth below the baseline of a line box, in em
OCR_REGION_MIN_AREA = 0.15  # share of the page a raster image needs before OCR is tried
SOFT_HYPHEN = "\xad"
WATERMARK_MIN_AREA = 0.02  # share of the page from which a background is taken for a watermark
FORMULA_DPI = 300
OBJ = "\ufffc"  # text of a run that is a formula picture
# Fonts whose glyphs are formula parts; Symbol joins a formula but alone stays text (Δ, ±, °).
MATH_FONT = re.compile(r"cambria\s*math|mt\s*extra|euclid|mathtype|stix|asana|latin\s*modern\s*math", re.I)
SYMBOL_FONT = re.compile(r"symbol", re.I)
# Math fonts whose glyphs carry real Unicode (Word's equations): their formulas can be rebuilt as
# equations. MathType's MT Extra / Euclid use their own encodings.
UNICODE_MATH_FONT = re.compile(r"cambria\s*math|stix|asana|latin\s*modern\s*math", re.I)
# a number or a figure's label ("(1)", "20%", "(3).") reads as text even in those characters
NUMBER_LABEL = re.compile(r"^[(\[]?[+-]?\d+([.,]\d+)*%?[)\]]?[.,;:?]*$")


def _opaque(text: str) -> bool:
    """Text of a span whose font has no real Unicode map: Word's PDF export of a
    MathType/Equation picture numbers the glyphs from "!" up ("!\"!!#!$")."""
    t = [c for c in text if not c.isspace()]
    if not t or not all(0x21 <= ord(c) <= 0x40 for c in t) or NUMBER_LABEL.match("".join(t)):
        return False
    if all(NUMBER_LABEL.match(word) for word in text.split()):  # "(1) 6273.", "(4) -183."
        return False
    return any(c in "!#$%&" for c in t) or len(set(t) & set("\"'()*+")) >= 2


def _maybe_opaque(text: str) -> bool:
    """Span text that could be such numbering ("\"" alone, a lone digit): joins a
    formula it touches, never starts one."""
    t = [c for c in text if not c.isspace()]
    return bool(t) and all(0x21 <= ord(c) <= 0x40 for c in t)


def _math_alnum(c: str) -> bool:
    return bool(c) and 0x1D400 <= ord(c[0]) <= 0x1D7FF


# Adobe's Symbol encoding, where MathType sets Greek letters and operators: code -> character. Bracket and
# integral pieces and the radical's and arrows' extenders are left out (a formula built with them stays a
# picture), as are signs no formula holds.
SYMBOL_ENCODING = {
    **{c: chr(c) for c in b"!#%&()+,./0123456789:;<=>?[]_{|}"},
    0x22: "∀", 0x24: "∃", 0x27: "∋", 0x2A: "∗", 0x2D: "−", 0x40: "≅", 0x5C: "∴", 0x5E: "⊥", 0x7E: "∼",
    **dict(zip(range(0x41, 0x5B), "ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ")),
    # Symbol draws its "phi" (0x66) stroked, "phi1" (0x6A) looped: ϕ and φ as Unicode shows them
    **dict(zip(range(0x61, 0x7B), "αβχδεϕγηιφκλμνοπθρστυϖωξψζ")),
    0xA1: "ϒ", 0xA2: "′", 0xA3: "≤", 0xA5: "∞", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓", 0xB0: "°",
    0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB8: "÷", 0xB9: "≠", 0xBA: "≡", 0xBB: "≈",
    0xBC: "…", 0xC6: "∅", 0xC7: "∩", 0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈",
    0xCF: "∉", 0xD0: "∠", 0xD1: "∇", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅", 0xD8: "¬", 0xD9: "∧", 0xDA: "∨", 0xDB: "⇔",
    0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓", 0xE5: "∑", 0xF2: "∫",
}
SYMBOL_CHARS = set(SYMBOL_ENCODING.values())
CONTENT_MAX = 4_000_000  # bytes of a page's content streams read for form frames; a larger one is not read
CONTENT_BUDGET = 16_000_000  # bytes of content streams read for form frames per file
CONTENT_STREAMS_MAX = 64  # streams in a page's /Contents read for form frames
DO_MAX = 20000  # "Do" operators of a page looked at for form frames
FRAMES_MAX = 300  # Word EMF forms on a page; a page with more is read as before (VL1108: up to 11)
OPERANDS_MAX = 16  # operands kept before an operator (cm takes 6)
FORM_CONTENT_MAX = 64 * 1024  # bytes of one EMF form's content (its own and the forms it draws); MathType's: < 2 KB
FORMS_WORK_MAX = 1_000_000  # bytes of form content the page copy draws, all frames together
COPY_OBJECTS_MAX = 1000  # objects a page copy (_form_contents) may take with it; Word's pages: 80–140 (VL1108)
COPY_OBJECTS_BUDGET = 20_000  # objects copied for form pages per file
FORM_IMAGES_MAX = 16  # images the forms draw (any image makes its form a picture, not a formula)
OBJ_REF = re.compile(r"(?<![\w/])(\d+)\s+0\s+R\b")
META_REF = re.compile(r"/(Meta\d+)\s+(\d+)\s+0\s+R")


def _math_italic(c: str, bold: bool = False) -> str | None:
    """The math italic letter of an ASCII or Greek small letter (f -> 𝑓, λ -> 𝜆), None if there is none."""
    if c == "h" and not bold:
        return "ℎ"
    name = unicodedata.name(c, "")
    style = "BOLD ITALIC" if bold else "ITALIC"
    for prefix in ("LATIN ", "GREEK "):
        if name.startswith(prefix):
            name = name[len(prefix):].replace("LETTER ", "")
            try:
                return unicodedata.lookup(f"MATHEMATICAL {style} {name}")
            except KeyError:
                return None
    return None


def _mathtype_char(c: str, span: dict, mapped: bool | None = None) -> str | None:
    """What a glyph of a MathType object reads as for app/nativemath.py, or None if it cannot be read.
    Symbol glyphs come as private-use or Latin codes (λ is "\\uf06c" or "l"): Symbol's encoding gives the
    character, a small Greek letter is a math italic (MathType's λ is a variable). A code from 0x80 to
    0xFF is the raw code of a Symbol font without a ToUnicode map (mapped False: "×" is Symbol's 0xD7,
    the dot ⋅) and real Unicode in one with a map (mapped True: "×" is ×); unknown (None): not read.
    Letters of the text font are math italic where it is italic (MathType's variables), upright otherwise
    (units, functions)."""
    font = span["font"]
    if SYMBOL_FONT.search(font):
        o = ord(c)
        if 0xF020 <= o <= 0xF0FF or o < 0x80 or (o <= 0xFF and mapped is False):
            u = SYMBOL_ENCODING.get(o & 0xFF)  # a raw code (Symbol has no Latin letters: "l" is λ)
        elif o > 0xFF or mapped:  # real Unicode from the font's map ("λ", "→"; "×", "µ" for mu)
            u = {"µ": "μ", "·": "⋅"}.get(c, c)
            u = u if u in SYMBOL_CHARS else None
        else:
            u = None
        if u is None:
            return None
        if "α" <= u <= "ω" or u in "ϑϕϖ":
            return _math_italic(u)
        return "∆" if u == "Δ" else u  # nativemath sets ∆ upright, as MathType does
    if MATH_FONT.search(font) or c == "\ufffd" or 0xE000 <= ord(c) <= 0xF8FF or not c.isprintable():
        return None  # MT Extra and other symbol fonts, unmapped glyphs
    if c.isascii() and c.isalpha():
        if _italic(span):
            return _math_italic(c, _bold(span))
        return unicodedata.lookup(f"MATHEMATICAL BOLD {'CAPITAL' if c.isupper() else 'SMALL'} {c.upper()}") if _bold(span) else c
    return c


_CONTENT_TOKEN = re.compile(rb"\s+|%[^\r\n]*|<<|>>|<[0-9A-Fa-f\s]*>|[\[\]{}]|/[^\s/\[\]()<>{}%]*|[^\s/\[\]()<>{}%]+")


def _content_ops(data: bytes):
    """(operator, operands) of a content stream; operands are numbers and names (strings, arrays and
    dictionaries are passed over). Stops at an inline image (its data is not tokens)."""
    operands: list = []
    i, n = 0, len(data)
    while i < n:
        if len(operands) > OPERANDS_MAX:
            del operands[0]
        if data[i] == 0x28:  # a literal string: balanced parentheses, backslash escapes
            depth, i = 1, i + 1
            while i < n and depth:
                ch = data[i]
                i += 2 if ch == 0x5C else 1
                depth += (ch == 0x28) - (ch == 0x29)
            operands.append(None)
            continue
        m = _CONTENT_TOKEN.match(data, i)
        if not m:
            i += 1
            continue
        tok, i = m.group(), m.end()
        first = tok[:1]
        if first.isspace() or first == b"%":
            continue
        if first == b"/":
            operands.append(tok[1:].decode("latin-1"))
        elif first in b"<>[]{}":
            operands.append(None)
        else:
            try:
                operands.append(float(tok))
            except ValueError:
                op = tok.decode("latin-1")
                if op == "BI":
                    return
                yield op, operands
                operands = []


def _matrix_mul(a: tuple, b: tuple) -> tuple:
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3], a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5])


WORD_EMF_FORM = re.compile(r"Meta\d+")  # the forms Word for Windows draws its EMF pictures (MathType objects) with


def _form_frames(page: pymupdf.Page) -> list[dict]:
    """The forms Word for Windows draws its EMF pictures with ("/Meta18 Do": a MathType or Equation
    object is one), as the page's content stream draws them: {"name", "ctm" (the transformation at the
    Do), "rect" (the form's /BBox through its /Matrix and that, in page coordinates)}. Only upright,
    unflipped placements; only the page's own stream (not forms drawn by forms), up to CONTENT_MAX bytes;
    none for a rotated page."""
    try:
        return _read_form_frames(page)
    except Exception:  # noqa: BLE001 – a damaged page keeps its text as it is
        return []


def _page_content(page: pymupdf.Page) -> bytes | None:
    """The page's content streams, read within CONTENT_MAX (and CONTENT_BUDGET per file): Flate streams are
    inflated with a bound, so a compression bomb costs nothing; other filters are not read (None)."""
    doc = page.parent
    xrefs = page.get_contents()
    if not xrefs or len(xrefs) > CONTENT_STREAMS_MAX or len(set(xrefs)) < len(xrefs):
        return None
    parts, total = [], 0
    for x in xrefs:
        kind, filt = doc.xref_get_key(x, "Filter")
        length = _stream_length(doc, x)
        if length is None or length > CONTENT_MAX:
            return None
        raw = doc.xref_stream_raw(x) or b""
        total += len(raw) // 4  # the raw bytes read count too (a short Flate stream padded inside its /Length)
        if kind == "null":
            data = raw[: CONTENT_MAX - total + 1]
        elif filt.strip("[] ") == "/FlateDecode" and doc.xref_get_key(x, "DecodeParms")[0] == "null":
            data = zlib.decompressobj().decompress(raw, CONTENT_MAX - total + 1)
        else:
            return None
        total += len(data)
        if total > CONTENT_MAX:
            return None
        parts.append(data)
    spent = doc.__dict__.get("_p2w_content_bytes", 0) + total
    doc.__dict__["_p2w_content_bytes"] = spent
    return b"\n".join(parts) if spent <= CONTENT_BUDGET else None


def _read_form_frames(page: pymupdf.Page) -> list[dict]:
    if page.rotation:
        return []
    doc = page.parent
    text = _resources(doc, page.xref, "XObject")  # the page's own dictionary: get_xobjects() walks every form tree
    forms = {}
    for name, ref in META_REF.findall(text or "")[: 2 * FRAMES_MAX]:
        if doc.xref_get_key(int(ref), "Subtype")[1] == "/Form":
            forms[name] = int(ref)
    if not forms:
        return []
    data = _page_content(page)
    if data is None:
        return []
    geometry: dict[int, tuple | None] = {}  # xref -> (bbox, matrix), read once
    ctm, stack, out, dos = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), [], [], 0
    marked: list[bool] = []  # marked content open at this point: True for optional content (/OC: may be hidden)
    for op, args in _content_ops(data):
        if op == "BDC":
            marked.append(bool(args) and args[0] == "OC")
        elif op == "BMC":
            marked.append(False)
        elif op == "EMC":
            if marked:
                marked.pop()
        elif op == "q":
            stack.append(ctm)
        elif op == "Q":
            ctm = stack.pop() if stack else ctm
        elif op == "cm" and len(args) >= 6 and all(isinstance(v, float) for v in args[-6:]):
            ctm = _matrix_mul(tuple(args[-6:]), ctm)
        elif op == "Do":
            dos += 1
            if dos > DO_MAX or len(out) > FRAMES_MAX:
                return []  # a page that draws that many forms is not a page of MathType objects
            if not args or args[-1] not in forms or any(marked):
                continue  # (a form in optional content may be hidden: the page shows nothing of it)
            xref = forms[args[-1]]
            if xref not in geometry:
                geometry[xref] = None
                bbox_text = doc.xref_get_key(xref, "BBox")[1]
                kind, value = doc.xref_get_key(xref, "Matrix")
                if len(bbox_text) <= 200 and len(value) <= 200:
                    try:
                        bbox = [float(v) for v in bbox_text.strip("[] ").split()]
                        matrix = tuple(float(v) for v in value.strip("[] ").split()) if kind == "array" else (1, 0, 0, 1, 0, 0)
                        geometry[xref] = (bbox, matrix) if len(bbox) == 4 and len(matrix) == 6 else None
                    except ValueError:
                        pass
            if geometry[xref] is None:
                continue
            bbox, matrix = geometry[xref]
            if not all(map(math.isfinite, (*bbox, *matrix, *ctm))):
                continue
            full = _matrix_mul(matrix, ctm)
            # upright and unflipped, as Word places an inline object (a mirrored text reads backwards)
            if not (full[0] > 0 and full[3] > 0 and abs(full[1]) <= 0.01 * full[0] and abs(full[2]) <= 0.01 * full[3]):
                continue
            pts = [(x * full[0] + y * full[2] + full[4], x * full[1] + y * full[3] + full[5])
                   for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])]
            r = pymupdf.Rect(min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts),
                             max(y for _, y in pts)) * page.transformation_matrix
            if r.is_valid and not r.is_empty and all(map(math.isfinite, r)):
                out.append({"name": args[-1], "ctm": ctm, "rect": r, "xref": xref})
    return out


def _form_symbol_maps(doc: pymupdf.Document, forms: set[int]) -> dict[str, bool]:
    """Whether each Symbol font the forms use has a ToUnicode map, by _font_key (Word's SymbolMT has one;
    a font without one gives its raw codes, read as Latin-1 above 0x7F). Read within the font budget."""
    out: dict[str, bool] = {}
    try:
        todo = sorted(forms)
        kind_of = {x: 0 for x in todo}  # the forms and the forms they draw, one level down
        for x in list(todo):
            kind, xobjects = _read_key(doc, x, "Resources/XObject")
            if kind == "dict":
                for ref in FONT_REF.findall(xobjects):
                    if int(ref) not in kind_of and _read_key(doc, int(ref), "Subtype")[1] == "/Form":
                        kind_of[int(ref)] = 1
                        todo.append(int(ref))
        for x in todo:
            kind, fonts = _read_key(doc, x, "Resources/Font")
            if kind != "dict":
                continue
            for ref in FONT_REF.findall(fonts):
                fx = int(ref)
                name = _read_key(doc, fx, "BaseFont")[1].lstrip("/")
                if SYMBOL_FONT.search(name):
                    key = _font_key(name)
                    out[key] = out.get(key, False) or _read_key(doc, fx, "ToUnicode")[0] != "null"
    except Exception:  # noqa: BLE001 – past the budget or damaged: the fonts not read yet stay unknown
        return out
    return out


def _stream_length(doc: pymupdf.Document, xref: int) -> int | None:
    """A stream's /Length, given in place or as a reference to a number."""
    kind, val = doc.xref_get_key(xref, "Length")
    if kind == "xref":
        val, kind = doc.xref_object(int(val.split()[0]), compressed=True).strip(), "int"
    return int(val) if kind == "int" and val.isdigit() else None


def _stream_size(doc: pymupdf.Document, xref: int, cap: int) -> int | None:
    """Decoded size of a content stream, inflated within cap; None past it or for a filter other than Flate."""
    kind, filt = doc.xref_get_key(xref, "Filter")
    length = _stream_length(doc, xref)
    if length is None or length > cap:
        return None
    raw = doc.xref_stream_raw(xref) or b""
    if kind == "null":
        return len(raw) if len(raw) <= cap else None
    if filt.strip("[] ") != "/FlateDecode" or doc.xref_get_key(xref, "DecodeParms")[0] != "null":
        return None
    try:
        size = len(zlib.decompressobj().decompress(raw, cap + 1))
    except zlib.error:
        return None
    return size if size <= cap else None


def _forms_small(doc: pymupdf.Document, frames: list[dict]) -> bool:
    """The forms the page copy would draw hold little: each at most FORM_CONTENT_MAX of content with the
    forms it draws (one level down), all frames together FORMS_WORK_MAX (a form may be drawn many times)."""
    sizes: dict[int, int | None] = {}
    total = 0
    for fr in frames:
        x = fr["xref"]
        if x not in sizes:
            size = _stream_size(doc, x, FORM_CONTENT_MAX)
            text = _resources(doc, x, "XObject") if size is not None else None
            if text is None:
                size = None  # (resources too large to read, or none found)
            for ref in OBJ_REF.findall(text or "")[:32]:
                if size is None:
                    break
                if doc.xref_get_key(int(ref), "Subtype")[1] == "/Form":
                    sub = _stream_size(doc, int(ref), FORM_CONTENT_MAX)
                    size = None if sub is None or size + sub > FORM_CONTENT_MAX else size + sub
            sizes[x] = size
        if sizes[x] is None:
            return False
        total += sizes[x]
        if total > FORMS_WORK_MAX:
            return False
    return True


def _copy_small(page: pymupdf.Page) -> bool:
    """What insert_pdf would copy with the page – everything its resources and contents reach – is a page's
    worth (COPY_OBJECTS_MAX objects, COPY_OBJECTS_BUDGET per file); counted without copying."""
    doc = page.parent
    kind, val = doc.xref_get_key(page.xref, "Resources")
    if kind == "xref":
        todo = [int(val.split()[0])]
    elif kind == "dict":
        todo = [int(r) for r in OBJ_REF.findall(val)]
    else:
        return False
    todo += page.get_contents()
    spent = doc.__dict__.get("_p2w_copy_objects", 0)
    limit = min(COPY_OBJECTS_MAX, COPY_OBJECTS_BUDGET - spent)
    children = doc.__dict__.setdefault("_p2w_obj_refs", {})  # xref -> the objects it names (None: too large), per file
    seen: set[int] = set()
    while todo:
        x = todo.pop()
        if x in seen or x <= 0:
            continue
        seen.add(x)
        if len(seen) > limit:
            doc.__dict__["_p2w_copy_objects"] = spent + len(seen)
            return False
        if x not in children:
            text = doc.xref_object(x, compressed=True)
            children[x] = None if len(text) > 200_000 else [int(r) for r in OBJ_REF.findall(text)]
        if children[x] is None:
            return False  # an object that large is not one of a Word page's
        todo += children[x]
    doc.__dict__["_p2w_copy_objects"] = spent + len(seen)
    return True


def _form_contents(page: pymupdf.Page, frames: list[dict]) -> tuple[pymupdf.Document, int, list] | None:
    """What each form draws, read from a copy of the page that draws only the forms: per frame, the
    origins and characters of its glyphs and the boxes of its strokes. Before each form the copy paints a
    tiny mark whose colour is the form's number, so the trace order tells whose glyph is whose (a
    position cannot: a fraction's frame reaches over the lines above and below). Returns the copy (it
    draws a form's picture later, _form_picture), the xref of its content stream and the lists; None if
    the page cannot be copied."""
    try:
        if not _forms_small(page.parent, frames) or not _copy_small(page):
            return None
        doc = pymupdf.open()
        doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number, annots=False, links=False)
        copy = doc[0]
        xrefs = copy.get_contents()
        if not xrefs:
            return None
        parts = []
        for k, fr in enumerate(frames):
            x0, y0 = fr["rect"].tl * ~page.transformation_matrix  # the mark: on the page, at the frame
            parts.append(f"q {(k & 255) / 255:.6f} {(k >> 8 & 255) / 255:.6f} 1 rg {x0:.3f} {y0:.3f} 0.01 0.01 re f Q\n")
            parts.append("q {} cm /{} Do Q\n".format(" ".join(f"{v:.6f}" for v in fr["ctm"]), fr["name"]))
        doc.update_stream(xrefs[0], "".join(parts).encode())
        for x in xrefs[1:]:
            doc.update_stream(x, b"")
        copy = doc[0]
        marks = []  # (seqno, frame number)
        strokes = []  # (seqno, rect)
        for d in copy.get_drawings():
            fill, r = d.get("fill"), pymupdf.Rect(d["rect"])
            if d.get("type") == "f" and fill and abs(fill[2] - 1) < 1e-3 and r.width < 0.02 and r.height < 0.02:
                marks.append((d["seqno"], round(fill[0] * 255) + 256 * round(fill[1] * 255)))
            else:
                strokes.append((d["seqno"], r))
        marks.sort()
        seqnos = [m[0] for m in marks]

        def owner(seqno: int) -> int | None:
            i = bisect.bisect_right(seqnos, seqno) - 1
            return marks[i][1] if i >= 0 and marks[i][1] < len(frames) else None

        out = [{"glyphs": set(), "spaces": set(), "strokes": [], "images": 0} for _ in frames]
        for span in copy.get_texttrace():
            k = owner(span["seqno"])
            if k is None:
                continue
            for code, _gid, origin, _bbox in span["chars"]:
                key = (round(origin[0], 2), round(origin[1], 2))
                # (spaces apart: the page's lines may drop some, _join_interleaved)
                out[k]["glyphs" if code > 32 and chr(code).strip() else "spaces"].add(key)
        for seqno, r in strokes:
            k = owner(seqno)
            if k is not None:
                out[k]["strokes"].append(r)
        images = copy.get_image_info()
        if len(images) > FORM_IMAGES_MAX:
            return None  # forms full of images are pictures, not formulas
        for info in images:
            r = pymupdf.Rect(info["bbox"])
            for k, fr in enumerate(frames):
                if not (fr["rect"] & r).is_empty:
                    out[k]["images"] += 1
        return doc, xrefs[0], out
    except Exception:  # noqa: BLE001 – a page that cannot be copied keeps its formulas as they were
        return None


CMAP_BEGIN = re.compile(r"begin(bfchar|bfrange)")
CMAP_TOKEN = re.compile(r"<[0-9A-Fa-f\s]*>|\[|\]")  # a hex string cannot run past a "<": linear
CMAP_MAX = 64 * 1024  # bytes: a math font's map is a few KB; a larger one is not read (its formulas stay pictures)
CMAP_BUDGET = 1024 * 1024  # bytes of maps read per file
FONT_LIST_MAX = 16000  # characters of a page's font dictionary (~800 fonts; Word's pages list ~20)
FONT_READ_BUDGET = 1_000_000  # characters of font objects read per file (Word's: a few hundred per font)
FORMS_MAX = 2000  # form XObjects of a page looked through for math fonts (Word writes each legacy equation as one)
TRACE_TEXTS_MAX = 20000  # text objects on a page before its glyph trace (for drawn marks) is not built
FORMULA_GLYPHS_MAX = 300  # glyphs and strokes of one formula read as an equation (Word's largest here: 73)
FORMULA_MARKS_MAX = 32  # combining marks of one formula
PAGE_INK_MAX = 1.2e6  # pt² of page rendered once to measure brackets (A4 is 0.5e6); larger: pictures
MATH_FONTS_MAX = 16  # simple Unicode math fonts on a page that are checked; more: pictures
STANDARD_ENCODINGS = {"/WinAnsiEncoding", "/MacRomanEncoding", "/StandardEncoding"}
NO_CID_TEXT = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE


def _font_key(name: str) -> str:
    """A font name as both the font list and the text spans give it ("BCDOEE+CambriaMath",
    "Cambria Math Regular", "STIXTwoMath-Regular" / "STIX Two Math Regular")."""
    name = re.sub(r"#([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), name.split("+", 1)[-1])
    name = re.sub(r"-Identity-[HV]$", "", name)  # a CFF CID font's Type0 name carries its encoding
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return re.sub(r"(regular|roman|book|psmt|mt|ps)+$", "", key) or key


def _simple_font_unmapped(doc: pymupdf.Document, xref: int) -> frozenset[str] | None:
    """Text a one-byte math font may give for codes its ToUnicode map leaves out: the raw code, or
    the code read through a standard encoding (an exporter can write an empty map: R/T of
    "pV = R/T" then reads "!\"", and ")" may stand for any glyph). None: a map too large, packed
    otherwise than by Flate, or not read as written (refused once per file)."""
    per_font = doc.__dict__.setdefault("_p2w_simple_fonts", {})  # each font object once per file
    if xref not in per_font:
        per_font[xref] = _simple_font_map(doc, xref)
    return per_font[xref]


def _simple_font_map(doc: pymupdf.Document, xref: int) -> frozenset[str] | None:
    tu_kind, tu = _read_key(doc, xref, "ToUnicode")
    enc_kind, enc = _read_key(doc, xref, "Encoding")
    first, last = (_read_key(doc, xref, k) for k in ("FirstChar", "LastChar"))
    try:  # a simple font's codes are one byte, whatever the file claims
        lo, hi = max(int(first[1]), 0), min(int(last[1]), 255)
    except ValueError:
        lo, hi = 0, 255
    if lo > hi:
        lo, hi = 0, 255
    cache = doc.__dict__.setdefault("_p2w_maps", {})  # per file, by map: fonts often share one
    key = (tu if tu_kind == "xref" else None, lo, hi, enc_kind, enc if enc_kind == "name" else "")
    if key in cache:
        return cache[key]
    if enc_kind == "name" and enc in STANDARD_ENCODINGS:
        cache[key] = frozenset()  # a standard encoding names every glyph the map leaves out
        return cache[key]
    cache[key] = None  # a map that cannot be read is refused once per file
    mapped: set[int] = set()
    if tu_kind == "xref":
        ref = int(tu.split()[0])
        raw = doc.xref_stream_raw(ref)
        if raw is None or len(raw) > CMAP_MAX:
            return None
        filt = doc.xref_get_key(ref, "Filter")[1].strip("[] ")
        if filt in ("null", ""):
            data = raw
        elif filt == "/FlateDecode" and doc.xref_get_key(ref, "DecodeParms")[0] == "null":
            try:  # inflated within bounds: a few bytes can stand for megabytes
                data = zlib.decompressobj().decompress(raw, 4 * CMAP_MAX + 1)
            except zlib.error:
                return None
        else:
            return None
        if len(data) > 4 * CMAP_MAX:
            return None
        budget = doc.__dict__.get("_p2w_map_bytes", 0) + len(data)
        doc.__dict__["_p2w_map_bytes"] = budget
        if budget > CMAP_BUDGET:
            return None  # maps of a few KB each in Word's files: more is not read (formulas stay pictures)
        cmap, pos = data.decode("latin-1"), 0
        while m := CMAP_BEGIN.search(cmap, pos):  # one pass: an unclosed block ends the map
            stop = cmap.find("end" + m.group(1), m.end())
            if stop < 0:
                break
            tokens = CMAP_TOKEN.findall(cmap, m.end(), stop)
            n = 0
            try:
                while n < len(tokens):  # bfchar: <src> <dst>; bfrange: <lo> <hi> <dst> or <lo> <hi> [ <dst> ... ]
                    if m.group(1) == "bfchar":
                        mapped.add(int(tokens[n][1:-1].replace(" ", ""), 16))
                        n += 2
                    else:
                        first_code, last_code = (int(t[1:-1].replace(" ", ""), 16) for t in tokens[n:n + 2])
                        mapped.update(range(max(first_code, 0), min(last_code, 255) + 1))
                        n += 2
                        if tokens[n] == "[":
                            n = tokens.index("]", n) + 1
                        else:
                            n += 1
            except (ValueError, IndexError):
                return None  # a map not read as written: its glyphs are not trusted
            pos = stop
    out = set()
    for code in range(lo, hi + 1):
        if code not in mapped:
            out.add(chr(code))
            for codec in ("cp1252", "mac_roman"):  # how MuPDF may read it instead
                out.update(bytes([code]).decode(codec, errors="ignore"))
    cache[key] = frozenset(out)
    return cache[key]


SPLIT_SIGNS = set("()[]{}=+−-×·.,;:/|<>≤≥≠≈→←⇒⇔∈∉′'%°∆∞∂!∀∃∅∖∩∪⊂⊆⋅…") | {"\u20d7", "\u0302", "\u0303", "\u0304", "\u0305", "\u0307"}
FONT_REF = re.compile(r"/[^\s/<>\[\]()]+\s+(\d+)\s+0\s+R")


def _spend(doc: pymupdf.Document, chars: int) -> None:
    """Count characters of font objects read against a per-file budget (a value can be megabytes:
    an array in an object stream). Raises ValueError past it."""
    spent = doc.__dict__.get("_p2w_font_chars", 0) + chars
    doc.__dict__["_p2w_font_chars"] = spent
    if spent > FONT_READ_BUDGET:
        raise ValueError("font objects too large to read")


def _read_key(doc: pymupdf.Document, xref: int, key: str) -> tuple[str, str]:
    kind, val = doc.xref_get_key(xref, key)
    _spend(doc, len(val))
    return kind, val


def _read_object(doc: pymupdf.Document, xref: int) -> str:
    text = doc.xref_object(xref, compressed=True)
    _spend(doc, len(text))
    return text


def _font_info(doc: pymupdf.Document, x: int) -> tuple[str, list[str]]:
    """(Subtype, names) of a font object, read once per file: BaseFont, and a Type0 font's
    descendant's BaseFont, which the text spans carry."""
    cache = doc.__dict__.setdefault("_p2w_font_info", {})
    if x not in cache:
        sub = _read_key(doc, x, "Subtype")[1].lstrip("/")
        names = [_read_key(doc, x, "BaseFont")[1].lstrip("/")]
        if sub == "Type0":
            kind, desc = _read_key(doc, x, "DescendantFonts")  # Word writes "143 0 R" -> "[144 0 R]"
            if kind == "xref":
                desc = _read_object(doc, int(desc.split()[0]))
            first = re.search(r"(\d+)\s+0\s+R", desc[:200])
            if first:
                names.append(_read_key(doc, int(first.group(1)), "BaseFont")[1].lstrip("/"))
        cache[x] = (sub, names)
    return cache[x]


def _resources(doc: pymupdf.Document, ref: int, key: str) -> str | None:
    """The page's (or a form's) /Resources/<key> dictionary as text: inherited ones too, bounded.
    "": none. None: not found or too large."""
    for _ in range(8):
        kind, val = doc.xref_get_key(ref, "Resources")
        if kind != "null":
            break
        if doc.xref_get_key(ref, "Subtype")[1] == "/Form":
            return ""  # a form without resources of its own draws with the page's
        kind, val = doc.xref_get_key(ref, "Parent")
        if kind != "xref":
            return None
        ref = int(val.split()[0])
    else:
        return None
    dkind, dval = doc.xref_get_key(int(val.split()[0]), key) if kind == "xref" else doc.xref_get_key(ref, f"Resources/{key}")
    if dkind == "xref":
        dval = doc.xref_object(int(dval.split()[0]), compressed=True)
    elif dkind != "dict":
        return ""
    return None if len(dval) > FONT_LIST_MAX else dval


def _page_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> list[tuple[int, str, list[str]]] | None:
    """(xref, Subtype, names) of the fonts the page's own resources name (inherited ones too),
    read from the /Font dictionary itself, bounded (FONT_LIST_MAX, FONT_READ_BUDGET), not by listing
    fonts: a file can name thousands, in forms too. None: no resources, a dictionary too large, or a
    Unicode math font of the page's forms (N-up pages, pasted PDF pictures) that the page itself does
    not list – its glyphs would pass as the page's own font of the same name."""
    fval = _resources(doc, page.xref, "Font")
    if fval is None:
        return None
    cache = doc.__dict__.setdefault("_p2w_font_lists", {})  # pages share their font dictionaries
    if fval not in cache:
        xrefs = list(dict.fromkeys(int(m.group(1)) for m in FONT_REF.finditer(fval)))
        cache[fval] = [(x, *_font_info(doc, x)) for x in xrefs]
    fonts = cache[fval]
    own = {x for x, _sub, _names in fonts}
    forms, seen, forms_seen = [(page.xref, 0)], set(), 0
    while forms:
        ref, depth = forms.pop()
        xval = _resources(doc, ref, "XObject")
        if xval is None:
            return None
        _spend(doc, len(xval))
        for m in FONT_REF.finditer(xval):
            x = int(m.group(1))
            if x in seen:
                continue
            seen.add(x)
            if doc.xref_get_key(x, "Subtype")[1] != "/Form":
                continue  # images
            forms_seen += 1
            if forms_seen > FORMS_MAX or depth > 2:
                return None
            ffonts = _resources(doc, x, "Font")
            if ffonts is None:
                return None
            _spend(doc, len(ffonts))
            for fm in FONT_REF.finditer(ffonts):
                fx = int(fm.group(1))
                if fx not in own and any(UNICODE_MATH_FONT.search(n) for n in _font_info(doc, fx)[1]):
                    return None
            forms.append((x, depth + 1))
    return fonts


def _math_fonts(page: pymupdf.Page) -> tuple[frozenset[str], frozenset[str]] | None:
    """(text the page's Unicode math fonts give without a real mapping (see _simple_font_unmapped),
    names of the page's fonts). None: glyphs that cannot be told apart, so the page's formulas stay
    pictures – a CID font's glyph outside its map (MuPDF writes the glyph id as the character: "ӈ"
    for ℎ; read without that, it is U+FFFD), a font list or map too large to read, a broken font."""
    doc = page.parent
    try:
        fonts = _page_fonts(doc, page)
        if fonts is None:
            return None
        names = frozenset(_font_key(n) for _x, _sub, ns in fonts for n in ns)
        simple = [x for x, sub, ns in fonts if sub != "Type0" and any(UNICODE_MATH_FONT.search(n) for n in ns)]
        if len(simple) > MATH_FONTS_MAX:
            return None
        unmapped: set[str] = set()
        for xref in simple:
            chars = _simple_font_unmapped(doc, xref)
            if chars is None:
                return None
            unmapped |= chars
        if any(sub == "Type0" for _x, sub, _ns in fonts):  # whatever its name: the span's own name may differ
            for b in page.get_text("dict", flags=NO_CID_TEXT)["blocks"]:
                for ln in b.get("lines", ()):
                    for s in ln["spans"]:
                        if UNICODE_MATH_FONT.search(s["font"]) and "\ufffd" in s["text"]:
                            return None
        return frozenset(unmapped), names
    except Exception:  # noqa: BLE001 – a broken font object: pictures, the file still converts
        return None


# Serif fonts with Vietnamese glyphs, for re-setting rotated watermark text.
SERIF_FONTS = (
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "C:/Windows/Fonts/times.ttf",
)


@dataclass
class _Seg:
    x0: float
    x1: float
    runs: list[Run]
    chars: list[dict] = field(default_factory=list, repr=False)

    @property
    def natural_width(self) -> float:
        """Width with the font's own advances, i.e. before justification
        stretched or squeezed the spaces."""
        chars = self.chars
        start, end = 0, len(chars)
        while start < end and not chars[start]["c"].strip():
            start += 1
        while end > start and not chars[end - 1]["c"].strip():
            end -= 1
        return sum(c["bbox"][2] - c["bbox"][0] for c in chars[start:end])


@dataclass
class _Row:
    block: int
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    size: float
    segs: list[_Seg]
    translucent: bool = False

    @property
    def text(self) -> str:
        return "".join(r.text for seg in self.segs for r in seg.runs)

    @property
    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.x0, self.y0, self.x1, self.y1)

    @property
    def natural_right(self) -> float:
        return self.segs[-1].x0 + self.segs[-1].natural_width


@dataclass
class _Rule:
    x0: float
    x1: float
    y: float
    index: int


def _half(size: float) -> float:
    return round(size * 2) / 2


def _family(font: str) -> str | None:
    name = font.split("+", 1)[-1]
    if not name or name.startswith("Type3"):
        return None
    name = re.split(r"[-,]", name)[0]
    name = re.sub(r"(PSMT|PS|MT)$", "", name)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip() or None


def _bold(span: dict) -> bool:
    return bool(span["flags"] & FLAG_BOLD) or bool(
        re.search(r"bold|black|heavy|demi", span["font"], re.I)
    )


def _italic(span: dict) -> bool:
    return bool(span["flags"] & FLAG_ITALIC) or bool(re.search(r"italic|oblique", span["font"], re.I))


def _color(span: dict, honour_alpha: bool = True) -> str | None:
    rgb = span["color"] & 0xFFFFFF
    alpha = span.get("alpha", 255)
    if honour_alpha and alpha < 255:
        # Word text has no transparency: blend the colour over white paper.
        f = alpha / 255
        channels = [round(255 - (255 - ((rgb >> sh) & 255)) * f) for sh in (16, 8, 0)]
        rgb = (channels[0] << 16) | (channels[1] << 8) | channels[2]
    code = f"{rgb:06X}"
    return None if code == "000000" else code


def _hex(rgb) -> str:
    return "".join(f"{round(max(0.0, min(1.0, c)) * 255):02X}" for c in rgb[:3])


def _highlight_paint(d: dict) -> bool:
    """An opaque, coloured, unstroked fill: what Word draws for highlighted or
    shaded text (white boxes and black bars are not highlights)."""
    fill = d.get("fill")
    if not fill or d.get("color") is not None or (d.get("fill_opacity") or 1.0) < 0.95:
        return False
    return min(fill[:3]) < 0.97 and max(fill[:3]) > 0.15


@cache
def _serif_font() -> pymupdf.Font:
    for path in SERIF_FONTS:
        if Path(path).exists():
            return pymupdf.Font(fontfile=path)
    return pymupdf.Font("tiro")  # built-in Times: Latin-1 only


def _dominant_style(row: _Row) -> tuple:
    weights: dict[tuple, int] = {}
    for seg in row.segs:
        for r in seg.runs:
            key = (r.bold, r.italic, r.color)
            weights[key] = weights.get(key, 0) + len(r.text.strip())
    return max(weights, key=weights.get)


def _merge_boxes(rects: list[pymupdf.Rect], gap: float) -> list[pymupdf.Rect]:
    """Union boxes that touch within `gap`; works for zero-height lines too."""
    boxes = [[r.x0 - gap, r.y0 - gap, r.x1 + gap, r.y1 + gap] for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        for b in boxes:
            for o in out:
                if b[0] <= o[2] and o[0] <= b[2] and b[1] <= o[3] and o[1] <= b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(list(b))
        boxes = out
    return [pymupdf.Rect(b[0] + gap, b[1] + gap, b[2] - gap, b[3] - gap) for b in boxes]


def _join_strips(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Images set side by side in one band (a line of WordArt text exported as
    strips, its missing letters left as real text on top) become one region,
    so the line is kept whole; overlapping or nested images stay apart."""
    out: list[pymupdf.Rect] = []
    for r in sorted(rects, key=lambda r: r.x0):
        for k, o in enumerate(out):
            same_band = abs(r.y0 - o.y0) <= 2 and abs(r.y1 - o.y1) <= 2
            if same_band and r.x0 <= o.x1 + 2 and r.x1 > o.x1:
                out[k] = o | r
                break
        else:
            out.append(pymupdf.Rect(r))
    return out


def _fraction_parts(bar: pymupdf.Rect, marks: list[tuple]) -> list[int]:
    """Formula glyphs (node indexes) of the fraction a thin bar draws: a
    numerator right above it and a denominator right below it, both centred
    within it. Glyph boxes reach the font's descent, so an underline runs
    through the boxes of the text it underlines, while a fraction bar keeps
    clear of its numerator and denominator."""
    over, under = [], []
    for rect, size, _c, node in marks:
        cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        if not bar.x0 - 1 <= cx <= bar.x1 + 1:
            continue
        if cy < bar.y0 and bar.y0 - 0.5 * size <= rect.y1 <= bar.y1 + 0.35 * size:
            if rect.y1 > bar.y1 + 0.5:
                return []
            over.append(node)
        elif cy > bar.y1 and bar.y0 - 0.35 * size <= rect.y0 <= bar.y1 + 0.5 * size:
            if rect.y0 < bar.y0 - 0.5:
                return []
            under.append(node)
    over, under = [k for k in over if k is not None], [k for k in under if k is not None]
    if not (over and under):
        return []
    # a part on the baseline of text beside the bar belongs to that text line (a lone glyph of the line above
    # an overline "\overline{abc}"): a fraction's parts sit off the baseline of the line around them
    parts = set(over + under)
    mine = [(rect, size, c["origin"][1]) for rect, size, c, node in marks if node in parts]
    lo_y = min(oy - 0.1 * size for _r, size, oy in mine)
    hi_y = max(oy + 0.1 * size for _r, size, oy in mine)
    lo_x = min(r.x0 - 1.5 * size for r, size, _oy in mine)
    hi_x = max(r.x1 + 1.5 * size for r, size, _oy in mine)
    beside = [(r2, c2["origin"][1]) for r2, _s2, c2, n2 in marks
              if n2 is None and lo_y <= c2["origin"][1] <= hi_y and r2.x1 >= lo_x and r2.x0 <= hi_x
              and (r2.x1 < bar.x0 - 1 or r2.x0 > bar.x1 + 1)]
    if len(mine) * len(beside) > 20000:
        return []  # a crowd of glyphs around one bar: not read as a fraction (its formula stays a picture)
    for rect, size, oy in mine:
        for r2, oy2 in beside:
            if abs(oy2 - oy) <= 0.1 * size and max(r2.x0 - rect.x1, rect.x0 - r2.x1) <= 1.5 * size:
                return []
    return over + under


def _split_lines(members: list[int], nodes: list[list], fractions: list[tuple[int, list[int]]]) -> list[list[int]]:
    """A formula group cut into its printed lines. Two formulas on lines set
    close together can join through a script of one that sits over a glyph of
    the other; each line is one formula (one picture or equation in its line).
    A fraction stays with the line its bar is on, and so does a script inside
    its box (the exponent of a numerator). Rows of one formula (a system, a
    matrix) sit closer than a line apart, have no formula glyph of their own or
    are held by other strokes: the group stays whole."""
    parts = {k for bar, ks in fractions if bar in members for k in (bar, *ks)}
    glyphs = [k for k in members if nodes[k][4] is not None and k not in parts]
    # a stroke that is not a fraction bar read as one (a display fraction's bar set apart, a radical)
    # may hold the lines together: such a group stays whole
    if not glyphs or any(nodes[k][3] is not None and k not in parts for k in members):
        return [members]
    top = max(nodes[k][2] for k in glyphs)
    baselines = sorted(nodes[k][4]["origin"][1] for k in glyphs if nodes[k][2] >= 0.85 * top)
    lines = [[baselines[0]]]
    for y in baselines[1:]:
        if y - lines[-1][-1] > 0.9 * top:
            lines.append([y])
        else:
            lines[-1].append(y)
    if len(lines) == 1:
        return [members]
    ys = [median(line) for line in lines]
    nearest = lambda y: min(range(len(ys)), key=lambda i: abs(ys[i] - y))  # noqa: E731
    line_of: dict[int, int] = {}
    bars = {bar: ks for bar, ks in fractions if bar in members}
    boxes = []
    for k, ks in bars.items():  # the bar sits on the maths axis, a quarter em over the baseline
        line = nearest(nodes[k][0].y0 + 0.25 * top)
        for j in (k, *ks):
            line_of.setdefault(j, line)  # a part of nested bars once
        box = pymupdf.Rect(nodes[k][0])
        for j in ks:
            box |= nodes[j][0]
        boxes.append((pymupdf.Rect(box.x0 - 1, box.y0 - 0.15 * top, box.x1 + 0.3 * top, box.y1 + 0.15 * top), line))
    if len(members) * max(len(boxes), 1) > 20000:
        return [members]  # a huge group: kept whole (one picture) rather than sorted glyph by glyph
    boxes = [(tuple(box), line) for box, line in boxes]
    for k in members:
        if k in line_of or k in parts:
            continue
        r = nodes[k][0]
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        inside = {line for (x0, y0, x1, y1), line in boxes if x0 <= cx <= x1 and y0 <= cy <= y1}
        if len(inside) > 1:
            return [members]
        if inside:
            line = inside.pop()
            # a script of the next line under the denominator is that line's: the group stays whole
            oy = nodes[k][4]["origin"][1] if nodes[k][4] is not None else None
            if oy is not None and nearest(oy) != line:
                return [members]  # its own baseline is nearer the other line: which one it belongs to is unclear
            line_of[k] = line
        elif nodes[k][4] is not None:
            oy = nodes[k][4]["origin"][1]
            line_of[k] = nearest(oy)
            if oy - ys[line_of[k]] > 0.45 * top:
                return [members]  # no script drops that far under its base: a raised script of the line below
        else:
            line_of[k] = nearest((r.y0 + r.y1) / 2 + 0.25 * top)
    out: list[list[int]] = [[] for _ in ys]
    for k in members:
        out[line_of[k]].append(k)
    out = [part for part in out if part]
    if any(not any(nodes[k][1] == 2 or k in bars for k in part) for part in out):
        return [members]
    for part in out:
        # a bracket open on one line has its rows on the others (a piecewise function's brace), and a glyph
        # that is no ordinary letter, digit or sign (Word for Mac's stretched brace reads "«") may be one
        depth = 0
        for k in sorted((k for k in part if nodes[k][4] is not None), key=lambda k: nodes[k][0].x0):
            c = nodes[k][4]["c"]
            depth += (c in "([{") - (c in ")]}")
            if depth < 0 or not (c.isalnum() or _math_alnum(c) or c in SPLIT_SIGNS):
                return [members]
        if depth:
            return [members]
    return out


def _overlaps(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    """Boxes sharing a good part of the smaller one (not just a grazing edge)."""
    common = a & b
    return not common.is_empty and abs(common) >= 0.3 * min(abs(a), abs(b))


def _mostly_inside(r: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    common = r & box
    return not common.is_empty and abs(common) >= 0.6 * abs(r)


def _min_size(rect: pymupdf.Rect, minimum: float = 2.0) -> pymupdf.Rect:
    r = pymupdf.Rect(rect)
    if r.width < minimum:
        r.x0, r.x1 = r.x0 - (minimum - r.width) / 2, r.x1 + (minimum - r.width) / 2
    if r.height < minimum:
        r.y0, r.y1 = r.y0 - (minimum - r.height) / 2, r.y1 + (minimum - r.height) / 2
    return r


def _rules(drawings: list[dict]) -> tuple[list[list[float]], list[list[float]]]:
    """Axis-aligned rules, joined end to end: [y, x0, x1] across and [x, y0, y1]
    down. From thin filled boxes (how Word draws table borders, often 0pt
    thick), stroked lines and the sides of stroked boxes; white paint is not a rule."""
    hs, vs = [], []
    for d in drawings:
        fill, stroke = d.get("fill"), d.get("color")
        dark_fill = fill is not None and min(fill[:3]) < 0.97
        dark_stroke = stroke is not None and min(stroke[:3]) < 0.97
        for item in d["items"]:
            if item[0] == "re":
                r = pymupdf.Rect(item[1])
                if dark_fill and r.width <= 1.5 and r.height > 2:
                    vs.append([(r.x0 + r.x1) / 2, r.y0, r.y1])
                elif dark_fill and r.height <= 1.5 and r.width > 2:
                    hs.append([(r.y0 + r.y1) / 2, r.x0, r.x1])
                elif dark_stroke and r.width > 2 and r.height > 2:
                    hs += [[r.y0, r.x0, r.x1], [r.y1, r.x0, r.x1]]
                    vs += [[r.x0, r.y0, r.y1], [r.x1, r.y0, r.y1]]
            elif item[0] == "l" and dark_stroke:
                p, q = item[1], item[2]
                if abs(p.y - q.y) <= 0.5 and abs(p.x - q.x) > 2:
                    hs.append([p.y, min(p.x, q.x), max(p.x, q.x)])
                elif abs(p.x - q.x) <= 0.5 and abs(p.y - q.y) > 2:
                    vs.append([p.x, min(p.y, q.y), max(p.y, q.y)])

    def join(segs: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for s in sorted(segs, key=lambda s: (round(s[0] / 0.8), s[1])):
            if out and abs(out[-1][0] - s[0]) <= 0.8 and s[1] <= out[-1][2] + 1.5:
                out[-1][2] = max(out[-1][2], s[2])
            else:
                out.append(list(s))
        return out

    return join(hs), join(vs)


def _covered(segs: list[list[float]], at: float, lo: float, hi: float) -> float:
    """Share of [lo, hi] that rules at coordinate `at` cover."""
    parts = sorted((max(s[1], lo), min(s[2], hi)) for s in segs if abs(s[0] - at) <= 1.5 and s[2] > lo and s[1] < hi)
    total, end = 0.0, lo
    for a, b in parts:
        if b > end:
            total += b - max(a, end)
            end = b
    return total / max(hi - lo, 1e-6)


def _ruled_grids(drawings: list[dict], words: list[pymupdf.Rect]) -> list[tuple]:
    """Tables read from the rules: each set of touching rules with a closed frame
    becomes a grid at its rule positions; where a rule between two cells is
    missing, they are one merged cell. A grid needs words in a cell (an answer
    key may fill 3 of 10) and no curve or slanted line inside: that is a graph."""
    hs, vs = _rules(drawings)
    plotted = [
        pymupdf.Rect(d["rect"]) for d in drawings
        if any(it[0] in ("c", "qu") or (it[0] == "l" and abs(it[1].x - it[2].x) > 0.5 and abs(it[1].y - it[2].y) > 0.5)
               for it in d["items"])
    ]
    nodes = [("h", s) for s in hs if s[2] - s[1] > 6] + [("v", s) for s in vs if s[2] - s[1] > 6]
    parent = list(range(len(nodes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, (ki, a) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            kj, b = nodes[j]
            if ki == kj:
                continue
            h, v = (a, b) if ki == "h" else (b, a)
            if h[1] - 1.5 <= v[0] <= h[2] + 1.5 and v[1] - 1.5 <= h[0] <= v[2] + 1.5:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for k in range(len(nodes)):
        groups.setdefault(find(k), []).append(k)
    out = []
    for members in groups.values():
        gh = [nodes[k][1] for k in members if nodes[k][0] == "h"]
        gv = [nodes[k][1] for k in members if nodes[k][0] == "v"]
        if len(gh) < 2 or len(gv) < 2:
            continue
        xs, ys = _cluster_values([s[0] for s in gv], 1.5), _cluster_values([s[0] for s in gh], 1.5)
        if len(xs) < 2 or len(ys) < 2 or (len(xs) - 1) * (len(ys) - 1) < 2:
            continue
        x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
        frame = min(_covered(gh, y0, x0, x1), _covered(gh, y1, x0, x1), _covered(gv, x0, y0, y1), _covered(gv, x1, y0, y1))
        if frame < 0.9:
            continue
        nrows, ncols = len(ys) - 1, len(xs) - 1
        cell = list(range(nrows * ncols))

        def root(a: int) -> int:
            while cell[a] != a:
                cell[a] = cell[cell[a]]
                a = cell[a]
            return a

        for r in range(nrows):
            for c in range(ncols):
                if c and _covered(gv, xs[c], ys[r], ys[r + 1]) < 0.7:
                    cell[root(r * ncols + c)] = root(r * ncols + c - 1)
                if r and _covered(gh, ys[r], xs[c], xs[c + 1]) < 0.7:
                    cell[root(r * ncols + c)] = root((r - 1) * ncols + c)
        areas: dict[int, list[tuple[int, int]]] = {}
        for k in range(nrows * ncols):
            areas.setdefault(root(k), []).append(divmod(k, ncols))
        merges, filled = [], 0
        for spots in areas.values():
            rs = [r for r, _ in spots]
            cs = [c for _, c in spots]
            r0, c0, rn, cn = min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            if len(spots) > 1 and rn * cn == len(spots):
                merges.append((r0, c0, rn, cn))
            box = pymupdf.Rect(xs[c0], ys[r0], xs[c0 + cn], ys[r0 + rn])
            filled += any(box.contains(pymupdf.Point((w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2)) for w in words)
        frame_rect = pymupdf.Rect(x0, y0, x1, y1)
        inner = pymupdf.Rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1)
        if not filled or any(not (r & inner).is_empty for r in plotted):
            continue
        out.append((frame_rect, xs, ys, merges))
    return out


def _merged_cells(table, xs: list[float], ys: list[float]) -> list[tuple[int, int, int, int]]:
    """(row, column, rows, columns) of each cell PyMuPDF reports over more than
    one grid cell (the rows and columns it covers report None there)."""
    out = []
    for row in table.rows:
        for cell in row.cells:
            if cell is None:
                continue
            x0, y0, x1, y1 = cell
            rs = 1 + sum(y0 + 1 < y < y1 - 1 for y in ys)
            cs = 1 + sum(x0 + 1 < x < x1 - 1 for x in xs)
            if rs > 1 or cs > 1:
                r = min(range(len(ys) - 1), key=lambda k: abs(ys[k] - y0))
                c = min(range(len(xs) - 1), key=lambda k: abs(xs[k] - x0))
                out.append((r, c, min(rs, len(ys) - 1 - r), min(cs, len(xs) - 1 - c)))
    return out


def _cluster_values(values: list[float], tol: float) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


class _Extractor:
    def __init__(self, page: pymupdf.Page, ocr_languages: list[str] | None):
        self.page = page
        self.width, self.height = page.rect.width, page.rect.height
        self.ocr_languages = ocr_languages
        self.drawings = page.get_drawings()
        self.rules = [
            _Rule(d["rect"].x0, d["rect"].x1, (d["rect"].y0 + d["rect"].y1) / 2, i)
            for i, d in enumerate(self.drawings)
            if d["rect"].height <= 1.6 and d["rect"].width >= 3
        ]
        self.used_drawings: set[int] = set()
        self.table_rects: list[pymupdf.Rect] = []  # set before rows(): table shading is not a highlight
        self.table_lines: list[tuple] = []  # (rect, xs, ys) of those tables: their borders join no formula
        self.highlights = [
            (pymupdf.Rect(d["rect"]), _hex(d["fill"]), i)
            for i, d in enumerate(self.drawings)
            if _highlight_paint(d) and 3 <= d["rect"].height <= 40 and d["rect"].width <= 0.9 * self.width
        ]
        self.formulas: list[dict] = []  # {"rect", "size", "run"}: formulas kept as pictures
        self.formula_chars: set[int] = set()  # id() of the glyphs drawn by those pictures
        self._forms_doc: pymupdf.Document | None = None  # the page drawing only its MathType forms (_form_contents)
        self.formula_drawings: set[int] = set()  # bars, radicals, arrows drawn as part of a formula
        self.legacy_spans = 0
        self.invisible_layer = False
        self.rotated_lines: list[dict] = []
        self._stripped_doc = None
        self._ground_doc = None
        self._unmarked_doc = None
        self._ink_refs: tuple[float | None, float | None] | None = None
        self._bold_cache: dict[tuple, bool] = {}
        self._math_fonts: tuple | None | bool = False  # _math_fonts(page), worked out when needed
        self._tiny: list | np.ndarray = []  # glyphs under 4 pt: left out of formulas, but drawn
        self._marks: list[tuple[str, tuple]] | None = None  # combining marks as drawn (get_texttrace)
        self._page_ink: np.ndarray | None | bool = None  # the page in grey at 150 dpi, for brackets (_ink_height)

    # --- spans and rows ---------------------------------------------------

    def _table_rule(self, rule: _Rule) -> bool:
        """A ruled table's own line: the border under text in a tight row (or
        under "Đáp án" just above the table) is no underline."""
        return any(
            t.x0 - 1.5 <= rule.x0 and rule.x1 <= t.x1 + 1.5 and t.y0 - 1.5 <= rule.y <= t.y1 + 1.5
            for t in self.table_rects
        )

    def _char_underlined(self, bbox, baseline: float, size: float) -> bool:
        """A thin rule just below the baseline under the character's centre."""
        cx = (bbox[0] + bbox[2]) / 2
        for rule in self.rules:
            if rule.index in self.formula_drawings or self._table_rule(rule):
                continue
            if baseline - 0.5 <= rule.y <= baseline + 0.3 * size + 0.6 and rule.x0 - 0.5 <= cx <= rule.x1 + 0.5:
                self.used_drawings.add(rule.index)
                return True
        return False

    def _char_highlight(self, bbox, size: float) -> str | None:
        """Colour of a filled box about one line high behind the character's centre."""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        for rect, colour, index in self.highlights:
            if not 0.6 * size <= rect.height <= 2.5 * size:
                continue
            if rect.x0 - 0.5 <= cx <= rect.x1 + 0.5 and rect.y0 - 0.5 <= cy <= rect.y1 + 0.5:
                if any(pymupdf.Rect(t.x0 - 1, t.y0 - 1, t.x1 + 1, t.y1 + 1).contains(rect) for t in self.table_rects):
                    continue  # cell shading of a ruled table
                self.used_drawings.add(index)
                return colour
        return None

    def _underlined(self, bbox, baseline: float, size: float) -> bool:
        x0, x1 = bbox[0], bbox[2]
        width = max(x1 - x0, 1e-6)
        for rule in self.rules:
            if self._table_rule(rule):
                continue
            if baseline - 0.5 <= rule.y <= baseline + 0.3 * size + 0.6:
                if min(rule.x1, x1) - max(rule.x0, x0) >= 0.6 * width:
                    self.used_drawings.add(rule.index)
                    return True
        return False

    def _runs(self, span: dict, main: float, baseline: float) -> list[Run]:
        """The span's runs: one, or several where underline/highlight change mid-span."""
        if span.get("formula") is not None:
            f = span["formula"]
            if f["pieces"]:
                return self._equation_runs(f, main, baseline)
            box, image = self._formula_picture(f)
            f["box"] = box
            run = Run(
                text=OBJ, size=_half(main), image=image, image_w=box.width,
                image_h=box.height, image_depth=box.y1 - baseline,
            )
            f["run"] = run
            return [run]
        text, legacy_family = decode_span(span["text"], span["font"])
        if not text:
            return []
        if legacy_family:
            self.legacy_spans += 1
        # No-break spaces would stop Word from wrapping lines. A soft hyphen
        # (in TCVN3 it is "ư", already decoded) is a printed hyphen unless it
        # ends the span, where it may be a line-end hyphenation point.
        text = text.replace("\xa0", " ")
        body, tail = (text[:-1], text[-1]) if text.endswith("\xad") else (text, "")
        text = body.replace("\xad", "-") + tail
        size, origin_y = span["size"], span["origin"][1]
        script = None
        if text.strip() and size < 0.85 * main:
            if span["flags"] & FLAG_SUPERSCRIPT or origin_y < baseline - 0.12 * main:
                script = "super"
            elif origin_y > baseline + 0.04 * main:
                script = "sub"
        unnamed = span["font"].split("+", 1)[-1].startswith("Type3")
        style = dict(
            # Word shrinks super/subscript itself, so keep the line's size.
            size=_half(main if script else size),
            bold=_bold(span) or (unnamed and bool(text.strip()) and self._looks_bold(span)),
            italic=_italic(span),
            color=_color(span, honour_alpha=not self.invisible_layer),
            font=legacy_family or _family(span["font"]),
            script=script,
        )
        chars = span.get("chars") or []
        if len(chars) == len(text):
            flags = [(self._char_underlined(c["bbox"], baseline, size), self._char_highlight(c["bbox"], size)) for c in chars]
        else:  # decoded legacy text has its own length: decide for the whole span
            whole = (bool(text.strip()) and self._underlined(span["bbox"], baseline, size), None)
            flags = [whole] * len(text)
        runs: list[Run] = []
        start = 0
        for i in range(1, len(text) + 1):
            if i == len(text) or flags[i] != flags[start]:
                piece = text[start:i]
                underline, highlight = flags[start]
                runs.append(Run(text=piece, underline=underline and bool(piece.strip()), highlight=highlight, **style))
                start = i
        return runs

    # --- spans without font information (Type3) ---------------------------

    def _ink_density(self, span: dict) -> float | None:
        """Ink per unit of text length and em: stroke weight, colour-independent."""
        rect = pymupdf.Rect(span["bbox"]) & self.page.rect
        if rect.is_empty or not span["text"].strip():
            return None
        pix = self.page.get_pixmap(dpi=150, clip=rect, colorspace=pymupdf.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
        darkest = np.percentile(img, 2)
        if darkest > 240:
            return None
        ink = np.count_nonzero(img < (255 + darkest) / 2)
        em_px = span["size"] * 150 / 72
        return ink / (pix.width * em_px)

    def _looks_bold(self, span: dict) -> bool:
        """Type3 fonts carry no name or weight; compare the span's stroke weight
        with regular and bold spans of this page."""
        key = (span["font"], round(span["size"], 1))
        if key in self._bold_cache:
            return self._bold_cache[key]
        if self._ink_refs is None:
            regular, bold = [], []
            for b in self.page.get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for sp in line["spans"]:
                        if len(sp["text"].strip()) < 8 or sp["font"].startswith("Type3"):
                            continue
                        target = bold if _bold(sp) else regular
                        if len(target) < 5:
                            target.append(sp)
            dens = [[d for d in map(self._ink_density, group) if d] for group in (regular, bold)]
            self._ink_refs = tuple(median(d) if d else None for d in dens)
        density = self._ink_density(span)
        reg, bol = self._ink_refs
        if density is None or reg is None:
            result = False
        elif bol is not None:
            result = abs(density - bol) < abs(density - reg)
        else:
            result = density > 1.3 * reg
        self._bold_cache[key] = bool(result)
        return bool(result)

    def _row(self, block: int, line: dict, skip_invisible: bool, hosted=()) -> _Row | None:
        dx, dy = line["dir"]
        if abs(dy) > 0.02 or dx < 0:  # rotated text (watermarks, stamps): collected in rows()
            return None
        # Text under 4pt cannot be read on paper (a 3pt white "c" left as a marker): dropped.
        spans = [
            s for s in line["spans"]
            if s["text"] and s["size"] >= 4 and not (skip_invisible and s.get("alpha", 255) == 0)
        ]
        if self.formulas:
            spans = self._place_formulas(spans, hosted)
        spans = _split_at_tabs(spans)
        visible = [s for s in spans if s["text"].strip()]
        if not visible:
            return None
        main = max(visible, key=lambda s: (len(s["text"].strip()), s["size"]))["size"]
        baseline = median(s["origin"][1] for s in visible if abs(s["size"] - main) < 0.5)

        segs: list[_Seg] = []
        last_x1 = None
        for s in spans:
            runs = self._runs(s, main, baseline)
            if not runs:
                continue
            text = "".join(r.text for r in runs)
            x0, x1 = s["bbox"][0], s["bbox"][2]
            n = max(len(text), 1)
            # Span boxes include their spaces; measure gaps between visible ink
            # (from the characters' own boxes when the text kept its length).
            ink = [c["bbox"] for c in s.get("chars") or [] if c["c"].strip()]
            if ink and len(s["chars"]) == len(text):
                vis_x0, vis_x1 = ink[0][0], ink[-1][2]
            else:
                vis_x0 = x0 + (len(text) - len(text.lstrip())) * (x1 - x0) / n
                vis_x1 = x0 + len(text.rstrip()) * (x1 - x0) / n
            blank = not text.strip()
            # A gap wider than ~1em inside a line is a tab or a column jump.
            jump = last_x1 is not None and (vis_x0 - last_x1 > max(0.9 * main, 8) or s.get("tab_before", False))
            new_seg = not segs or (not blank and jump)
            if new_seg:
                if blank:
                    continue
                if segs:
                    segs[-1].runs[-1].text = segs[-1].runs[-1].text.rstrip()
                while runs and not runs[0].text.strip():
                    runs.pop(0)
                runs[0].text = runs[0].text.lstrip()
                segs.append(_Seg(vis_x0, vis_x1, runs))
                segs[-1].chars.extend(s["chars"])
            else:
                segs[-1].runs.extend(runs)
                segs[-1].chars.extend(s["chars"])
                if not blank:
                    segs[-1].x1 = max(segs[-1].x1, vis_x1)
            if not blank:
                last_x1 = vis_x1
        if not segs:
            return None
        y0, y1 = line["bbox"][1], line["bbox"][3]
        row = _Row(block, segs[0].x0, y0, max(seg.x1 for seg in segs), y1, baseline, main, segs)
        row.translucent = not self.invisible_layer and all(s.get("alpha", 255) < 255 for s in visible)
        return row

    def rows(self) -> list[_Row]:
        flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
        blocks = self.page.get_text("rawdict", flags=flags)["blocks"]
        spans = [s for b in blocks if b["type"] == 0 for ln in b["lines"] for s in ln["spans"]]
        for s in spans:  # rawdict has per-character boxes instead of "text"
            s["text"] = "".join(c["c"] for c in s["chars"])
        for b in blocks:
            if b["type"] == 0:
                b["lines"] = _join_interleaved(b["lines"])
        total = sum(len(s["text"].strip()) for s in spans)
        hidden = sum(len(s["text"].strip()) for s in spans if s.get("alpha", 255) == 0)
        # Searchable scans carry an invisible OCR layer: keep it as the text.
        self.invisible_layer = total > 0 and hidden > 0.5 * total
        # rotated text (watermarks, stamps) goes to the background; formula pictures leave it out
        self.rotated_lines = [
            ln for b in blocks if b["type"] == 0 for ln in b["lines"]
            if (abs(ln["dir"][1]) > 0.02 or ln["dir"][0] < 0) and any(s["text"].strip() for s in ln["spans"])
        ]
        hosts: dict[int, list[dict]] = {}
        if not self.invisible_layer:
            self.formulas = self._find_formulas(blocks)
            hosts = self._formula_hosts(blocks)
        rows: list[_Row] = []
        for bno, b in enumerate(blocks):
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                row = self._row(bno, line, skip_invisible=not self.invisible_layer, hosted=hosts.get(id(line), ()))
                if row is None:
                    continue
                prev = rows[-1] if rows else None
                if prev and prev.block == bno and abs(row.baseline - prev.baseline) <= 0.3 * row.size:
                    # Same baseline in one block: further items of the row.
                    _add_row(prev, row)
                    continue
                rows.append(row)
        return _join_split_lines(rows)

    # --- formulas -----------------------------------------------------------

    def _find_formulas(self, blocks: list[dict]) -> list[dict]:
        """Regions of formulas without usable text: connected groups of formula
        glyphs (opaque spans, math alphanumerics), Symbol/math-font glyphs and
        the small drawings touching them (fraction bars, radicals, arrows). A
        group needs at least one formula glyph or a fraction bar; Symbol alone
        stays text."""
        nodes: list[list] = []  # [rect, kind, size, drawing index, char]; kind 2 formula, 1 joins, 0 drawing
        marks: list[tuple] = []  # every visible upright glyph: (rect, size, char, node index or None)
        span_of: dict[int, dict] = {}  # id(char) -> its span
        try:
            objects = self._mathtype_objects(blocks)  # whole formulas, read apart from the grouping below
        except Exception:  # noqa: BLE001 – a page this trips over keeps its formulas as before
            objects = []
        taken = {id(c) for o in objects for c in o["chars"]}
        taken_drawings = {i for o in objects for i in o["drawings"]}
        for b in blocks:
            if b["type"] != 0:
                continue
            # ")?" after a formula is punctuation of the sentence unless its block holds formula glyphs
            # (Word's export puts a MathType picture's glyphs in blocks of their own)
            has_formula = any(
                _opaque(s["text"]) or any(_math_alnum(ch) for ch in s["text"]) for ln in b["lines"] for s in ln["spans"]
            )
            for ln in b["lines"]:
                dx, dy = ln["dir"]
                if abs(dy) > 0.02 or dx < 0:
                    continue
                for s in ln["spans"]:
                    # Cambria Math with real text (a figure's labels "Đinh thép") is text:
                    # only glyphs without usable text, or math letters, start a formula.
                    formula_span = _opaque(s["text"])
                    joins = bool(SYMBOL_FONT.search(s["font"]) or MATH_FONT.search(s["font"])) or (
                        has_formula and _maybe_opaque(s["text"])
                    )
                    for c in s["chars"]:
                        if not c["c"].strip():
                            continue
                        if id(c) in taken:
                            span_of[id(c)] = s
                            continue
                        if s["size"] < 4:
                            self._tiny.append(pymupdf.Rect(c["bbox"]))
                            continue
                        span_of[id(c)] = s
                        kind = 2 if formula_span or _math_alnum(c["c"]) else 1 if joins else 0
                        marks.append((pymupdf.Rect(c["bbox"]), s["size"], c, len(nodes) if kind else None))
                        if kind:
                            nodes.append([pymupdf.Rect(c["bbox"]), kind, s["size"], None, c])
        if not nodes and not objects:
            return []
        in_table = lambda r: any(  # noqa: E731
            t.x0 - 1 <= r.x0 and r.x1 <= t.x1 + 1 and t.y0 - 1 <= r.y0 and r.y1 <= t.y1 + 1 for t in self.table_rects
        )
        # a ruled table's own borders (on its grid lines) join no formula: a formula in a cell would take the
        # border, and through it its neighbours' cells (Word draws each cell side as a rect of its own)
        on_grid = lambda r: any(  # noqa: E731
            pymupdf.Rect(t.x0 - 2, t.y0 - 2, t.x1 + 2, t.y1 + 2).contains(r) and (
                (r.height <= 2 and any(abs((r.y0 + r.y1) / 2 - y) <= 1.5 for y in ys))
                or (r.width <= 2 and any(abs((r.x0 + r.x1) / 2 - x) <= 1.5 for x in xs))
            )
            for t, xs, ys in self.table_lines
        )
        fractions: list[tuple[int, list[int]]] = []  # (bar node, glyph nodes over and under it)
        for i, d in enumerate(self.drawings):
            r = pymupdf.Rect(d["rect"])
            if on_grid(r) or i in taken_drawings:
                continue
            # an upright stroke outside ruled tables: an absolute-value bar or a bracket
            upright = r.width <= 1 and 0 < r.height <= 40 and not in_table(r)
            parts = (
                _fraction_parts(r, marks) if r.height <= 2 and 3 <= r.width <= 0.9 * self.width and not in_table(r) else []
            )
            if upright:  # a zero-width box is empty to PyMuPDF and would drop out of the region
                nodes.append([pymupdf.Rect(r.x0 - 0.5, r.y0, r.x1 + 0.5, r.y1), 0, None, i, None, True])
            elif parts or r.width <= 0.3 * self.width and r.height <= 40 and not r.is_empty or (r.height <= 2 and r.width <= 0.3 * self.width):
                if parts:
                    fractions.append((len(nodes), parts))
                nodes.append([r, 0, None, i, None, False])
        parent = list(range(len(nodes)))
        in_fraction = {id(nodes[k]) for _bar, parts in fractions for k in parts}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def near(a: list, b: list) -> bool:
            ra, rb = a[0], b[0]
            if a[1] == 0 or b[1] == 0:  # a drawing joins only formula glyphs it touches
                glyph = b if a[1] == 0 else a
                d = a if a[1] == 0 else b
                if glyph[1] == 0:
                    return False
                # "|x|": a bar about one line high, set apart like a letter, joins its neighbours
                if d[5] and 0.8 * glyph[2] <= d[0].height <= 3 * glyph[2]:
                    cy = (glyph[0].y0 + glyph[0].y1) / 2
                    if d[0].y0 <= cy <= d[0].y1 and max(glyph[0].x0 - d[0].x1, d[0].x0 - glyph[0].x1) <= 0.3 * glyph[2]:
                        return True
                if glyph[1] != 2:
                    return False
                # a bar sits a little off the glyph boxes (they end at the descender line)
                tol = 0.25 * glyph[2]
                return not (pymupdf.Rect(d[0].x0 - 1.5, d[0].y0 - tol, d[0].x1 + 1.5, d[0].y1 + tol) & glyph[0]).is_empty
            size = min(a[2], b[2])
            hgap = max(rb.x0 - ra.x1, ra.x0 - rb.x1)
            vgap = max(rb.y0 - ra.y1, ra.y0 - rb.y1)
            # on one line: neighbours, scripts (a script's gap scales with its base); the boxes of
            # two tightly set lines touch, which does not make them one line
            if vgap < -0.25 * size:
                return hgap <= 0.35 * max(a[2], b[2])
            # stacked without a bar (limits under a big operator): only a smaller glyph centred
            # right under/over a larger one, never two lines of text nor a fraction's part and
            # the line next to it
            if id(a) in in_fraction or id(b) in in_fraction:
                return False
            small, big = (a, b) if a[2] < b[2] else (b, a)
            centred = big[0].x0 <= (small[0].x0 + small[0].x1) / 2 <= big[0].x1
            return centred and vgap <= 0.3 * size and size <= 0.85 * big[2]

        order = sorted(range(len(nodes)), key=lambda k: nodes[k][0].y0)
        for x, a in enumerate(order):
            ra = nodes[a][0]
            for b in order[x + 1 :]:
                if nodes[b][0].y0 > ra.y1 + 20:
                    break
                if near(nodes[a], nodes[b]):
                    parent[find(a)] = find(b)
        # A fraction bar holds its numerator and denominator together, whatever their glyphs
        # (Word's equations set words "Năng lượng nhận được" in Cambria Math too).
        for bar, parts in fractions:
            for k in parts:
                parent[find(k)] = find(bar)
        groups: dict[int, list[int]] = {}
        for k in range(len(nodes)):
            groups.setdefault(find(k), []).append(k)
        bars = {bar for bar, _parts in fractions}
        found = []
        for members in (part for g in groups.values() for part in _split_lines(g, nodes, fractions)):
            # a formula glyph, or a fraction of math-font glyphs ("1/110" in Cambria Math)
            if not any(nodes[k][1] == 2 or k in bars for k in members):
                continue
            rect = pymupdf.Rect(nodes[members[0]][0])
            for k in members[1:]:
                rect |= nodes[k][0]
            found.append({
                "rect": rect,
                "chars": [nodes[k][4] for k in members if nodes[k][4] is not None],
                "drawings": {nodes[k][3] for k in members if nodes[k][3] is not None},
            })
        found = self._close_formulas(found, marks)
        size_of = {id(m[2]): m[1] for m in marks}
        for o in objects:
            rect = pymupdf.Rect(o["chars"][0]["bbox"])
            for c in o["chars"][1:]:
                rect |= c["bbox"]
            for i in o["drawings"]:
                rect |= self.drawings[i]["rect"]
            found.append({"rect": rect, "chars": o["chars"], "drawings": o["drawings"], "mt": True, "blanks": o["blanks"],
                          "form": o["form"], "strong": o["strong"]})
            size_of.update((id(c), span_of[id(c)]["size"]) for c in o["chars"])
        fraction_bars = {nodes[bar][3] for bar, _parts in fractions}
        out = []
        for f in found:
            sizes = [size_of[id(c)] for c in f["chars"]]
            # the main line's baseline: where the largest glyphs sit (not MathType's brackets, set larger)
            top = max([z for c, z in zip(f["chars"], sizes) if not (f.get("mt") and c["c"] in "()[]{}")] or sizes)
            baseline = median(c["origin"][1] for c, size in zip(f["chars"], sizes) if top - 0.5 < size <= top)
            try:
                pieces = self._equation_mt(f, span_of) if f.get("mt") else None
            except Exception:  # noqa: BLE001 – unreadable: its picture
                pieces = None
            if f.get("mt") and pieces is None and not f["strong"]:
                continue  # variables in italics, nothing the text does not show as well: left to the text
            self.formula_drawings |= f["drawings"]
            self.used_drawings |= f["drawings"]
            self.formula_chars.update(id(c) for c in f["chars"] + f.get("blanks", []))
            # what the picture shows: the glyph boxes (italics lean out of them) and the strokes
            parts = [(pymupdf.Rect(c["bbox"]), 0.2 * size, 0.1 * size) for c, size in zip(f["chars"], sizes)]
            ink = pymupdf.Rect(f["rect"].x0, min(c["origin"][1] - 0.7 * size for c, size in zip(f["chars"], sizes)),
                               f["rect"].x1, max(c["origin"][1] + 0.2 * size for c, size in zip(f["chars"], sizes)))
            for i in f["drawings"]:
                ink |= self.drawings[i]["rect"]
            parts += [(pymupdf.Rect(self.drawings[i]["rect"]), 1.0, 1.0) for i in f["drawings"]]
            out.append({
                "rect": f["rect"], "size": median(sizes), "baseline": baseline, "parts": parts, "run": None, "ink": ink,
                "pieces": pieces if f.get("mt") else self._equation(f, span_of, fraction_bars),
                "mt": bool(f.get("mt")), "form": f.get("form"),
            })
        return sorted(out, key=lambda f: (f["rect"].y0, f["rect"].x0))

    def _mathtype_objects(self, blocks: list[dict]) -> list[dict]:
        """MathType objects that Word for Windows exports as text: each drawn by a form of its own (Word's
        "/MetaN", the object's EMF picture: _form_frames), whose glyphs and strokes are known exactly
        (_form_contents), so the formula is taken whole and nothing of the lines around it. A form counts
        when everything it draws is found on the page, on horizontal lines, opaque, with no image, at most a
        few lines high, inside one cell of a ruled table, within FORMULA_GLYPHS_MAX, and it shows maths: a
        Symbol or MT Extra glyph, a fraction bar or a script ("strong"), or else only italic letters in
        short runs (variables, not an italic sentence) and no Vietnamese word. Word draws its other EMF
        pictures the same way (an Excel range, a diagram): those stay as they were.
        {"form", "chars", "blanks" (its spaces, which go with it), "drawings", "strong"} each."""
        frames = _form_frames(self.page)
        if not frames:
            return []
        got = _form_contents(self.page, frames)
        if got is None:
            return []
        self._forms_doc, self._forms_xref, contents = got
        self._symbol_maps = _form_symbol_maps(self.page.parent, {fr["xref"] for fr in frames})
        owners: dict[tuple, set[int]] = {}
        spaces: dict[tuple, int] = {}
        blanks: list[tuple] = []  # (space, the form drawing it or None)
        for k, c in enumerate(contents):
            for key in c["glyphs"]:
                owners.setdefault(key, set()).add(k)
            spaces.update((key, k) for key in c["spaces"])
        objs = [{"form": fr, "chars": [], "blanks": [], "drawings": set(), "found": set(), "spans": {},
                 "bad": bool(c["images"])} for fr, c in zip(frames, contents)]
        for b in blocks:
            if b["type"] != 0:
                continue
            for ln in b["lines"]:
                dx, dy = ln["dir"]
                upright = abs(dy) <= 0.02 and dx > 0
                for sp in ln["spans"]:
                    for ch in sp["chars"]:
                        key = (round(ch["origin"][0], 2), round(ch["origin"][1], 2))
                        if not ch["c"].strip():
                            if upright:
                                blanks.append((ch, spaces.get(key)))
                            continue
                        ks = owners.get(key)
                        if not ks:
                            continue
                        if len(ks) > 1 or not upright or sp.get("alpha", 255) < 255:
                            for k in ks:  # two forms' glyph, rotated or translucent text: a stamp, a watermark
                                objs[k]["bad"] = True
                            continue
                        k = next(iter(ks))
                        objs[k]["found"].add(key)
                        objs[k]["spans"][id(ch)] = sp
                        objs[k]["chars"].append(ch)
        by_rect: dict[tuple, list[int]] = {}
        for i, d in enumerate(self.drawings):
            by_rect.setdefault(tuple(round(v, 2) for v in d["rect"]), []).append(i)
        for o, c in zip(objs, contents):
            if o["found"] != c["glyphs"]:
                o["bad"] = True  # a glyph of the form the page's text does not hold (clipped, invisible)
            for r in c["strokes"]:
                ids = by_rect.get(tuple(round(v, 2) for v in r))
                if not ids:
                    o["bad"] = True
                    continue
                o["drawings"].add(ids.pop(0))
        out = []
        for o in objs:
            if o["bad"] or not o["chars"]:
                continue
            frame = o["form"]["rect"]
            heights = [c["bbox"][3] - c["bbox"][1] for c in o["chars"]]
            if (len(o["chars"]) + len(o["drawings"]) > FORMULA_GLYPHS_MAX or len(o["drawings"]) > FORMULA_MARKS_MAX
                    or frame.height > 6 * median(heights)):
                continue
            # a ruled table drawn as one picture (an Excel range): its grid crosses the frame
            if any(not (frame & t).is_empty and (any(frame.x0 + 1 < x < frame.x1 - 1 for x in xs)
                                                  or any(frame.y0 + 1 < y < frame.y1 - 1 for y in ys))
                   for t, xs, ys in self.table_lines):
                continue
            strong = self._maths_shown(o)
            if strong is None:
                continue
            o["strong"] = strong
            o["chars"].sort(key=lambda c: c["bbox"][0])
            out.append(o)
        # spaces the form draws, and the page's own spaces under its frame (Word sets the space after an object
        # inside the object's frame): they go with the formula, not into the text after it
        taken = {id(o["form"]): o for o in out}
        lines = {}  # the object's baseline and size: the page's spaces on that line only
        cells: dict[tuple[int, int], list[dict]] = {}  # (x // 50, baseline // 50) -> objects reaching there
        for o in out:
            sizes = [o["spans"][id(c)]["size"] for c in o["chars"]]
            top = max(sizes)
            lines[id(o)] = (median(c["origin"][1] for c, z in zip(o["chars"], sizes) if z > top - 0.5), top)
            r, y = o["form"]["rect"], lines[id(o)][0]
            for cx in range(int(r.x0 // 50), int(r.x1 // 50) + 1):
                for cy in range(int((y - top) // 50), int((y + top) // 50) + 1):
                    cells.setdefault((cx, cy), []).append(o)
        for ch, k in blanks:
            o = taken.get(id(frames[k])) if k is not None else None
            if o is None:
                x, y = (ch["bbox"][0] + ch["bbox"][2]) / 2, ch["origin"][1]
                o = next((o for o in cells.get((int(x // 50), int(y // 50)), ())
                          if o["form"]["rect"].x0 <= x <= o["form"]["rect"].x1
                          and abs(y - lines[id(o)][0]) <= 0.3 * lines[id(o)][1]), None)
            if o is not None:
                o["blanks"].append(ch)
        return out

    def _maths_shown(self, o: dict) -> bool | None:
        """How surely a form's content is maths: True for a Symbol / MT Extra glyph, a fraction bar (a
        horizontal stroke with a numerator right over it and a denominator right under it, no longer than
        they are) or a script (a smaller glyph off the baseline); False for italic letters in runs of at
        most three (variables) and no Vietnamese word; None otherwise (a line of ordinary text, a name, a
        date), and for any curve or filled shape (a diagram: MathType draws its bars and radicals as
        straight strokes, everything else as glyphs)."""
        spans = o["spans"]
        sizes = [spans[id(c)]["size"] for c in o["chars"]]
        for i in o["drawings"]:
            d = self.drawings[i]
            r = d["rect"]
            thin = r.height <= 1.5 or r.width <= 1.5
            if any(item[0] not in ("l", "re") for item in d["items"]) or (d.get("fill") is not None and not thin):
                return None
        if any(SYMBOL_FONT.search(spans[id(c)]["font"]) or MATH_FONT.search(spans[id(c)]["font"]) for c in o["chars"]):
            return True
        for i in o["drawings"]:
            r = self.drawings[i]["rect"]
            if r.height <= 1.5 and r.width >= 2:
                xs = [(c, spans[id(c)]["size"]) for c in o["chars"] if r.x0 - 1 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= r.x1 + 1]
                if not xs:
                    continue
                em = max(z for _c, z in xs)
                over = [c["bbox"] for c, z in xs if r.y0 - 1.5 * z <= c["bbox"][3] <= r.y0 + 1]
                under = [c["bbox"] for c, z in xs if r.y1 - 1 <= c["bbox"][1] <= r.y1 + 1.5 * z]
                if not over or not under:
                    continue
                for j in o["drawings"]:  # a radical's strokes over or under the bar widen that part
                    q = self.drawings[j]["rect"]
                    if j != i and r.x0 - 1 <= (q.x0 + q.x1) / 2 <= r.x1 + 1:
                        (over if q.y1 <= r.y0 + 1 else under if q.y0 >= r.y1 - 1 else []).append(tuple(q))
                if r.width <= max(max(b[2] for b in part) - min(b[0] for b in part) for part in (over, under)) + em:
                    return True
        top = max(sizes)
        base = median(c["origin"][1] for c, z in zip(o["chars"], sizes) if z > top - 0.5)
        if any(z < nativemath.SCRIPT * top and abs(c["origin"][1] - base) > 0.1 * top for c, z in zip(o["chars"], sizes)):
            return True
        italic, run, words, word = False, 0, [], ""
        prev = None
        for c in o["chars"]:  # left to right
            sp = spans[id(c)]
            near = prev is not None and c["bbox"][0] - prev["bbox"][2] <= 0.3 * sp["size"]
            letter = c["c"].isalpha()
            word = word + c["c"] if letter and near and word else c["c"] if letter else ""
            if word:
                words.append(word)
            if letter and c["c"].isascii() and _italic(sp):
                italic = True
                run = run + 1 if near and prev is not None and prev["c"].isalpha() else 1
                if run > 3:
                    return None
            prev = c
        if not italic or any(len(w) >= 2 and not w.isascii() for w in words):
            return None
        return False

    def _equation_mt(self, f: dict, span_of: dict[int, dict]) -> list | None:
        """A MathType object (_mathtype_objects) as a Word equation (nativemath pieces), or None to keep
        its picture: its Times New Roman and Symbol glyphs read as math letters (_mathtype_char), its
        horizontal strokes as fraction bars, upright ones as absolute-value bars."""
        if self._tiny_inside(f["rect"]):
            return None
        glyphs, fonts = [], Counter()
        for c in f["chars"]:
            sp = span_of.get(id(c))
            if sp is None or sp["size"] < 4:
                return None
            ch = _mathtype_char(c["c"], sp, self._symbol_maps.get(_font_key(sp["font"])))
            if ch is None:
                return None
            if not SYMBOL_FONT.search(sp["font"]):
                fonts[_family(sp["font"]) or ""] += 1
            x0, y0, x1, y1 = c["bbox"]
            glyphs.append(nativemath.Glyph(
                ch, x0, y0, x1, y1, c["origin"][1], sp["size"], font=_family(sp["font"]) or "", bold=_bold(sp),
                italic=_italic(sp), color=_color(sp, honour_alpha=not self.invisible_layer),
            ))
        size = max(g.size for g in glyphs if g.c not in "()[]{}") if any(g.c not in "()[]{}" for g in glyphs) else 0
        # an answer label typed into the object ("B. x = 4") stays text before the equation (nativemath splits it)
        order = sorted(range(len(glyphs)), key=lambda i: glyphs[i].x0)[:3]
        text = [f["chars"][i]["c"] for i in order]
        first = [glyphs[i] for i in order]
        if (len(first) == 3 and text[0] in "ABCDabcd" and text[1] in ".)" and not first[0].italic
                and first[2].x0 - first[1].x1 >= 0.2 * size and abs(first[1].oy - first[0].oy) <= 0.1 * size):
            for g, ch in zip(first[:2], text):
                g.c, g.math_font = ch, False
        bars, uprights, others = [], [], 0
        for i in f["drawings"]:
            r = pymupdf.Rect(self.drawings[i]["rect"])
            if r.height <= 1.5 and r.width >= 2:  # MathType's fraction bar: a stroked line
                bars.append(nativemath.Bar(r.x0, r.x1, (r.y0 + r.y1) / 2, nativemath.FRACTION_RULE * size))
            elif r.width <= 1.5 and r.height > 0:
                uprights.append((r.x0, r.y0, r.x1, r.y1))
            else:
                others += 1
        font = fonts.most_common(1)[0][0] if fonts and fonts.most_common(1)[0][0] else "Times New Roman"
        pieces = nativemath.build(glyphs, bars, uprights, others, mathtype=True, text_font=font)
        if pieces and not self._brackets_fit(glyphs, f):
            return None
        return pieces

    def _tiny_inside(self, rect: pymupdf.Rect) -> bool:
        """Glyphs too small to be read lie in the formula: its picture shows them."""
        ink = pymupdf.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
        if not len(self._tiny):
            return False
        if not isinstance(self._tiny, np.ndarray):
            self._tiny = np.asarray([tuple(r) for r in self._tiny])
        t = self._tiny
        return bool(np.any((t[:, 0] < ink.x1) & (t[:, 2] > ink.x0) & (t[:, 1] < ink.y1) & (t[:, 3] > ink.y0)))

    def _brackets_fit(self, glyphs: list, f: dict) -> bool:
        """No bracket is drawn taller than everything the formula holds (one that is encloses rows outside
        it: a matrix, a system). The text layer gives a stretched bracket the box of a plain one, so its
        ink is measured."""
        content = [g for g in glyphs if g.c not in "()[]{}"]
        if not content or len(content) == len(glyphs):
            return True
        top = min([g.oy - 0.78 * g.size for g in content] + [self.drawings[i]["rect"].y0 for i in f["drawings"]])
        bottom = max([g.oy + 0.22 * g.size for g in content] + [self.drawings[i]["rect"].y1 for i in f["drawings"]])
        for g in glyphs:
            if g.c in "()[]{}":
                height = self._ink_height(g)
                if height is None or height > bottom - top + 0.5 * g.size:
                    return False
        return True

    def _equation(self, f: dict, span_of: dict[int, dict], fraction_bars: set[int]) -> list | None:
        """The formula as a Word equation (nativemath pieces), or None to keep its picture."""
        marks = sum(c["c"] in nativemath.COMBINING for c in f["chars"])
        if len(f["chars"]) + len(f["drawings"]) > FORMULA_GLYPHS_MAX or marks > FORMULA_MARKS_MAX:
            return None
        if self._math_fonts is False:
            self._math_fonts = _math_fonts(self.page)
        if self._math_fonts is None:
            return None
        unmapped, font_names = self._math_fonts
        ink = pymupdf.Rect(f["rect"].x0 - 1, f["rect"].y0 - 1, f["rect"].x1 + 1, f["rect"].y1 + 1)
        if self._tiny_inside(f["rect"]):
            return None
        glyphs = []
        for c in f["chars"]:
            s = span_of.get(id(c))
            if s is None or SYMBOL_FONT.search(s["font"]) or (MATH_FONT.search(s["font"]) and not UNICODE_MATH_FONT.search(s["font"])):
                return None  # Symbol / MathType glyphs: their text may not be what is drawn
            unicode_math = bool(UNICODE_MATH_FONT.search(s["font"]))
            # MathType glyphs without real text ("!\"#") stay a picture; Cambria Math's ") + (" is real
            if _opaque(s["text"]) and not unicode_math:
                return None
            if unicode_math and (c["c"] in unmapped or _font_key(s["font"]) not in font_names):
                return None  # text the font's map does not give, or a font the page does not list
            x0, y0, x1, y1 = c["bbox"]
            stretched = False
            if c["c"] in nativemath.COMBINING:  # the text layer puts a mark at the end of the glyph before it
                drawn = self._drawn_mark(c, ink, f)
                if drawn is None:
                    return None
                if drawn[2] - drawn[0] >= 0.5:  # drawn with a width (a stretched hat or arrow): where it is drawn
                    x0, y0, x1, y1 = drawn
                    stretched = True
            glyphs.append(nativemath.Glyph(
                c["c"], x0, y0, x1, y1, c["origin"][1], s["size"], font=_family(s["font"]) or "",
                bold=_bold(s), italic=_italic(s), color=_color(s, honour_alpha=not self.invisible_layer),
                math_font=unicode_math, drawn=stretched,
            ))
        bars, uprights, others = [], [], 0
        for i in f["drawings"]:
            r = pymupdf.Rect(self.drawings[i]["rect"])
            if i in fraction_bars:
                bars.append(nativemath.Bar(r.x0, r.x1, (r.y0 + r.y1) / 2, r.height))
            elif r.width <= 1 and r.height > 0:
                uprights.append((r.x0, r.y0, r.x1, r.y1))
            else:
                others += 1
        pieces = nativemath.build(glyphs, bars, uprights, others)
        if pieces and not self._brackets_fit(glyphs, f):
            return None
        return pieces

    def _ink_height(self, g) -> float | None:
        """Height of a bracket's ink: the stroke through its middle, within its column, on the page
        rendered once at 150 dpi; dark is relative to the glyph's colour (a light green bracket too).
        None: not measured (rotated or very large page, a bracket in a very light colour)."""
        s = 150 / 72
        if self._page_ink is None:
            page = self.page
            if page.rotation or abs(page.rect) > PAGE_INK_MAX:
                self._page_ink = False
            else:
                try:
                    pix = page.get_pixmap(dpi=150, alpha=False)  # RGB: a grey render of an image page is slow
                    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
                    self._page_ink = ((rgb[:, :, 0].astype(np.uint16) * 77 + rgb[:, :, 1].astype(np.uint16) * 150
                                       + rgb[:, :, 2].astype(np.uint16) * 29) >> 8).astype(np.uint8)
                except Exception:  # noqa: BLE001
                    self._page_ink = False
        if self._page_ink is False:
            return None
        color = int(g.color or "000000", 16)
        lum = 0.299 * (color >> 16) + 0.587 * (color >> 8 & 255) + 0.114 * (color & 255)
        if lum > 215:
            return None
        img = self._page_ink
        x0, x1 = max(int((g.x0 - 0.3) * s), 0), min(int((g.x1 + 0.3) * s) + 1, img.shape[1])
        y0, y1 = max(int((g.oy - 3.5 * g.size) * s), 0), min(int((g.oy + 2.5 * g.size) * s) + 1, img.shape[0])
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        count, labels = cv2.connectedComponents((img[y0:y1, x0:x1] < (lum + 255) / 2).astype(np.uint8), connectivity=8)
        mid = int((g.oy - 0.25 * g.size) * s) - y0  # the maths axis: a bracket's middle
        hits = set(np.unique(labels[max(mid - 2, 0):max(mid + 3, 0), :])) - {0}
        if not hits:
            return None
        rows = np.flatnonzero(np.isin(labels, list(hits)).any(axis=1))
        return (rows[-1] - rows[0] + 1) / s

    def _drawn_mark(self, char: dict, ink: pymupdf.Rect, f: dict) -> tuple | None:
        """Where a combining mark of the formula is drawn (get_texttrace: the text layer reports it
        at the end of the glyph before it, a hat over "ABC" too). The formula's marks of one kind are
        matched to the drawn ones in order across, once per kind; None when they do not pair up."""
        if self._marks is None:
            self._marks = []
            try:
                # the trace holds every glyph drawn, off the page too: only for pages of ordinary text
                texts = [b for t, b in self.page.get_bboxlog() if t.endswith("-text")]
                area = pymupdf.Rect(self.page.rect).normalize()
                area = pymupdf.Rect(area.x0 - 5, area.y0 - 5, area.x1 + 5, area.y1 + 5)
                if len(texts) > TRACE_TEXTS_MAX or any(not area.contains(pymupdf.Point(b[0], b[1])) for b in texts):
                    return None
                for span in self.page.get_texttrace():
                    for code, _glyph, _origin, bbox in span["chars"]:
                        if code > 0 and chr(code) in nativemath.COMBINING:
                            self._marks.append((chr(code), tuple(bbox)))
            except Exception:  # noqa: BLE001
                return None
        ch = char["c"]
        paired = f.setdefault("_marks", {})
        if ch not in paired:
            drawn = sorted((r for c, r in self._marks if c == ch
                            and ink.x0 <= (r[0] + r[2]) / 2 <= ink.x1 and ink.y0 <= (r[1] + r[3]) / 2 <= ink.y1),
                           key=lambda r: r[0])
            mine = sorted((c for c in f["chars"] if c["c"] == ch), key=lambda c: c["bbox"][0])
            paired[ch] = {id(c): r for c, r in zip(mine, drawn)} if len(drawn) == len(mine) else None
        return paired[ch] and paired[ch].get(id(char))

    def _equation_runs(self, f: dict, main: float, baseline: float) -> list[Run]:
        """Runs of a formula rebuilt as an equation: the equation and any label set before or
        after it ("b)", a trailing dot in the text font)."""
        runs = []
        for piece in f["pieces"]:
            if piece.kind == "math":
                size = piece.size or main
                if f.get("mt") and abs(size - main) <= 0.1 * main:
                    size = main  # Word scales a MathType object a little (12.27 pt glyphs on a 12 pt line)
                run = Run(text=OBJ, size=_half(size), color=piece.color, font=piece.font, math=piece.text,
                          math_text=piece.text)
                f["run"] = run
                f["display"], f["frac_only"] = piece.display, piece.frac_only  # settled with its paragraph
                # the box Word's equation fills: glyph ink (about 0.7 em over the baseline, 0.2 under) and bars
                f["box"] = f["ink"]
            else:
                run = Run(text=piece.text, size=_half(piece.size), bold=piece.bold, italic=piece.italic,
                          color=piece.color, font=piece.font)
            runs.append(run)
        # kept for _settle_equations, which may still turn it into its picture (as _runs would have)
        f["piece_runs"], f["line_main"], f["line_baseline"] = list(runs), main, baseline  # `runs` grows into its segment
        return runs

    def _close_formulas(self, found: list[dict], marks: list[tuple]) -> list[dict]:
        """A formula's picture shows everything inside its box, so formulas whose
        boxes overlap become one, and glyphs and strokes lying inside a box join
        it (a denominator or an index set a little apart) instead of being
        written a second time beside the picture."""
        taken = {id(c) for f in found for c in f["chars"]}
        for _ in range(5):
            changed = False
            merged: list[dict] = []
            for f in sorted(found, key=lambda f: -abs(f["rect"])):
                host = next((g for g in merged if _overlaps(g["rect"], f["rect"])), None)
                if host is None:
                    merged.append(f)
                    continue
                host["rect"] |= f["rect"]
                host["chars"] += f["chars"]
                host["drawings"] |= f["drawings"]
                changed = True
            found = merged
            for f in found:
                r = f["rect"]
                for rect, _size, c, _node in marks:
                    if id(c) in taken or not _mostly_inside(rect, r):
                        continue
                    f["chars"].append(c)
                    taken.add(id(c))
                    f["rect"] |= rect
                    changed = True
                for i, d in enumerate(self.drawings):
                    if i in f["drawings"] or i in self.used_drawings:
                        continue
                    dr = pymupdf.Rect(d["rect"])
                    if r.x0 - 1 <= dr.x0 and dr.x1 <= r.x1 + 1 and r.y0 - 1 <= dr.y0 and dr.y1 <= r.y1 + 1:
                        f["drawings"].add(i)
                        changed = True
            if not changed:
                break
        return found

    def _in_formula(self, char: dict) -> bool:
        """A glyph drawn by a formula picture."""
        return id(char) in self.formula_chars

    def _formula_hosts(self, blocks: list[dict]) -> dict[int, list[dict]]:
        """The line that carries each formula: a line with text outside the
        formulas whose baseline lies inside the formula, nearest the baseline
        of its largest glyphs and horizontally close; else the
        line holding most of the formula's glyphs (a formula on its own line)."""
        lines = []
        for b in blocks:
            if b["type"] != 0:
                continue
            for ln in b["lines"]:
                dx, dy = ln["dir"]
                visible = [s for s in ln["spans"] if s["text"].strip()]
                if abs(dy) > 0.02 or dx < 0 or not visible:
                    continue
                outside = sum(1 for s in visible for c in s["chars"] if c["c"].strip() and not self._in_formula(c))
                baseline = median(s["origin"][1] for s in visible)
                size = max(s["size"] for s in visible)
                lines.append((ln, baseline, size, outside))
        hosts: dict[int, list[dict]] = {}
        for f in self.formulas:
            r = f["rect"]
            target = f["baseline"]
            best, best_key = None, None
            for ln, baseline, size, outside in lines:
                x0, x1 = ln["bbox"][0], ln["bbox"][2]
                if not outside or not r.y0 - 0.5 <= baseline <= r.y1 + 0.5:
                    continue
                if x1 < r.x0 - 3 * size or x0 > r.x1 + 3 * size:
                    continue
                key = (abs(baseline - target), max(x0 - r.x1, r.x0 - x1, 0))
                if best_key is None or key < best_key:
                    best, best_key = ln, key
            if best is None:  # alone on its line: the line with most of its glyphs
                count = lambda ln: sum(1 for s in ln["spans"] for c in s["chars"] if c["c"].strip() and r.contains(pymupdf.Point((c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2)))
                best = max((x[0] for x in lines), key=count, default=None)
                if best is None or not count(best):
                    continue
            hosts.setdefault(id(best), []).append(f)
        return hosts

    def _place_formulas(self, spans: list[dict], hosted) -> list[dict]:
        """Drop the glyphs inside formulas from a line's spans and insert the
        formulas this line carries as one-character picture spans."""
        out = []
        for s in spans:
            chars = s.get("chars") or []
            if not chars or not any(self._in_formula(c) for c in chars):
                out.append(s)
                continue
            keep: list[list[dict]] = [[]]
            for c in chars:
                if self._in_formula(c):
                    if keep[-1]:
                        keep.append([])
                else:
                    keep[-1].append(c)
            for part in keep:
                if not part:
                    continue
                piece = dict(s)
                piece["chars"] = part
                piece["text"] = "".join(c["c"] for c in part)
                piece["bbox"] = (part[0]["bbox"][0], s["bbox"][1], part[-1]["bbox"][2], s["bbox"][3])
                piece["origin"] = part[0]["origin"]
                out.append(piece)
        visible = [s for s in out if s["text"].strip()]
        base_y = median(s["origin"][1] for s in visible) if visible else None
        for f in hosted:
            r = pymupdf.Rect(f["rect"])
            r.x0 -= nativemath.lead_width(f["pieces"] or [])  # the line's natural width must hold it
            y = base_y if base_y is not None else f["baseline"]
            out.append({
                "text": OBJ, "chars": [{"c": OBJ, "bbox": tuple(r), "origin": (r.x0, y)}], "bbox": tuple(r),
                "origin": (r.x0, y), "size": f["size"], "font": "", "flags": 0, "color": 0, "alpha": 255,
                "formula": f,
            })
        return sorted(out, key=lambda s: s["bbox"][0])

    def _form_picture(self, f: dict) -> tuple[pymupdf.Rect, bytes]:
        """A MathType object's picture: its form drawn alone on a clear ground (strokes and all, none of
        the page behind it), across its glyph boxes as _formula_picture, up and down to its ink."""
        fr = f["form"]
        self._forms_doc.update_stream(
            self._forms_xref, "q {} cm /{} Do Q\n".format(" ".join(f"{v:.6f}" for v in fr["ctm"]), fr["name"]).encode())
        page = self._forms_doc[0]
        rect = f["rect"]
        clip = pymupdf.Rect(rect.x0, rect.y0 - 0.6, rect.x1, rect.y1 + 0.6) & page.rect
        s = FORMULA_DPI / 72
        pix = page.get_pixmap(dpi=FORMULA_DPI, clip=clip, alpha=True, annots=False)
        if not pix.h or not pix.w:
            _ok, buf = cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))
            return pymupdf.Rect(rect), buf.tobytes()
        rgba = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        rows = np.flatnonzero(rgba[:, :, 3].max(axis=1) > 0)
        top, bottom = (max(rows[0] - int(s), 0), min(rows[-1] + 1 + int(s), pix.h)) if rows.size else (0, pix.h)
        _ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgba[top:bottom]), cv2.COLOR_RGBA2BGRA))
        return pymupdf.Rect(clip.x0, clip.y0 + top / s, clip.x1, clip.y0 + bottom / s), buf.tobytes()

    def _formula_picture(self, f: dict) -> tuple[pymupdf.Rect, bytes]:
        """The formula's picture and the box it shows. It holds the formula's own
        glyphs and strokes on a clear ground: no watermark or shading from behind
        (the page drawn without its text and formula strokes is taken away), no
        bits of the lines around it. Across, it spans exactly the glyph boxes
        (their advances carry the spacing to the text around it, and any extra
        width moves Word's line breaks); up and down, only the ink – glyph boxes
        reach the font's whole ascent and descent and would make the line taller."""
        if f.get("form") is not None:
            try:
                return self._form_picture(f)
            except Exception:  # noqa: BLE001 – drawn from the page instead
                f = {**f, "form": None}
        rect = f["rect"]
        clip = pymupdf.Rect(rect.x0, rect.y0 - 0.6, rect.x1, rect.y1 + 0.6) & self.page.rect
        full = _rgb(self.page.get_pixmap(dpi=FORMULA_DPI, clip=clip, alpha=False))
        ground = _rgb(self._ground_page().get_pixmap(dpi=FORMULA_DPI, clip=clip, alpha=False))
        h, w = full.shape[:2]
        if not h or not w:  # nothing to show (a box off the page, or a /Rotate page: its text frame is not the render's):
            # an empty picture – an empty array crashes cv2 and with it the worker
            ok, buf = cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))
            return pymupdf.Rect(rect), buf.tobytes()
        if ground.shape != full.shape:  # rounding of the clip: never expected
            ground = np.full_like(full, 255)
        ink = np.abs(full.astype(np.int16) - ground).max(axis=2) > 10
        # rotated watermark text is text too: its pale paint is not the formula's ink
        ink &= ~_paint_mask(full, clip, FORMULA_DPI, self.rotated_paints())
        own = np.zeros((h, w), dtype=bool)
        s = FORMULA_DPI / 72
        for r, padx, pady in f["parts"]:
            x0, x1 = max(int((r.x0 - padx - clip.x0) * s), 0), min(int((r.x1 + padx - clip.x0) * s) + 1, w)
            y0, y1 = max(int((r.y0 - pady - clip.y0) * s), 0), min(int((r.y1 + pady - clip.y0) * s) + 1, h)
            own[y0:y1, x0:x1] = True
        shown = ink & own
        # specks: pieces of ink without a single strong pixel (overlapping strokes of a pale
        # watermark) are not part of the formula
        count, labels = cv2.connectedComponents(shown.astype(np.uint8), connectivity=8)
        strong = np.abs(full.astype(np.int16) - ground).max(axis=2) >= 80
        keep = np.zeros(count, dtype=bool)
        keep[np.unique(labels[shown & strong])] = True
        keep[0] = False
        shown = keep[labels]
        rows = np.flatnonzero(shown.any(axis=1))
        top, bottom = (max(rows[0] - int(s), 0), min(rows[-1] + 1 + int(s), h)) if rows.size else (0, h)
        rgba = np.dstack([full, np.where(shown, 255, 0).astype(np.uint8)])[top:bottom]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        box = pymupdf.Rect(clip.x0, clip.y0 + top / s, clip.x1, clip.y0 + bottom / s)
        return box, buf.tobytes()

    # --- tables -----------------------------------------------------------

    def tables(self) -> list[tuple[pymupdf.Rect, list[float], list[float], list]]:
        """Ruled tables: grids read from the page's rules first (PyMuPDF misses a
        one-row answer key and turns a true/false grid with merged "Câu N" cells
        into one bogus 1×3 table), then PyMuPDF's finds elsewhere."""
        words = [pymupdf.Rect(w[:4]) for w in self.page.get_text("words")]
        out = _ruled_grids(self.drawings, words)
        try:
            found = self.page.find_tables().tables
        except Exception:
            found = []
        for t in found:
            if t.row_count * t.col_count < 2:
                continue
            xs = _cluster_values([c[i] for row in t.rows for c in row.cells if c for i in (0, 2)], 2.0)
            # Row tops, then the lowest bottom: a row holding a cell merged down
            # ("Câu 1" over five rows) reports that cell's bottom as its own.
            ys = _cluster_values([r.bbox[1] for r in t.rows] + [max(r.bbox[3] for r in t.rows)], 1.0)
            if len(xs) >= 2 and len(ys) >= 2:
                rect = self._ruled_extent(pymupdf.Rect(t.bbox))
                if any(abs(rect & g[0]) > 0.3 * min(abs(rect), abs(g[0])) for g in out):
                    continue
                xs[0], xs[-1] = min(xs[0], rect.x0), max(xs[-1], rect.x1)
                ys[0], ys[-1] = min(ys[0], rect.y0), max(ys[-1], rect.y1)
                out.append((rect, xs, ys, _merged_cells(t, xs, ys)))
        return out

    def _ruled_extent(self, rect: pymupdf.Rect) -> pymupdf.Rect:
        """A ruled table out to its frame: PyMuPDF may cut the outer cells at the
        text (63–532pt drawn, 76–339pt found), and the rest of the rules and
        shading would come out again as a picture of the table. The row rules
        inside the table give its width, the column rules its height."""
        out = pymupdf.Rect(rect)
        for d in self.drawings:
            x0, y0, x1, y1 = d["rect"]
            if y1 - y0 <= 1.5 and x1 - x0 > 1 and rect.y0 + 1 < y0 < rect.y1 - 1 and x0 < rect.x1 and x1 > rect.x0:
                out.x0, out.x1 = min(out.x0, x0), max(out.x1, x1)
            elif x1 - x0 <= 1.5 and y1 - y0 > 1 and rect.x0 + 1 < x0 < rect.x1 - 1 and y0 < rect.y1 and y1 > rect.y0:
                out.y0, out.y1 = min(out.y0, y0), max(out.y1, y1)
        return out

    def _cell_shading(self, x0: float, y0: float, x1: float, y1: float) -> str | None:
        """Fill colour of a ruled table's cell: the coloured box covering most
        of it (the last one drawn when several do)."""
        area = max((x1 - x0) * (y1 - y0), 1e-6)
        colour = None
        for d in self.drawings:
            if not _highlight_paint(d):
                continue
            r = d["rect"]
            w, h = min(r.x1, x1) - max(r.x0, x0), min(r.y1, y1) - max(r.y0, y0)
            if w > 0 and h > 0 and w * h >= 0.6 * area:
                colour = _hex(d["fill"])
        return colour

    def _table_element(self, rect, xs, ys, merges, rows: list[_Row]) -> Table:
        nrows, ncols = len(ys) - 1, len(xs) - 1
        content: list[list[list[tuple[float, float, float, list[Run]]]]] = [
            [[] for _ in range(ncols)] for _ in range(nrows)
        ]
        # A merged cell's text (centred over its rows: "Câu 1") goes to its first grid cell.
        owner, extent = {}, {}
        for r0, c0, rs, cs in merges:
            extent[(r0, c0)] = (r0 + rs, c0 + cs)
            for rr in range(r0, r0 + rs):
                for cc in range(c0, c0 + cs):
                    owner[(rr, cc)] = (r0, c0)
        for row in rows:
            for seg in row.segs:
                cx = (seg.x0 + seg.x1) / 2
                r = max(0, min(nrows - 1, sum(y <= row.baseline for y in ys) - 1))
                c = max(0, min(ncols - 1, sum(x <= cx for x in xs) - 1))
                r, c = owner.get((r, c), (r, c))
                content[r][c].append((row.baseline, seg.x0, seg.x1, seg.runs))
        cells = []
        sizes = [run.size for row in rows for seg in row.segs for run in seg.runs if run.size]
        pads: list[float] = []  # gap from the left rule to left-aligned text
        for r in range(nrows):
            line = []
            for c in range(ncols):
                runs: list[Run] = []
                spans: list[list[float]] = []  # [x0, x1] of each text line in the cell
                last_baseline = None
                for baseline, x0, x1, seg_runs in sorted(content[r][c], key=lambda t: t[:2]):
                    same_line = last_baseline is not None and baseline - last_baseline <= 2
                    if last_baseline is not None:
                        runs.append(Run(" " if same_line else LINE_BREAK))
                    runs.extend(seg_runs)
                    if same_line:
                        spans[-1][1] = max(spans[-1][1], x1)
                    else:
                        spans.append([x0, x1])
                    last_baseline = baseline
                if runs:
                    runs[-1].text = runs[-1].text.rstrip()
                r1, c1 = extent.get((r, c), (r + 1, c + 1))
                align = cell_align(spans, xs[c], xs[c1])
                if spans and align is None:
                    pads.append(min(a for a, _ in spans) - xs[c])
                line.append(
                    Cell(
                        text="".join(x.text for x in runs), runs=runs,
                        shading=self._cell_shading(xs[c], ys[r], xs[c1], ys[r1]),
                        align=align,
                    )
                )
            cells.append(line)
        return Table(
            cells=cells,
            col_widths_pt=[b - a for a, b in pairwise(xs)],
            row_heights_pt=[b - a for a, b in pairwise(ys)],
            x0=rect.x0,
            y0=rect.y0,
            x1=rect.x1,
            y1=rect.y1,
            font_pt=median(sizes) if sizes else 11.0,
            merges=merges,
            # Word's default 5.4pt would break words the PDF's narrow columns fit ("Đúng").
            cell_margin_pt=max(0.5, min(5.4, min(pads) - 0.5)) if pads else 1.5,
        )

    # --- figures ------------------------------------------------------------

    def _unmarked_page(self, hidden: list[pymupdf.Rect], figures: list[pymupdf.Rect]) -> pymupdf.Page:
        """This page without the watermark images (raster regions `hidden`; a
        figure with a transparent ground shows them through). A watermark may
        span the whole page, so an image inside one is still kept when it is a
        figure itself."""
        if self._unmarked_doc is None:
            inside = lambda r, box: (  # noqa: E731
                box.x0 - 1 <= r.x0 and r.x1 <= box.x1 + 1 and box.y0 - 1 <= r.y0 and r.y1 <= box.y1 + 1
            )
            doc = pymupdf.open()
            doc.insert_pdf(self.page.parent, from_page=self.page.number, to_page=self.page.number)
            p = doc[0]
            xrefs = {
                info["xref"] for info in p.get_image_info(xrefs=True)
                if info["xref"]
                and any(inside(pymupdf.Rect(info["bbox"]), h) for h in hidden)
                and not any(inside(pymupdf.Rect(info["bbox"]), f) for f in figures)
            }
            for xref in xrefs:
                p.delete_image(xref)
            self._unmarked_doc = doc
        return self._unmarked_doc[0]

    def _ground_page(self) -> pymupdf.Page:
        """This page with its text and formula strokes removed: what lies
        behind a formula picture."""
        if self._ground_doc is None:
            self._ground_doc = self._without_text()
        return self._ground_doc[0]

    def _stripped_page(self) -> pymupdf.Page:
        """The ground with rotated text (a watermark) drawn back, for rendering
        backgrounds."""
        if self._stripped_doc is None:
            doc = self._without_text()
            for line in self.rotated_lines:
                self._reset_rotated(doc[0], line)
            self._stripped_doc = doc
        return self._stripped_doc[0]

    def _without_text(self) -> pymupdf.Document:
        doc = pymupdf.open()
        doc.insert_pdf(self.page.parent, from_page=self.page.number, to_page=self.page.number)
        p = doc[0]
        p.add_redact_annot(p.rect, fill=False, cross_out=False)
        p.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )
        for i in self.formula_drawings:
            r = pymupdf.Rect(self.drawings[i]["rect"])
            p.add_redact_annot(pymupdf.Rect(r.x0 - 0.5, r.y0 - 0.5, r.x1 + 0.5, r.y1 + 0.5), fill=False, cross_out=False)
        if self.formula_drawings:
            p.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                text=pymupdf.PDF_REDACT_TEXT_NONE,
            )
        return doc

    @staticmethod
    def _reset_rotated(page: pymupdf.Page, line: dict) -> None:
        """Draw rotated text (a watermark) back onto the text-free page so it
        is kept in the background picture. The original font is often an
        unnamed Type3 font, so a serif font is stretched to the same length."""
        chars = [c for s in line["spans"] for c in s["chars"]]
        text = "".join(c["c"] for c in chars).strip()
        span = max(line["spans"], key=lambda s: len(s["chars"]))
        font = _serif_font()
        natural = font.text_length(text, span["size"])
        if not text or not natural:
            return
        start = pymupdf.Point(chars[0]["origin"])
        end = pymupdf.Point(chars[-1]["origin"])
        length = abs(end - start) + font.text_length(chars[-1]["c"], span["size"])
        dx, dy = line["dir"]
        rgb = span["color"] & 0xFFFFFF
        color = tuple(((rgb >> sh) & 255) / 255 for sh in (16, 8, 0))
        writer = pymupdf.TextWriter(page.rect, opacity=span.get("alpha", 255) / 255, color=color)
        writer.append(start, text, font=font, fontsize=span["size"])
        # Stretch to the original length, then turn along the line direction
        # (morph works in PDF space, where y points up).
        matrix = pymupdf.Matrix(length / natural, 0, 0, 1, 0, 0) * pymupdf.Matrix(dx, -dy, dy, dx, 0, 0)
        writer.write_text(page, morph=(start, matrix))

    def raster_regions(self) -> list[pymupdf.Rect]:
        page_area = self.width * self.height
        out = []
        for info in self.page.get_image_info():
            r = pymupdf.Rect(info["bbox"]) & self.page.rect
            if r.is_empty or abs(r) < 16:
                continue
            if self.invisible_layer and abs(r) > 0.8 * page_area:
                continue  # the scan under a searchable PDF's text layer
            out.append(r)
        return _join_strips(out)

    def vector_regions(self, table_rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
        page_area = self.width * self.height
        rects = []
        for i, d in enumerate(self.drawings):
            if i in self.used_drawings:
                continue
            r = pymupdf.Rect(d["rect"])
            if r.width * r.height > 0.85 * page_area:
                continue  # page frames and backgrounds
            if any(pymupdf.Rect(t.x0 - 2, t.y0 - 2, t.x1 + 2, t.y1 + 2).contains(r) for t in table_rects):
                continue  # table rules
            fill, stroke = d.get("fill"), d.get("color")
            if fill is not None and tuple(fill) == (1.0, 1.0, 1.0) and not stroke:
                continue  # white boxes
            rects.append(r)
        return _merge_boxes(rects, gap=6)

    def _classify(self, region: pymupdf.Rect, rows: list[_Row]) -> tuple[str, list[int]]:
        inside, crossing, grazing, text_area = [], 0, 0, 0.0
        # the last short line of a paragraph ("thể khí." over a footer rule) is text, not a label
        prose = {row.block for row in rows if len(row.text.strip()) > 12}
        for i, row in enumerate(rows):
            rect = row.rect
            overlap = rect & region
            # A short label set just outside the drawing or across its edge (axis
            # names "s (km)", ticks "O", "24 5") belongs to the figure; above or
            # below it, only within its width ("PHẦN I." over a table is a heading).
            m = 1.2 * row.size
            # a rule (the line over a footer) only takes what touches it: the page number
            # beside it, not the last line of text above it
            mv = m if region.height >= 3 else 0.3 * row.size
            near = not (rect & pymupdf.Rect(region.x0 - m, region.y0 - mv, region.x1 + m, region.y1 + mv)).is_empty and (
                rect.y1 > region.y0 and rect.y0 < region.y1 or region.x0 - m <= rect.x0 and rect.x1 <= region.x1 + m
            )
            # (a formula picture is one character of text: its width tells a label "𝑡(𝑠)"
            # from a line of working "0,9 = … → 𝑡 = 28,8 ℃" beside a figure)
            label = len(row.text.strip()) <= 12 and len(row.segs) <= 3 and row.block not in prose and (
                OBJ not in row.text or row.x1 - row.x0 <= 6 * row.size
            )
            if overlap.is_empty and not (near and label):
                continue
            if (not overlap.is_empty and abs(overlap) >= 0.8 * max(abs(rect), 1e-6)) or (near and label):
                inside.append(i)
                # Only sentences make a drawing a background under text; short
                # labels ("Hình 1", letters filling in a line of image strips) do not.
                if len(row.text.strip()) > 12:
                    text_area += abs(rect)
            elif overlap.height >= 0.5 * rect.height:
                crossing += 1
            else:
                grazing += 1
        # A line only grazing the top or bottom edge (its line box reaching into a
        # graph's tick labels) does not cross a drawing that holds just labels; with
        # sentences inside, the drawing is a background under the text.
        if any(len(rows[i].text.strip()) > 12 for i in inside):
            crossing += grazing
        area = max(abs(region), 1e-6)
        small = area < 0.5 * self.width * self.height
        if crossing == 0 and (not inside or (len(inside) <= 12 and text_area < 0.3 * area and small)):
            return "figure", inside
        return "background", []

    def _render(self, rect: pymupdf.Rect, background: bool, paints=None, hidden=(), figures=()) -> bytes:
        clip = _min_size(rect) & self.page.rect
        if background:
            pix = self._stripped_page().get_pixmap(dpi=BACKGROUND_DPI, clip=clip, alpha=True)
            return pix.tobytes("png")
        page = self._unmarked_page(hidden, figures) if hidden else self.page
        pix = page.get_pixmap(dpi=FIGURE_DPI, clip=clip, alpha=False)
        if not paints:
            return pix.tobytes("png")
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        img = img[:, : pix.width * 3].reshape(pix.height, pix.width, 3).copy()
        _clear_paint(img, clip, FIGURE_DPI, paints)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return buf.tobytes() if ok else pix.tobytes("png")

    def watermark_paints(self, backgrounds: list[pymupdf.Rect]) -> list[tuple[pymupdf.Rect, np.ndarray]]:
        """(box, colour on white paper) of each filled drawing inside a
        background region: a watermark that may show through figures."""
        paints = []
        for i, d in enumerate(self.drawings):
            fill, r = d.get("fill"), pymupdf.Rect(d["rect"])
            if not fill or i in self.used_drawings:
                continue
            if not any(not (r & b).is_empty for b in backgrounds):
                continue
            opacity = d.get("fill_opacity") or 1.0
            paints.append((r, 255 - opacity * (255 - 255 * np.array(fill[:3], dtype=float))))
        return paints

    def rotated_paints(self) -> list[tuple[pymupdf.Rect, np.ndarray]]:
        """(box, colour on white paper) of each rotated text line (a watermark)."""
        paints = []
        for line in self.rotated_lines:
            for span in line["spans"]:
                rgb = span["color"] & 0xFFFFFF
                color = np.array([(rgb >> sh) & 255 for sh in (16, 8, 0)], dtype=float)
                opacity = span.get("alpha", 255) / 255
                paints.append((pymupdf.Rect(line["bbox"]), 255 - opacity * (255 - color)))
        return paints

    def is_watermark(self, region: pymupdf.Rect, rotated: list[pymupdf.Rect]) -> bool:
        """A background that is a watermark: large, holding rotated text, or
        painted pale or translucent. Small dark line art behind the text
        (rules, bars) is not."""
        if abs(region) >= WATERMARK_MIN_AREA * self.width * self.height:
            return True
        if any(not (region & r).is_empty for r in rotated):
            return True
        for i, d in enumerate(self.drawings):
            if i in self.used_drawings or (pymupdf.Rect(d["rect"]) & region).is_empty:
                continue
            paint = d.get("fill") or d.get("color")
            opacity = d.get("fill_opacity") if d.get("fill") else d.get("stroke_opacity")
            if (opacity or 1.0) < 0.95 or (paint and min(paint[:3]) >= 0.7):
                return True
        return False

    # --- paragraphs ---------------------------------------------------------

    @staticmethod
    def _continues(prev: _Row, row: _Row, right: float, obstacles=()) -> bool:
        if len(prev.segs) > 1 or len(row.segs) > 1 or abs(prev.size - row.size) > 1:
            return False
        pitch = row.baseline - prev.baseline
        if pitch <= 0 or pitch > 1.7 * row.size or row.x0 > prev.x0 + 3:
            return False
        prev_text, text = prev.text.rstrip(), row.text.lstrip()
        if not text:
            return False
        # A question or answer label ("Câu 3.", "B.") always starts a paragraph.
        if LINE_LABEL.match(text):
            return False
        # A sentence end followed by a capital, or a bold lead-in ("Câu 2:"), starts anew.
        if prev_text.endswith((".", ":", "?", "!", ";")) and (text[0].isupper() or text[0].isdigit()):
            return False
        first_run = row.segs[0].runs[0]
        last_run = next((r for r in reversed(prev.segs[-1].runs) if r.text.strip()), first_run)
        if first_run.bold and not last_run.bold:
            return False
        # Same look where the lines meet is enough: a bold lead-in ("PHẦN I. …")
        # followed by plain text wraps into a plain line.
        junction = not first_run.bold and (last_run.bold, last_run.italic, last_run.color) == (
            first_run.bold, first_run.italic, first_run.color
        )
        if _dominant_style(prev) != _dominant_style(row) and not junction:
            return False
        # Wrapped only if the next word would not have fitted on the previous line;
        # a figure beside the line (text wrapped around it) ends the line early.
        for o in obstacles:
            if o.y0 < prev.y1 and o.y1 > prev.y0 and o.x0 >= prev.x1 - 1:
                right = min(right, o.x0 - 2)
        first_word = text.split()[0]
        char_w = (row.x1 - row.x0) / max(len(row.text), 1)
        return prev.x1 + (len(first_word) + 1) * char_w >= right - 2

    @staticmethod
    def _paragraph(rows: list[_Row]) -> Paragraph:
        runs: list[Run] = []
        for i, row in enumerate(rows):
            if i and runs and runs[-1].text.endswith(SOFT_HYPHEN):
                runs[-1].text = runs[-1].text[:-1]  # hyphenation point: rejoin the word
            elif i and runs and not runs[-1].text.endswith((" ", "-")):
                runs.append(Run(" ", size=_half(row.size)))
            for j, seg in enumerate(row.segs):
                if j:
                    if runs:
                        runs[-1].text = runs[-1].text.rstrip()
                    runs.append(Run(TAB))
                runs.extend(seg.runs)
        for r in runs:
            r.text = r.text.replace(SOFT_HYPHEN, "-")
        if runs:
            runs[-1].text = runs[-1].text.rstrip()
        visible = [r for r in runs if r.text.strip() and r.text != TAB]
        pitches = [b.baseline - a.baseline for a, b in pairwise(rows)]
        return Paragraph(
            text="".join(r.text for r in runs),
            x0=min(r.x0 for r in rows),
            y0=min(r.y0 for r in rows),
            x1=max(r.x1 for r in rows),
            y1=max(r.y1 for r in rows),
            font_pt=_half(median(r.size for r in rows)),
            runs=runs,
            tab_stops=[(seg.x0, "left") for seg in rows[0].segs[1:]] if len(rows) == 1 else [],
            bold=bool(visible) and all(r.bold for r in visible),
            line_pitch_pt=median(pitches) if pitches else None,
            natural_right=max(r.natural_right for r in rows),
            translucent=all(r.translucent for r in rows),
            baselines=(rows[0].baseline, rows[-1].baseline),
            line_spans=[(r.x0, r.x1) for r in rows],
        )

    def paragraphs(self, rows: list[_Row], right: float, obstacles=()) -> list[Paragraph]:
        """Group consecutive rows into paragraphs; `right` is the column's right edge,
        `obstacles` the figures text wraps around."""
        if not rows:
            return []
        paras, current = [], [rows[0]]
        for prev, row in pairwise(rows):
            if self._continues(prev, row, right, obstacles):
                current.append(row)
            else:
                paras.append(self._paragraph(current))
                current = [row]
        paras.append(self._paragraph(current))
        return paras


OPTION_LABEL = re.compile(r"[A-Ha-h][.)]")
LINE_LABEL = re.compile(r"(?:Câu|Bài|Question)\s*\d+|[A-H][.)]\s")
# A row that starts a question, an answer option or a true/false item ("c) …").
ITEM_LABEL = re.compile(r"(?:Câu|Bài|Question)\s*\d+|[A-Ha-h][.)]")


def _split_at_tabs(spans: list[dict]) -> list[dict]:
    """Split spans where Word put a tab, and mark the pieces that follow one
    (`tab_before`): a whitespace glyph much wider than the line's other spaces,
    or a jump between neighbouring glyphs – at least 1.5em, or 0.4em before an
    answer label ("C." after a long option B) – which may be narrower than the
    usual 1em jump between items."""
    spaces = [c["bbox"][2] - c["bbox"][0] for s in spans for c in s.get("chars") or [] if not c["c"].strip()]
    usual = median(spaces) if spaces else None
    out: list[dict] = []
    last_x1 = None  # right edge of the previous glyph on the line
    for s in spans:
        chars = s.get("chars") or []
        if not chars or "".join(c["c"] for c in chars) != s["text"]:
            out.append(s)
            last_x1 = s["bbox"][2]
            continue
        size = s["size"]
        cuts: list[tuple[int, bool]] = []  # (index, drop the char at index)
        for i, c in enumerate(chars):
            x0, x1 = c["bbox"][0], c["bbox"][2]
            if not c["c"].strip() and usual and x1 - x0 >= max(0.6 * size, 2.5 * usual):
                cuts.append((i, True))
            elif c["c"].strip() and last_x1 is not None:
                gap = x0 - last_x1
                rest = "".join(ch["c"] for ch in chars[i : i + 2])
                if gap >= 1.5 * size or (gap >= 0.4 * size and OPTION_LABEL.match(rest)):
                    cuts.append((i, False))
            last_x1 = x1
        if not cuts:
            out.append(s)
            continue
        bounds = [(0, False)] + cuts + [(len(chars), False)]
        for k, ((a, drop_a), (b, _)) in enumerate(pairwise(bounds)):
            part = chars[a + (1 if drop_a else 0) : b]
            if not part:
                continue
            piece = dict(s)
            piece["chars"] = part
            piece["text"] = "".join(c["c"] for c in part)
            piece["bbox"] = (part[0]["bbox"][0], s["bbox"][1], part[-1]["bbox"][2], s["bbox"][3])
            piece["origin"] = part[0]["origin"]
            piece["tab_before"] = k > 0 or s.get("tab_before", False)
            out.append(piece)
    return out


def _line_baseline(line: dict) -> tuple[float, float] | None:
    """(baseline, size) of a line's main text, or None for a blank or rotated line."""
    visible = [s for s in line["spans"] if s["text"].strip()]
    if not visible or abs(line["dir"][1]) > 0.02 or line["dir"][0] < 0:
        return None
    main = max(s["size"] for s in visible)
    return median(s["origin"][1] for s in visible if s["size"] > main - 0.5), main


def _join_interleaved(lines: list[dict]) -> list[dict]:
    """Lines of a block that run into each other on one baseline become one
    line, their spans in order across: Word's export puts a superscript and the
    text after it in a line of its own that starts inside the previous one
    ("J/kg" "C" + "⁰" ") được"). A blank span starting on a glyph of the other
    line is dropped; one running under a glyph is cut short at it (the printed
    space before the superscript, not a tab)."""
    out: list[dict] = []
    for line in lines:
        base = _line_baseline(line)
        host = None
        if base is not None:
            for q in out:
                qb = _line_baseline(q)
                if (
                    qb is not None
                    and abs(qb[0] - base[0]) <= 0.3 * max(qb[1], base[1])
                    and line["bbox"][0] < q["bbox"][2] - 1
                    and q["bbox"][0] < line["bbox"][2] - 1
                ):
                    host = q
                    break
        if host is None:
            out.append(line)
            continue
        spans = host["spans"] + line["spans"]
        boxes = [pymupdf.Rect(s["bbox"]) for s in spans if s["text"].strip()]
        spans = [
            s for s in spans
            if s["text"].strip() or not any(r.x0 - 0.5 <= s["bbox"][0] < r.x1 - 0.5 for r in boxes)
        ]
        for k, s in enumerate(spans):
            if s["text"].strip():
                continue
            end = min((r.x0 for r in boxes if s["bbox"][0] < r.x0 < s["bbox"][2]), default=None)
            if end is not None:
                cut = lambda bb: (bb[0], bb[1], min(bb[2], end), bb[3])  # noqa: E731
                spans[k] = {**s, "bbox": cut(s["bbox"]), "chars": [{**c, "bbox": cut(c["bbox"])} for c in s["chars"]]}
        host["spans"] = sorted(spans, key=lambda s: s["bbox"][0])
        host["bbox"] = tuple(pymupdf.Rect(host["bbox"]) | pymupdf.Rect(line["bbox"]))
    return out


def _add_row(q: _Row, r: _Row) -> None:
    """Put row r's items into row q (one printed line). Where an item of r
    touches one of q (text cut around a formula: "phần trăm (" + "%)?") the two
    become one item, so no tab comes between them; an answer label keeps its own."""
    near = max(0.9 * max(q.size, r.size), 8)
    segs = sorted([(s, 0) for s in q.segs] + [(s, 1) for s in r.segs], key=lambda p: p[0].x0)
    out: list[list] = []  # [seg, row its last piece came from]
    for seg, src in segs:
        if out and out[-1][1] != src and 0 <= seg.x0 - out[-1][0].x1 + 1 <= near + 1:
            text = "".join(run.text for run in seg.runs).lstrip()
            if not OPTION_LABEL.match(text[:2]):
                prev = out[-1][0]
                tail, head = prev.runs[-1], seg.runs[0]
                if seg.x0 - prev.x1 >= 0.2 * min(q.size, r.size) and tail.text[-1:] != " " and head.text[:1] != " ":
                    prev.runs.append(Run(" ", size=head.size if tail.image or tail.math else tail.size, font=head.font))
                prev.runs.extend(seg.runs)
                prev.chars.extend(seg.chars)
                prev.x1 = max(prev.x1, seg.x1)
                out[-1][1] = src
                continue
        out.append([seg, src])
    q.segs = [seg for seg, _ in out]
    q.x0, q.x1 = min(q.x0, r.x0), max(q.x1, r.x1)
    q.y0, q.y1 = min(q.y0, r.y0), max(q.y1, r.y1)


def _join_split_lines(rows: list[_Row]) -> list[_Row]:
    """One printed line that MuPDF cut into one-line blocks (around formulas,
    between answer options) becomes one row again: a row whose block has no
    other line joins a row on the same baseline beside it. Blocks of several
    lines (two-column letterheads) stay apart."""
    counts = Counter(r.block for r in rows)
    formula = lambda row: any(run.image or run.math for seg in row.segs for run in seg.runs)
    option = lambda row: bool(ITEM_LABEL.match(row.text.lstrip()))
    out = list(rows)
    changed = True
    while changed:
        changed = False
        for r in sorted(out, key=lambda x: x.x0):
            if counts[r.block] != 1:
                continue
            for q in out:
                if q is r or abs(q.baseline - r.baseline) > 0.3 * min(q.size, r.size):
                    continue
                # only where formulas or answer options cut the line (a two-part
                # letterhead of one-line blocks stays side by side)
                if not (formula(r) or formula(q) or option(r)):
                    continue
                if not (q.x1 <= r.x0 + 1 or r.x1 <= q.x0 + 1):
                    continue
                # beside a formula, a smaller or distant item is a figure's label ("(1)" by a graph)
                gap = max(r.x0 - q.x1, q.x0 - r.x1)
                if not option(r) and (gap > 4 * max(q.size, r.size) or min(q.size, r.size) < 0.8 * max(q.size, r.size)):
                    continue
                _add_row(q, r)
                q.translucent = q.translucent and r.translucent
                out.remove(r)
                changed = True
                break
            if changed:
                break
    return out


def _side_by_side(rows: list[_Row]) -> list[list[list[int]]]:
    """Blocks laid out next to each other. Each group is a list of columns,
    each column a list of block numbers."""
    boxes: dict[int, list[float]] = {}
    for r in rows:
        b = boxes.setdefault(r.block, [r.x0, r.y0, r.x1, r.y1, r.size])
        b[0], b[1], b[2], b[3] = min(b[0], r.x0), min(b[1], r.y0), max(b[2], r.x1), max(b[3], r.y1)
        b[4] = min(b[4], r.size)
    parent = {b: b for b in boxes}

    def find(b: int) -> int:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    ids = list(boxes)
    paired: set[int] = set()
    for i, a in enumerate(ids):
        ax0, ay0, ax1, ay1, asz = boxes[a]
        for b in ids[i + 1 :]:
            bx0, by0, bx1, by1, bsz = boxes[b]
            # Side by side only when the blocks share lines (overlap at least half a
            # line), not when one line merely follows another a little to the right.
            if min(ay1, by1) - max(ay0, by0) >= 0.5 * min(asz, bsz) and (ax1 <= bx0 or bx1 <= ax0):
                parent[find(a)] = find(b)
                paired.update((a, b))
    # A letterhead line by line ("SỞ GIÁO DỤC…" | "ĐỀ KIỂM TRA…", then the next pair):
    # paired blocks stacked in one column within a line of each other are one group.
    for i, a in enumerate(ids):
        ax0, ay0, ax1, ay1, asz = boxes[a]
        for b in ids[i + 1 :]:
            bx0, by0, bx1, by1, bsz = boxes[b]
            if a in paired and b in paired and min(ax1, bx1) > max(ax0, bx0):
                if max(by0 - ay1, ay0 - by1) <= max(asz, bsz):
                    parent[find(a)] = find(b)

    members: dict[int, list[int]] = {}
    for b in ids:
        members.setdefault(find(b), []).append(b)
    groups = []
    for blocks in members.values():
        cols: list[list[int]] = []
        for b in sorted(blocks, key=lambda b: boxes[b][0]):
            if cols and boxes[b][0] < max(boxes[m][2] for m in cols[-1]):
                cols[-1].append(b)
            else:
                cols.append([b])
        if len(cols) >= 2:
            groups.append(cols)
    return groups


def _columns(
    cols: list[list[int]], paras_by_block: dict[int, list[Paragraph]], left: float, right: float
) -> Columns:
    """Lay blocks out as table cells. A column whose lines share one centre is
    centred in a cell symmetric about that centre, so Word's slightly wider
    glyphs grow both ways instead of wrapping; spacer cells fill the gaps."""
    groups = [sorted((p for b in c for p in paras_by_block[b]), key=lambda p: p.y0) for c in cols]
    spans = [(min(p.x0 for p in g), max(p.x1 for p in g)) for g in groups]
    mids = [min(left, spans[0][0])] + [(a[1] + b[0]) / 2 for a, b in pairwise(spans)]
    mids.append(max(right, spans[-1][1]))
    bounds, cells = [mids[0]], []
    for col, (lo, hi) in zip(groups, pairwise(mids), strict=True):
        centers = [(p.x0 + p.x1) / 2 for p in col]
        if len(col) > 1 and max(centers) - min(centers) < 3:
            c = sum(centers) / len(centers)
            half = min(c - lo, hi - c)
            for p in col:
                p.align, p.first_line_indent_pt = "center", 0.0
            if c - half > lo + 0.5:
                bounds.append(c - half)
                cells.append([])
            bounds.append(c + half)
            cells.append(col)
            if c + half < hi - 0.5:
                bounds.append(hi)
                cells.append([])
            continue
        for p in col:
            if len(p.line_spans) > 1:
                p.align, indent = alignment(p.line_spans, min(q.x0 for q in col), max(q.x1 for q in col))
                p.first_line_indent_pt = indent if abs(indent) > 3 else 0.0
        bounds.append(hi)
        cells.append(col)
    flat = [p for g in groups for p in g]
    return Columns(
        bounds=bounds,
        cells=cells,
        x0=bounds[0],
        y0=min(p.y0 for p in flat),
        x1=bounds[-1],
        y1=max(p.y1 for p in flat),
    )


def _rgb(pix: pymupdf.Pixmap) -> np.ndarray:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return img[:, : pix.width * 3].reshape(pix.height, pix.width, 3).copy()


def _paint_mask(img: np.ndarray, clip: pymupdf.Rect, dpi: int, paints) -> np.ndarray:
    """Pixels that are exactly a watermark's colour over white paper, or a
    blend of it with white (anti-aliased edges). A figure's own pixels
    (darker, or of another hue) are not."""
    s = dpi / 72
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for rect, paint in paints:
        box = rect & clip
        if box.is_empty:
            continue
        x0, y0 = max(int((box.x0 - clip.x0) * s), 0), max(int((box.y0 - clip.y0) * s), 0)
        x1, y1 = min(int((box.x1 - clip.x0) * s) + 1, w), min(int((box.y1 - clip.y0) * s) + 1, h)
        sub = img[y0:y1, x0:x1].astype(float)
        gap = 255 - paint
        channels = gap >= 8  # channels where the paint differs visibly from white
        if not channels.any():
            continue
        t = (255 - sub[..., channels]) / gap[channels]  # how much paint each pixel holds
        mask[y0:y1, x0:x1] |= (t.min(axis=2) > -0.05) & (t.max(axis=2) < 1.15) & (t.max(axis=2) - t.min(axis=2) < 0.25)
    return mask


def _clear_paint(img: np.ndarray, clip: pymupdf.Rect, dpi: int, paints) -> None:
    """Whiten the pixels of a watermark's paint (see _paint_mask)."""
    img[_paint_mask(img, clip, dpi, paints)] = 255


def _settle_equations(ex: _Extractor, flow: list) -> None:
    """Equations whose fractions are as large as their line are display equations in Word, which
    stand alone in their paragraph: alone there, the run says so (Run.display); beside text they
    keep their picture. Fractions alone beside text are an inline equation at their line's size
    (Word sets inline fraction parts at about 0.72 of the line). A MathType object sets its fractions
    that large in running text: beside text it is an inline equation all the same (its fractions
    come out smaller), since its picture could not be edited."""
    pending = {id(f["run"]): f for f in ex.formulas
               if f["run"] is not None and f["run"].math and (f.get("display") or f.get("frac_only"))}
    if not pending:
        return
    holders: list[tuple[list[Run], float]] = []
    for e in flow:
        if isinstance(e, Paragraph):
            holders.append((e.runs, e.font_pt))
        elif isinstance(e, Columns):
            holders += [(p.runs, p.font_pt) for col in e.cells for p in col]
        elif isinstance(e, Table):
            holders += [(c.runs, e.font_pt) for row in e.cells for c in row]
    for runs, font_pt in holders:
        for r in list(runs):  # a picture takes its label runs out of `runs` as we go
            f = pending.get(id(r))
            if f is None or not any(x is r for x in runs):
                continue
            alone = all(not (x.math or x.image or x.field or x.text == TAB or x.text.strip()) for x in runs if x is not r)
            if alone:
                r.display = True
                runs[:] = [r]  # the blanks beside it are the equation's own spacing: a display paragraph holds it alone
            elif f.get("mt"):
                if f.get("frac_only"):
                    r.size = _half(font_pt)
            elif f.get("frac_only") and r.size <= nativemath.SCRIPT * font_pt:
                r.size = _half(font_pt)
            else:  # the picture, as _runs makes it: it shows the label and trailing text too
                box, image = ex._formula_picture(f)
                r.math = r.math_text = None
                r.size = _half(f["line_main"])
                r.image, r.image_w, r.image_h, r.image_depth = image, box.width, box.height, box.y1 - f["line_baseline"]
                f["box"] = box
                others = {id(x) for x in f["piece_runs"] if x is not r}
                runs[:] = [x for x in runs if id(x) not in others]


def native_page(
    page: pymupdf.Page, ocr_languages: list[str] | None = None, strip: bool = False
) -> PageLayout:
    ex = _Extractor(page, ocr_languages)
    # Ruled tables first: their cell shading must not be taken for highlighted text.
    table_specs = ex.tables()
    table_rects = [spec[0] for spec in table_specs]
    ex.table_rects = table_rects
    ex.table_lines = [(pymupdf.Rect(spec[0]), spec[1], spec[2]) for spec in table_specs]
    rows = ex.rows()
    layout = PageLayout(width_pt=ex.width, height_pt=ex.height, source="native")
    table_rows: list[list[_Row]] = [[] for _ in table_specs]
    free_rows: list[_Row] = []
    for row in rows:
        center = pymupdf.Point((row.x0 + row.x1) / 2, row.baseline - 0.3 * row.size)
        for i, rect in enumerate(table_rects):
            if center in rect:
                table_rows[i].append(row)
                break
        else:
            free_rows.append(row)
    flow: list = [ex._table_element(*spec, trs) for spec, trs in zip(table_specs, table_rows, strict=True)]

    # Raster regions holding text are OCR'd in OCR mode.
    rasters = ex.raster_regions()
    if ocr_languages:
        kept = []
        for r in rasters:
            has_native_text = any(not (row.rect & r).is_empty for row in free_rows)
            if abs(r) >= OCR_REGION_MIN_AREA * ex.width * ex.height and not has_native_text:
                region = ocr_region(page, r, ocr_languages)
                if region.word_count >= 15 and region.conf_sum / region.word_count >= 70:
                    flow.extend(region.elements)
                    layout.word_count += region.word_count
                    layout.conf_sum += region.conf_sum
                    layout.ocr_regions += 1
                    continue
            kept.append(r)
        rasters = kept

    # Figures and backgrounds; rotated text (watermarks) joins the background.
    consumed: set[int] = set()
    figures: list[pymupdf.Rect] = []
    backgrounds = [pymupdf.Rect(line["bbox"]) & page.rect for line in ex.rotated_lines]
    behind: list[pymupdf.Rect] = []  # raster images behind the text
    for n, region in enumerate(rasters + ex.vector_regions(table_rects)):
        kind, inside = ex._classify(region, free_rows)
        if kind == "figure":
            figures.append(region)
            consumed.update(inside)
        else:
            backgrounds.append(region)
            if n < len(rasters):
                behind.append(region)
    rotated = [pymupdf.Rect(line["bbox"]) & page.rect for line in ex.rotated_lines]
    marks: list[pymupdf.Rect] = []  # watermark regions
    for region in _merge_boxes(backgrounds, gap=0):
        watermark = ex.is_watermark(region, rotated)
        layout.pictures.append(
            Picture(ex._render(region, background=True), *region, behind_text=True, wrap=False, watermark=watermark)
        )
        if watermark:
            marks.append(region)
    # When watermarks are stripped, figures drop the watermark showing through them: watermark
    # images are left out, painted ones and rotated text are cleared by their colour.
    paints = ex.watermark_paints(backgrounds) + ex.rotated_paints() if strip else None
    hidden = [r for r in behind if any(not (r & m).is_empty for m in marks)] if strip else []
    obstacles: list[pymupdf.Rect] = []  # figures text wraps around
    for region in _merge_boxes(figures, gap=2):
        labels = ex._classify(region, free_rows)[1]
        consumed.update(labels)
        for i in labels:  # labels may stick out of the image they annotate
            region |= free_rows[i].rect
        thin = region.width < 6 or region.height < 6
        r = _min_size(region)
        image = ex._render(region, background=False, paints=paints, hidden=hidden, figures=figures)
        layout.pictures.append(Picture(image, *r, wrap=not thin))
        if not thin:
            obstacles.append(r)
    free_rows = [row for i, row in enumerate(free_rows) if i not in consumed]

    # Blocks set side by side keep their own paragraphs (written as Columns);
    # the rest flows top to bottom and paragraphs may span MuPDF blocks.
    groups = _side_by_side(free_rows)
    in_band = {b for cols in groups for col in cols for b in col}
    flow_rows = sorted((r for r in free_rows if r.block not in in_band), key=lambda r: r.baseline)
    text_left = min((r.x0 for r in free_rows), default=0)
    right = max((r.x1 for r in flow_rows), default=ex.width)
    flow.extend(ex.paragraphs(flow_rows, right, obstacles))
    for cols in groups:
        by_block: dict[int, list[_Row]] = {}
        for r in free_rows:
            if any(r.block in col for col in cols):
                by_block.setdefault(r.block, []).append(r)
        paras = {b: ex.paragraphs(rs, max(x.x1 for x in rs), obstacles) for b, rs in by_block.items()}
        element = _columns(cols, paras, text_left, right)
        in_zone = element.y1 < HEADER_ZONE * ex.height or element.y0 > (1 - HEADER_ZONE) * ex.height
        # Pieces of a header/footer line stay separate paragraphs; they are
        # joined with tabs when moved into the Word header/footer.
        flow.extend([p for col in element.cells for p in col] if in_zone else [element])

    # Header/footer zones.
    top_paras = [e for e in flow if isinstance(e, Paragraph) and e.baselines]
    for p in top_paras:
        middle = (p.y0 + p.y1) / 2
        if middle < HEADER_ZONE * ex.height:
            p.zone = "top"
        elif middle > (1 - HEADER_ZONE) * ex.height:
            p.zone = "bottom"
    links = [pymupdf.Rect(lk["from"]) for lk in page.get_links() if lk.get("uri")]
    for p in top_paras:
        # A link annotation can stretch over neighbours (the footer link here
        # also covers "Trang 1/2"), so it only marks text it starts at.
        middle = pymupdf.Point((p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2)
        p.link = looks_like_link(p.text) or any(middle in r and abs(r.x0 - p.x0) < 3 for r in links)
    body = [e for e in flow if not (isinstance(e, Paragraph) and e.zone)] or flow
    left = min(e.x0 for e in body) if body else 0
    right = max(e.x1 for e in body) if body else ex.width
    for p in top_paras:
        p.align, indent = alignment(p.line_spans, left, right)
        p.first_line_indent_pt = indent if abs(indent) > 3 else 0.0

    _settle_equations(ex, flow)

    # Line heights from the measured pitch; boxes become line-box extents so
    # spacing reproduces the original baselines.
    all_paras = top_paras + [p for e in flow if isinstance(e, Columns) for col in e.cells for p in col]
    ratios = [p.line_pitch_pt / p.font_pt for p in all_paras if p.line_pitch_pt]
    ratio = min(max(median(ratios), 1.0), 2.5) if ratios else 1.2
    boxes = {id(f["run"]): f["box"] for f in ex.formulas if f["run"] is not None}
    for p in all_paras:
        p.line_height_pt = p.line_pitch_pt or p.font_pt * ratio
        first, last = p.baselines
        p.y0 = first - p.line_height_pt + DESCENT * p.font_pt
        p.y1 = last + DESCENT * p.font_pt
        # A formula picture taller than the line makes Word's line taller too: the
        # box reaches its top and bottom, so the gap to the next paragraph (which
        # already held that height in the PDF) is not added twice.
        for box in (boxes[id(r)] for r in p.runs if id(r) in boxes):
            if box.y0 <= first <= box.y1:
                p.y0 = min(p.y0, box.y0)
            if box.y0 <= last <= box.y1:
                p.y1 = max(p.y1, box.y1)
    for col_el in (e for e in flow if isinstance(e, Columns)):
        flat = [p for col in col_el.cells for p in col]
        col_el.y0, col_el.y1 = min(p.y0 for p in flat), max(p.y1 for p in flat)
        for col in col_el.cells:
            prev = col_el.y0
            for p in col:
                p.space_before_pt = max(p.y0 - prev, 0)
                prev = p.y1

    flow.sort(key=lambda e: e.y0)
    figures_y = [(p.y0, p.y1) for p in layout.pictures if p.wrap and not p.behind_text]
    prev_bottom = None
    for el in flow:
        if prev_bottom is not None:
            gap = max(el.y0 - prev_bottom, 0)
            # A gap holding a figure keeps its height; others are capped (stray space).
            held = any(y0 < el.y0 and y1 > prev_bottom for y0, y1 in figures_y)
            el.space_before_pt = gap if held else min(gap, 72)
        prev_bottom = el.y1
    layout.elements = flow
    layout.legacy_spans = ex.legacy_spans
    placed = [f["run"] for f in ex.formulas if f["run"] is not None]
    layout.equations = sum(1 for r in placed if r.math)
    layout.native_formula_pictures = len(placed) - layout.equations
    if ex._forms_doc is not None:  # the page copy of MathType forms: its pictures are drawn
        ex._forms_doc.close()
        ex._forms_doc = None
    return layout


def image_page(page: pymupdf.Page, dpi: int = 200) -> PageLayout:
    """A page kept as one picture (basic mode for scans)."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    picture = Picture(pix.tobytes("jpeg", jpg_quality=85), *page.rect, wrap=False)
    return PageLayout(page.rect.width, page.rect.height, pictures=[picture], source="image")
