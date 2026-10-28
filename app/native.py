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

import re
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


CMAP_BEGIN = re.compile(r"begin(bfchar|bfrange)")
HEX = re.compile(r"<([0-9A-Fa-f\s]*)>")  # cannot run past a "<": linear on any line
CMAP_MAX = 64 * 1024  # bytes: a math font's map is a few KB; a larger one is not read (its formulas stay pictures)
FONT_LIST_MAX = 16000  # characters of a page's font dictionary (~800 fonts; Word's pages list ~20)
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
    "pV = R/T" then reads "!\"", and ")" may stand for any glyph). None: a map too large to read."""
    tu_kind, tu = doc.xref_get_key(xref, "ToUnicode")
    enc_kind, enc = doc.xref_get_key(xref, "Encoding")
    first, last = (doc.xref_get_key(xref, k) for k in ("FirstChar", "LastChar"))
    try:  # a simple font's codes are one byte, whatever the file claims
        lo, hi = max(int(first[1]), 0), min(int(last[1]), 255)
    except ValueError:
        lo, hi = 0, 255
    cache = doc.__dict__.setdefault("_p2w_maps", {})  # per file, by map: fonts often share one
    key = (tu if tu_kind == "xref" else None, lo, hi, enc_kind, enc if enc_kind == "name" else "")
    if key in cache:
        return cache[key]
    mapped: set[int] = set()
    if tu_kind == "xref":
        ref = int(tu.split()[0])
        raw = doc.xref_stream_raw(ref)
        if raw is None or len(raw) > CMAP_MAX:
            return None
        data = doc.xref_stream(ref)
        if data is None or len(data) > 4 * CMAP_MAX:
            return None
        cmap, pos = data.decode("latin-1"), 0
        while m := CMAP_BEGIN.search(cmap, pos):  # one pass: an unclosed block ends the map
            stop = cmap.find("end" + m.group(1), m.end())
            if stop < 0:
                break
            for line in cmap[m.end():stop].splitlines():
                codes = [c.replace(" ", "") for c in HEX.findall(line)]  # "<21><d835 dc5d>"
                try:
                    if m.group(1) == "bfchar" and len(codes) >= 2:
                        mapped.add(int(codes[0], 16))
                    elif m.group(1) == "bfrange" and len(codes) >= 2:
                        mapped.update(range(max(int(codes[0], 16), 0), min(int(codes[1], 16), 255) + 1))
                except ValueError:
                    continue
            pos = stop
    elif enc_kind == "name" and enc in STANDARD_ENCODINGS:
        cache[key] = frozenset()  # a standard encoding names every glyph
        return cache[key]
    out = set()
    for code in range(lo, hi + 1):
        if code not in mapped:
            out.add(chr(code))
            for codec in ("cp1252", "mac_roman"):  # how MuPDF may read it instead
                out.update(bytes([code]).decode(codec, errors="ignore"))
    cache[key] = frozenset(out)
    return cache[key]


FONT_REF = re.compile(r"/[^\s/<>\[\]()]+\s+(\d+)\s+0\s+R")


def _page_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> list[tuple[int, str, list[str]]] | None:
    """(xref, Subtype, names) of the fonts the page's own resources name (inherited ones too):
    BaseFont, and a Type0 font's descendant's name, which the text spans carry. Read from the
    /Font dictionary itself, bounded (FONT_LIST_MAX), not by listing fonts: a file can name
    thousands, in forms too. Forms' own fonts are not in it: their formulas stay pictures.
    None: no resources or a dictionary too large."""
    ref = page.xref
    for _ in range(8):
        kind, val = doc.xref_get_key(ref, "Resources")
        if kind != "null":
            break
        kind, val = doc.xref_get_key(ref, "Parent")
        if kind != "xref":
            return None
        ref = int(val.split()[0])
    else:
        return None
    fkind, fval = doc.xref_get_key(int(val.split()[0]), "Font") if kind == "xref" else doc.xref_get_key(ref, "Resources/Font")
    if fkind == "xref":
        fval = doc.xref_object(int(fval.split()[0]), compressed=True)
    elif fkind != "dict":
        return []
    if len(fval) > FONT_LIST_MAX:
        return None
    cache = doc.__dict__.setdefault("_p2w_font_lists", {})  # pages share their font dictionaries
    if fval not in cache:
        out = []
        for m in FONT_REF.finditer(fval):
            x = int(m.group(1))
            sub = doc.xref_get_key(x, "Subtype")[1].lstrip("/")
            names = [doc.xref_get_key(x, "BaseFont")[1].lstrip("/")]
            if sub == "Type0":
                desc = re.search(r"(\d+)\s+0\s+R", doc.xref_get_key(x, "DescendantFonts")[1])
                if desc:
                    names.append(doc.xref_get_key(int(desc.group(1)), "BaseFont")[1].lstrip("/"))
            out.append((x, sub, names))
        cache[fval] = out
    return cache[fval]


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
    return over + under if over and under else []


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
        self.highlights = [
            (pymupdf.Rect(d["rect"]), _hex(d["fill"]), i)
            for i, d in enumerate(self.drawings)
            if _highlight_paint(d) and 3 <= d["rect"].height <= 40 and d["rect"].width <= 0.9 * self.width
        ]
        self.formulas: list[dict] = []  # {"rect", "size", "run"}: formulas kept as pictures
        self.formula_chars: set[int] = set()  # id() of the glyphs drawn by those pictures
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
                return self._equation_runs(f, main)
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
                        if s["size"] < 4:
                            self._tiny.append(pymupdf.Rect(c["bbox"]))
                            continue
                        span_of[id(c)] = s
                        kind = 2 if formula_span or _math_alnum(c["c"]) else 1 if joins else 0
                        marks.append((pymupdf.Rect(c["bbox"]), s["size"], c, len(nodes) if kind else None))
                        if kind:
                            nodes.append([pymupdf.Rect(c["bbox"]), kind, s["size"], None, c])
        if not nodes:
            return []
        in_table = lambda r: any(  # noqa: E731
            t.x0 - 1 <= r.x0 and r.x1 <= t.x1 + 1 and t.y0 - 1 <= r.y0 and r.y1 <= t.y1 + 1 for t in self.table_rects
        )
        fractions: list[tuple[int, list[int]]] = []  # (bar node, glyph nodes over and under it)
        for i, d in enumerate(self.drawings):
            r = pymupdf.Rect(d["rect"])
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
        fraction_bars = {nodes[bar][3] for bar, _parts in fractions}
        out = []
        for f in found:
            sizes = [size_of[id(c)] for c in f["chars"]]
            # the main line's baseline: where the largest glyphs sit
            top = max(sizes)
            baseline = median(c["origin"][1] for c, size in zip(f["chars"], sizes) if size > top - 0.5)
            self.formula_drawings |= f["drawings"]
            self.used_drawings |= f["drawings"]
            self.formula_chars.update(id(c) for c in f["chars"])
            # what the picture shows: the glyph boxes (italics lean out of them) and the strokes
            parts = [(pymupdf.Rect(c["bbox"]), 0.2 * size, 0.1 * size) for c, size in zip(f["chars"], sizes)]
            ink = pymupdf.Rect(f["rect"].x0, min(c["origin"][1] - 0.7 * size for c, size in zip(f["chars"], sizes)),
                               f["rect"].x1, max(c["origin"][1] + 0.2 * size for c, size in zip(f["chars"], sizes)))
            for i in f["drawings"]:
                ink |= self.drawings[i]["rect"]
            parts += [(pymupdf.Rect(self.drawings[i]["rect"]), 1.0, 1.0) for i in f["drawings"]]
            out.append({
                "rect": f["rect"], "size": median(sizes), "baseline": baseline, "parts": parts, "run": None, "ink": ink,
                "pieces": self._equation(f, span_of, fraction_bars),
            })
        return sorted(out, key=lambda f: (f["rect"].y0, f["rect"].x0))

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
        if len(self._tiny):
            if not isinstance(self._tiny, np.ndarray):
                self._tiny = np.asarray([tuple(r) for r in self._tiny])
            t = self._tiny
            if np.any((t[:, 0] < ink.x1) & (t[:, 2] > ink.x0) & (t[:, 1] < ink.y1) & (t[:, 3] > ink.y0)):
                return None  # glyphs too small to be read are part of it: its picture shows them
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
                bars.append(nativemath.Bar(r.x0, r.x1, (r.y0 + r.y1) / 2))
            elif r.width <= 1 and r.height > 0:
                uprights.append((r.x0, r.y0, r.x1, r.y1))
            else:
                others += 1
        pieces = nativemath.build(glyphs, bars, uprights, others)
        # a bracket drawn taller than everything the formula holds encloses rows outside it (a matrix, a
        # system): the text layer gives a stretched bracket the box of a plain one, so its ink is measured
        content = [g for g in glyphs if g.c not in "()[]{}"]
        if pieces and content and len(content) < len(glyphs):
            top = min([g.oy - 0.78 * g.size for g in content] + [self.drawings[i]["rect"].y0 for i in f["drawings"]])
            bottom = max([g.oy + 0.22 * g.size for g in content] + [self.drawings[i]["rect"].y1 for i in f["drawings"]])
            for g in glyphs:
                if g.c in "()[]{}":
                    height = self._ink_height(g)
                    if height is None or height > bottom - top + 0.5 * g.size:
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
                    self._page_ink = (rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
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

    def _equation_runs(self, f: dict, main: float) -> list[Run]:
        """Runs of a formula rebuilt as an equation: the equation and any label set before or
        after it ("b)", a trailing dot in the text font)."""
        runs = []
        for piece in f["pieces"]:
            if piece.kind == "math":
                run = Run(text=OBJ, size=_half(piece.size or main), color=piece.color, font=piece.font, math=piece.text,
                          math_text=piece.text)
                f["run"] = run
                f["display"], f["frac_only"] = piece.display, piece.frac_only  # settled with its paragraph
                # the box Word's equation fills: glyph ink (about 0.7 em over the baseline, 0.2 under) and bars
                f["box"] = f["ink"]
            else:
                run = Run(text=piece.text, size=_half(piece.size), bold=piece.bold, italic=piece.italic,
                          color=piece.color, font=piece.font)
            runs.append(run)
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

    def _formula_picture(self, f: dict) -> tuple[pymupdf.Rect, bytes]:
        """The formula's picture and the box it shows. It holds the formula's own
        glyphs and strokes on a clear ground: no watermark or shading from behind
        (the page drawn without its text and formula strokes is taken away), no
        bits of the lines around it. Across, it spans exactly the glyph boxes
        (their advances carry the spacing to the text around it, and any extra
        width moves Word's line breaks); up and down, only the ink – glyph boxes
        reach the font's whole ascent and descent and would make the line taller."""
        rect = f["rect"]
        clip = pymupdf.Rect(rect.x0, rect.y0 - 0.6, rect.x1, rect.y1 + 0.6) & self.page.rect
        full = _rgb(self.page.get_pixmap(dpi=FORMULA_DPI, clip=clip, alpha=False))
        ground = _rgb(self._ground_page().get_pixmap(dpi=FORMULA_DPI, clip=clip, alpha=False))
        h, w = full.shape[:2]
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
    (Word sets inline fraction parts at about 0.72 of the line)."""
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
        for r in runs:
            f = pending.get(id(r))
            if f is None:
                continue
            alone = all(not (x.math or x.image or x.field or x.text == TAB or x.text.strip()) for x in runs if x is not r)
            if alone:
                r.display = True
            elif f.get("frac_only") and r.size <= nativemath.SCRIPT * font_pt:
                r.size = _half(font_pt)
            else:
                box, image = ex._formula_picture(f)
                r.math = r.math_text = None
                r.image, r.image_w, r.image_h, r.image_depth = image, box.width, box.height, box.y1 - f["baseline"]
                f["box"] = box


def native_page(
    page: pymupdf.Page, ocr_languages: list[str] | None = None, strip: bool = False
) -> PageLayout:
    ex = _Extractor(page, ocr_languages)
    # Ruled tables first: their cell shading must not be taken for highlighted text.
    table_specs = ex.tables()
    table_rects = [spec[0] for spec in table_specs]
    ex.table_rects = table_rects
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
    return layout


def image_page(page: pymupdf.Page, dpi: int = 200) -> PageLayout:
    """A page kept as one picture (basic mode for scans)."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    picture = Picture(pix.tobytes("jpeg", jpg_quality=85), *page.rect, wrap=False)
    return PageLayout(page.rect.width, page.rect.height, pictures=[picture], source="image")
