"""Page layout shared by the native-text and OCR pipelines and the DOCX writer.

Coordinates are PDF points from the page's top-left corner.
"""

from dataclasses import dataclass, field


@dataclass
class Run:
    text: str = ""
    size: float | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: str | None = None  # "RRGGBB"; None = automatic
    font: str | None = None
    script: str | None = None  # "super" | "sub"
    field: str | None = None  # "PAGE" | "NUMPAGES"
    highlight: str | None = None  # "RRGGBB" fill painted behind the text (Word highlight / shading)
    # A formula kept as a picture (its glyphs had no usable text): PNG, size in points and how
    # far it reaches below the baseline (text is "\ufffc").
    image: bytes | None = None
    image_w: float = 0.0
    image_h: float = 0.0
    image_depth: float = 0.0
    stamp: bool = False  # part of a faded stamp or a link (OCR tab rows)
    # A Word equation: tidy LaTeX (app/omml.py), text "\ufffc". Should the LaTeX not convert,
    # the run is written as its picture (image) if it has one, else as math_text.
    math: str | None = None
    math_text: str | None = None
    formula: bool = False  # read by the formula model (OCR): keeps its own weight in a bold paragraph


TAB = "\t"
LINE_BREAK = "\n"


@dataclass
class Paragraph:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_pt: float
    runs: list[Run] = field(default_factory=list)  # empty: one run of `text`
    tab_stops: list[tuple[float, str]] = field(default_factory=list)  # (absolute x, left|center|right)
    bold: bool = False
    align: str = "left"
    first_line_indent_pt: float = 0.0
    space_before_pt: float = 0.0
    line_pitch_pt: float | None = None  # OCR: measured baseline pitch
    line_height_pt: float | None = None  # native: "at least" line height
    # native: where the longest line would end with unjustified spacing; Word
    # needs that much room to keep the PDF's line breaks.
    natural_right: float | None = None
    zone: str | None = None  # "top" / "bottom": header or footer candidate
    link: bool = False  # a hyperlink (annotation starting at it) or URL-looking text
    translucent: bool = False  # all text semi-transparent: a stamped overlay
    baselines: tuple[float, float] | None = field(default=None, repr=False)  # first, last
    line_spans: list[tuple[float, float]] = field(default_factory=list, repr=False)


@dataclass
class Cell:
    text: str = ""
    bold: bool = False
    runs: list[Run] = field(default_factory=list)
    shading: str | None = None  # "RRGGBB" cell fill
    align: str | None = None  # "center" | "right"; None: left


@dataclass
class Table:
    cells: list[list[Cell]]
    col_widths_pt: list[float]
    row_heights_pt: list[float]
    x0: float
    y0: float
    x1: float
    y1: float
    font_pt: float
    space_before_pt: float = 0.0
    borders: bool = True
    merges: list[tuple[int, int, int, int]] = field(default_factory=list)  # (row, column, rows, columns)
    cell_margin_pt: float | None = None  # left/right padding in the cells; None: Word's default


@dataclass
class Columns:
    """Blocks set side by side (e.g. a two-part letterhead), written as a
    borderless one-row table so their positions survive."""

    bounds: list[float]  # x edges, one more than the number of columns
    cells: list[list[Paragraph]]
    x0: float
    y0: float
    x1: float
    y1: float
    space_before_pt: float = 0.0

    @property
    def font_pt(self) -> float:
        sizes = [p.font_pt for col in self.cells for p in col]
        return max(sizes) if sizes else 11.0


@dataclass
class Picture:
    """A floating image positioned on the page."""

    data: bytes  # PNG or JPEG
    x0: float
    y0: float
    x1: float
    y1: float
    behind_text: bool = False  # watermark / background: no text wrapping
    wrap: bool = True  # text flows around it (False: in front, no wrapping)
    scanned: bool = False  # cut from a scan, so it may carry bits of a pale watermark
    # behind_text only: True for a watermark (large, rotated text, pale or translucent paint), False
    # for small dark line art drawn behind the text (rules, bars); None: unknown (treated as watermark)
    watermark: bool | None = None


@dataclass
class PageLayout:
    width_pt: float
    height_pt: float
    elements: list = field(default_factory=list)  # Paragraph | Table | Columns, in reading order
    pictures: list[Picture] = field(default_factory=list)
    source: str = "native"  # native | ocr | image
    word_count: int = 0  # OCR words
    conf_sum: float = 0.0
    legacy_spans: int = 0  # spans converted from VNI/TCVN3
    ocr_regions: int = 0  # raster regions replaced by OCR text
    formula_pictures: int = 0  # OCR: formula regions kept as pictures (fractions, roots not read)
    equations: int = 0  # native: Word equations rebuilt from the text layer
    native_formula_pictures: int = 0  # native: formulas kept as pictures (MathType, layouts not read)
