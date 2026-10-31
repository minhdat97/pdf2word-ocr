"""The `pdf-to-word` tool.

Modes:
- basic:    pages with a text layer are rebuilt from it; scanned pages become pictures.
- ocr:      as basic, but scanned pages and large text-bearing images are OCR'd.
- ocr-full: every page is OCR'd, ignoring the text layer (for broken text layers).

With strip_watermarks, background watermarks, semi-transparent stamped text and
links in the header/footer zone are dropped before the document is written.
"""

import hashlib
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from . import mathocr, nativeread, vietocr
from .analyze import open_pdf, page_kind
from .docx_builder import build_docx
from .errors import TaskError
from .layout import canonical, looks_like_page_label, looks_like_page_number
from .model import TAB, PageLayout, Paragraph, Picture, Run
from .native import HEADER_ZONE, ITEM_LABEL, image_page, native_page
from .ocr import formula_reader, formulas_mode, ocr_page
from .settings import MAX_DRAWINGS, MFR_MAX_FORMULAS, MFR_MAX_PAGES, VIETOCR_MAX_PAGES

MODES = ("basic", "ocr", "ocr-full")


def _zone_text(p: Paragraph) -> str:
    """A zone line's text for telling repeated lines: an equation counts by its LaTeX and a formula picture
    by its bytes, not as the one placeholder character they share ("A. ￼ B. ￼" on every page)."""
    if not any(r.math or r.image for r in p.runs):
        return p.text

    def picture(r: Run) -> str:
        if r.formula:  # cut from a scan: its pixels differ on every page, its size does not
            return f"\x00{round(r.image_w / 3)}x{round(r.image_h / 3)}"
        return "\x00" + hashlib.sha1(r.image).hexdigest()[:12]

    return "".join((r.math_text or r.math) if r.math else picture(r) if r.image else r.text for r in p.runs)


def _same_numbers(found: list[tuple[int, str]], total: int) -> bool:
    """The numbers of a line seen on several pages (page index, text) are the same on each, or the ones that
    change follow the page number (with a fixed offset: a file that starts at page 5) or are the page count."""
    runs = [(i, re.findall(r"\d+", text)) for i, text in found]
    if len({len(r) for _, r in runs}) != 1:
        return False
    for pos in range(len(runs[0][1])):
        values = [int(r[pos]) for _, r in runs]
        if len(set(values)) == 1:
            continue
        if len({v - i for v, (i, _) in zip(values, runs)}) != 1 and set(values) != {total}:
            return False
    return True


def _norm(text: str) -> str:
    # Spaces are ignored: OCR reads a site name in the footer with a space
    # on one page and without it on the next.
    return re.sub(r"\d+", "#", "".join(canonical(text).split())).lower()


def _page_number_fields(template: Paragraph, occurrences: list[tuple[int, str]], total: int) -> None:
    """Turn numbers that count pages ("Trang 1/2") into PAGE / NUMPAGES fields."""
    tokens = [(i, re.findall(r"\d+", text)) for i, text in occurrences]
    if len({len(t) for _, t in tokens}) != 1:
        return
    kinds = {}
    for k in range(len(tokens[0][1])):
        if all(t[k] == str(i + 1) for i, t in tokens):
            kinds[k] = "PAGE"
        elif all(t[k] == str(total) for _, t in tokens):
            kinds[k] = "NUMPAGES"
    if not kinds:
        return
    runs: list[Run] = []
    k = 0
    for r in template.runs:
        if r.field or r.text == TAB:
            runs.append(r)
            continue
        pos = 0
        for m in re.finditer(r"\d+", r.text):
            if k in kinds:
                if m.start() > pos:
                    runs.append(replace(r, text=r.text[pos : m.start()]))
                runs.append(replace(r, text=m.group(), field=kinds[k]))
                pos = m.end()
            k += 1
        if pos < len(r.text):
            runs.append(replace(r, text=r.text[pos:]))
    template.runs = [replace(r, text=canonical(r.text)) for r in runs]


def extract_header_footer(pages: list[PageLayout]) -> tuple[list[Paragraph], list[Paragraph]]:
    """Remove what repeats in the top/bottom zone – lines and pictures on most
    pages, or on all odd or all even pages of a two-sided book (logo on one,
    slogan on the other, the page number at the outer edge) – and number the
    pages instead: the Word header/footer holds "Trang x/y" and nothing else.
    Returns (header, footer) paragraphs; one of them holds the page number."""
    if len(pages) < 2:
        return [], []

    templates: dict[tuple, tuple[str, object]] = {}  # key -> (side, first occurrence)
    _move_repeated(pages, templates)
    # Page labels OCR read differently on each page, then whatever shared the
    # line with them (a site link) and now repeats on its own.
    numbering = _page_number_template(pages)
    if numbering:
        templates[("P", numbering.zone, "#page-number")] = ("all", numbering)
        _move_repeated(pages, templates)
    line = _page_number_line(pages, [(k, t) for k, (_, t) in templates.items()])
    return ([line], []) if line.zone == "top" else ([], [line])


def _page_number_line(pages: list[PageLayout], removed: list[tuple]) -> Paragraph:
    """ "Trang x/y" with Word fields, where the PDF numbered its pages (bottom
    centre when it did not, or when the number flipped between the edges); y is
    the PDF's own total when its labels say so ("Trang 1/4" in a 6-page file
    with 2 answer pages), else the page count."""
    labels = [
        t for k, t in removed
        if isinstance(t, Paragraph) and (k[2] == "#page-number" or looks_like_page_label(t.text))
    ]
    body = [e for page in pages for e in page.elements if isinstance(e, Paragraph)]
    width, height = pages[0].width_pt, pages[0].height_pt
    left = min((e.x0 for e in body), default=0)
    right = max((e.x1 for e in body), default=width)
    style = next((r for t in labels for r in t.runs if r.text.strip() and not r.field), None)
    size = (style.size if style and style.size else None) or round(median([e.font_pt for e in body] or [11.0]) * 2) / 2
    total = next((r.text for t in labels for r in t.runs if r.field == "NUMPAGES"), None)
    shown = next((m.group(1) for t in labels if (m := re.search(r"/\s*(\d+)", t.text))), None)
    denominator = Run(total or str(len(pages)), size=size, field="NUMPAGES") if total or not shown else Run(shown, size=size)
    runs = [Run("Trang ", size=size), Run("1", size=size, field="PAGE"), Run("/", size=size), denominator]
    zone, align, y1 = "bottom", "center", height - 18
    if labels:
        zone = Counter(t.zone for t in labels).most_common(1)[0][0]
        y1 = labels[0].y1
        edges = {"c" if abs((t.x0 + t.x1) / 2 - (left + right) / 2) < 0.1 * width else "l" if t.x1 < (left + right) / 2 else "r" for t in labels}
        align = {"l": "left", "r": "right", "c": "center"}[edges.pop()] if len(edges) == 1 else "center"
    return Paragraph(
        text="Trang #/#", x0=left, y0=y1 - 1.2 * size, x1=right, y1=y1, font_pt=size, runs=runs, align=align, zone=zone
    )


def _whiten_pale(data: bytes) -> bytes:
    """Clear pale grey pixels (watermark remnants) from a picture cut from a scan."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return data
    pale = (img.min(axis=2) >= 200) & (img.max(axis=2).astype(int) - img.min(axis=2) < 40)
    img[pale] = 255
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else data


def strip_watermarks(pages: list[PageLayout]) -> tuple[int, int]:
    """Remove pictures behind the text, translucent text and header/footer
    links. Returns (pictures removed, paragraphs removed)."""
    pictures = paragraphs = 0
    for page in pages:
        # Only watermarks go; small dark line art drawn behind the text (rules,
        # fraction bars) is content.
        gone = [p for p in page.pictures if p.behind_text and p.watermark is not False]
        pictures += len(gone)
        page.pictures = [p for p in page.pictures if p not in gone]
        for p in page.pictures:
            if p.scanned:
                p.data = _whiten_pale(p.data)
        kept = []
        for el in page.elements:
            # On scans "faded" is a guess from grey ink, and coloured text in a
            # greyscale scan looks the same, so only header/footer lines (and
            # tiny fragments OCR read from a watermark) are removed there.
            ocr_guess = page.source == "ocr" and not el_zone(el) and not _fragment(el)
            if not isinstance(el, Paragraph) or looks_like_page_number(el.text) or ocr_guess:
                kept.append(el)
            elif el.tab_stops and any(r.stamp for r in el.runs):
                # A row of pieces (OCR merged "https://site" with "Trang 1/2"):
                # drop the stamped pieces only.
                if _drop_stamped_pieces(el):
                    kept.append(el)
                paragraphs += 1
            elif el.translucent or (el.zone and el.link):
                paragraphs += 1
            else:
                kept.append(el)
        _keep_only(page, kept)
    return pictures, paragraphs


def _keep_only(page: PageLayout, kept: list) -> None:
    """Keep these of the page's elements. One that followed a dropped line (a
    header, a stamp, a link) gets its space before from what is now above it –
    nothing at the top of the page – not from the line that is gone: once the
    text flows on in Word that space would open up mid-page."""
    ids = {id(e) for e in kept}
    out, prev, after_gap = [], None, False
    held = [(p.y0, p.y1) for p in page.pictures if p.wrap and not p.behind_text]
    for el in page.elements:
        if id(el) not in ids:
            after_gap = True
            continue
        if after_gap:
            if prev is None:
                el.space_before_pt = 0.0
            else:
                gap = max(el.y0 - prev.y1 - (0.3 * el.font_pt if page.source == "ocr" else 0.0), 0.0)
                figure = any(y0 < el.y0 and y1 > prev.y1 for y0, y1 in held)
                el.space_before_pt = gap if figure else min(gap, 72.0)
            after_gap = False
        out.append(el)
        prev = el
    page.elements = out


def el_zone(el) -> str | None:
    return el.zone if isinstance(el, Paragraph) else None


def _fragment(el) -> bool:
    return isinstance(el, Paragraph) and el.translucent and len(el.text.strip()) <= 3


def _pieces_of(p: Paragraph) -> list[tuple[float, float, list[Run]]]:
    """(start, end, runs) of each tab-separated piece of a paragraph."""
    if not p.runs:
        return [(p.x0, p.x1, [Run(p.text, size=p.font_pt, bold=p.bold)])]
    groups: list[list[Run]] = [[]]
    for r in p.runs:
        if r.text == TAB:
            groups.append([])
        else:
            groups[-1].append(r)
    starts = [p.x0] + [x for x, _ in p.tab_stops][: len(groups) - 1]
    ends = starts[1:] + [p.x1]
    return list(zip(starts, ends, groups, strict=False))


def _keep_pieces(p: Paragraph, keep) -> bool:
    """Rebuild a paragraph from the pieces `keep(index, runs)` accepts; False if none is left."""
    kept = [(x, runs) for i, (x, _, runs) in enumerate(_pieces_of(p)) if runs and keep(i, runs)]
    if not kept:
        return False
    p.x0 = kept[0][0]
    p.runs = [r for i, (_, runs) in enumerate(kept) for r in ([Run(TAB)] if i else []) + runs]
    p.tab_stops = [(x, "left") for x, _ in kept[1:]]
    p.text = "".join(r.text for r in p.runs)
    return True


def _drop_stamped_pieces(p: Paragraph) -> bool:
    """Keep the pieces of a tab row that are not stamps; False if none is left."""
    def keep(_: int, runs: list[Run]) -> bool:
        return not all(r.stamp for r in runs) or looks_like_page_number("".join(r.text for r in runs))

    return _keep_pieces(p, keep)


def _picture_zone(pic: Picture, page: PageLayout) -> str | None:
    if pic.y1 <= HEADER_ZONE * page.height_pt + 2:
        return "top"
    if pic.y0 >= (1 - HEADER_ZONE) * page.height_pt - 2:
        return "bottom"
    return None


def _side(on: list[int], total: int) -> str | None:
    """Where something seen on pages `on` (0-based) of `total` belongs: on most
    pages of both kinds -> every page; on most odd (or even) pages and none of
    the other kind -> odd (even) pages only; else nowhere (it is body content)."""
    n_odd, n_even = (total + 1) // 2, total // 2
    odd = sum(1 for i in on if i % 2 == 0)
    even = len(on) - odd
    if len(on) >= max(2, (total + 1) // 2) and odd >= 0.5 * n_odd and even >= 0.5 * n_even:
        return "all"
    if not even and odd >= max(2, 0.6 * n_odd):
        return "odd"
    if not odd and even >= max(2, 0.6 * n_even):
        return "even"
    return None


def _move_repeated(pages: list[PageLayout], templates: dict[tuple, tuple[str, object]]) -> None:
    """Move zone lines and pictures repeated on the pages into `templates` (one
    per kind, with its side); numbers that count pages become PAGE / NUMPAGES
    fields. A line is told by its text and by where it sits (left, centre,
    right: a book's page number flips between the outer edges)."""
    width = pages[0].width_pt

    def key(el, page: PageLayout) -> tuple | None:
        if isinstance(el, Paragraph) and el.zone:
            mid = (el.x0 + el.x1) / 2
            edge = "c" if abs(mid - width / 2) < 0.1 * width else "l" if mid < width / 2 else "r"
            return ("P", el.zone, _norm(_zone_text(el)), edge)
        if isinstance(el, Picture) and (zone := _picture_zone(el, page)):
            return ("I", zone, round(el.x0 / 3), round(el.y0 / 3), round((el.x1 - el.x0) / 3), round((el.y1 - el.y0) / 3))
        return None

    on: dict[tuple, list[int]] = {}
    spans: dict[tuple, list[tuple[float, float]]] = {}
    texts: dict[tuple, list[tuple[int, str]]] = {}
    for i, page in enumerate(pages):
        for el in page.elements + page.pictures:
            if (k := key(el, page)) is not None:
                spans.setdefault(k, []).append((el.y0, el.y1))
                if i not in on.setdefault(k, []):
                    on[k].append(i)
                if isinstance(el, Paragraph):
                    texts.setdefault(k, []).append((i, _zone_text(el)))
    # Lines that only look alike once numbers are masked are not repeated unless their numbers are the same on
    # every page or count the pages ("A. v = 2 m/s …" at the foot of one page, "A. v = 1 m/s …" of the next) –
    # on text pages: OCR may read one digit of a header differently on one page; a question or an answer row
    # ("Câu 12.", "A. …") is never a header or footer
    for k in [k for k in on if k in texts]:
        native = all(pages[i].source == "native" for i, _ in texts[k])
        if ITEM_LABEL.match(texts[k][0][1].lstrip()) or (native and not _same_numbers(texts[k], len(pages))):
            del on[k]
    sides = {k: s for k, pages_on in on.items() if (s := _side(pages_on, len(pages)))}
    if any(s in ("odd", "even") for s in sides.values()):
        # Printed two-sided: what sits on every page of one kind and none of the
        # other belongs there too, even when that kind has a single page – then
        # only where the header/footer found so far sits: a 3-page OCR file whose
        # header was read differently on page 2 made all of page 2's top lines
        # "even-page headers", the first question with them.
        band: dict[str, tuple[float, float]] = {}
        for k in sides:
            for y0, y1 in spans[k]:
                lo, hi = band.get(k[1], (y0, y1))
                band[k[1]] = (min(lo, y0), max(hi, y1))
        kinds = {"odd": {i for i in range(len(pages)) if i % 2 == 0}, "even": {i for i in range(len(pages)) if i % 2}}
        for k, pages_on in on.items():
            for side, members in kinds.items():
                if k not in sides and members and set(pages_on) == members:
                    lo, hi = band.get(k[1], (0.0, float("inf")))
                    if len(members) == 1 and any(y0 > hi + 3 or y1 < lo - 3 for y0, y1 in spans[k]):
                        continue
                    sides[k] = side
    if not sides:
        return
    occurrences: dict[tuple, list[tuple[int, str]]] = {}
    for i, page in enumerate(pages):
        for attr in ("elements", "pictures"):
            kept = []
            for el in getattr(page, attr):
                k = key(el, page)
                if k in sides:
                    templates.setdefault(k, (sides[k], el))
                    if isinstance(el, Paragraph):
                        occurrences.setdefault(k, []).append((i, el.text))
                else:
                    kept.append(el)
            if attr == "elements":
                _keep_only(page, kept)
            else:
                page.pictures = kept
    for k, found in occurrences.items():
        _page_number_fields(templates[k][1], found, len(pages))


def _page_number_template(pages: list[PageLayout]) -> Paragraph | None:
    """Fallback for page numbers the exact-repeat pass missed (OCR reads
    "Trang 1/2" on one page and "Trang" on the next): when most pages carry a
    page label in the same zone, remove the static labels and return one
    header/footer paragraph built from Word PAGE / NUMPAGES fields."""
    found = []
    for i, page in enumerate(pages):
        for el in page.elements:
            if not (isinstance(el, Paragraph) and el.zone):
                continue
            for j, (x0, x1, runs) in enumerate(_pieces_of(el)):
                text = "".join(r.text for r in runs).strip()
                if text and looks_like_page_label(text):
                    found.append((i, el, x0, x1, text, runs, j))
    if not found:
        return None
    zone = Counter(item[1].zone for item in found).most_common(1)[0][0]
    first_per_page = {}
    for item in found:
        if item[1].zone == zone:
            first_per_page.setdefault(item[0], item)
    if len(first_per_page) < max(2, (len(pages) + 1) // 2):
        return None

    hits = list(first_per_page.values())
    for i, el, *_, j in hits:
        if not _keep_pieces(el, lambda index, _, target=j: index != target):
            _keep_only(pages[i], [e for e in pages[i].elements if e is not el])
    texts = [canonical(h[4]).lower() for h in hits]
    labels = (("trang", "Trang "), ("page", "Page "))
    prefix = next((label for key, label in labels if any(key in t for t in texts)), "")
    style = next((r for h in hits for r in h[5] if r.text.strip()), Run())
    look = dict(size=style.size, bold=style.bold, italic=style.italic, color=style.color, font=style.font)
    runs = ([Run(prefix, **look)] if prefix else []) + [Run("1", field="PAGE", **look)]
    if any("/" in t for t in texts):
        runs += [Run("/", **look), Run(str(len(pages)), field="NUMPAGES", **look)]
    # Place it where the fullest reading sat: OCR may have read only "Trang"
    # on one page, while "Trang 2/2" on another shows it reaches the margin.
    body = [e for page in pages for e in page.elements]
    right = max((e.x1 for e in body), default=pages[0].width_pt)
    center = (min((e.x0 for e in body), default=0) + right) / 2
    x0, x1 = median(h[2] for h in hits), max(h[3] for h in hits)
    align = "right" if x1 >= right - 6 else "center" if abs((x0 + x1) / 2 - center) < 6 else "left"
    _, el0, *_ = hits[0]
    return Paragraph(
        text=prefix + "#",
        x0=x0,
        y0=el0.y0,
        x1=x1,
        y1=el0.y1,
        font_pt=style.size or el0.font_pt,
        runs=runs,
        align=align,
        zone=zone,
        line_height_pt=el0.line_height_pt,
    )


def convert(
    pdf_path: Path,
    mode: str,
    languages: list[str],
    progress: Callable[[int, int], None],
    strip: bool = False,
    careful: bool = False,
    model_unused: Callable[[], None] | None = None,
) -> tuple[Path, dict]:
    """Convert a PDF to .docx. careful ("Chuyển thành công thức sửa được"): the formula model reads the formulas of
    scanned pages (up to about 2 minutes per dense page on a 2-core server) where the file has at most MFR_MAX_PAGES
    of them, otherwise they are read as settings.FORMULAS says (by default superscripts and subscripts without the
    model); and it reads the formulas text pages keep as pictures, up to MFR_MAX_FORMULAS per file
    (app/nativeread.py). Raster pictures OCR'd on pages with text are read as settings.FORMULAS says.
    model_unused: called as soon as it is known that the formula model will not be used (app/tasks.py gives the
    careful turn to the next conversion)."""
    if mode not in MODES:
        raise TaskError("invalid_options", mode)
    doc = open_pdf(pdf_path)
    total = doc.page_count
    kinds = [page_kind(page)["kind"] for page in doc]
    ocr_pages = sum(mode == "ocr-full" or (mode == "ocr" and kind == "scan") for kind in kinds)
    text_pages = sum(not (mode == "ocr-full" or kind == "scan") for kind in kinds)
    how = "model" if careful else None  # how formulas are read (None: settings.FORMULAS)
    # the formula pictures of text pages, read with the model at the teacher's request (budget shared by the pages)
    budget = (
        nativeread.FormulaBudget(MFR_MAX_FORMULAS, list(languages) or ["vie"])
        if careful and text_pages and MFR_MAX_FORMULAS > 0
        else None
    )
    if budget is not None and not mathocr.available():
        budget.failed = True  # the model went off since the page offered it (mathocr.broken()): the result says so
    with formulas_mode(how):
        # long scans without the slower models, so one file does not hold a worker for long (a small server)
        skip_formulas = ocr_pages > MFR_MAX_PAGES and formula_reader() == "model"
        scans_use_model = bool(ocr_pages) and not skip_formulas and formula_reader() == "model"
        uses_model = scans_use_model or (budget is not None and not budget.failed)
    if not uses_model and model_unused is not None:
        model_unused()
    # with the scanned pages read without the model, the text pages go first and the careful turn goes to the next
    # conversion after the last of them: the scanned pages' OCR (minutes on the server) does not hold it
    text = [not (mode == "ocr-full" or kind == "scan") for kind in kinds]
    order = sorted(range(total), key=lambda i: not text[i]) if uses_model and not scans_use_model else list(range(total))
    last_text = max((k for k, i in enumerate(order) if text[i]), default=-1)
    skip_vietocr = ocr_pages > VIETOCR_MAX_PAGES and vietocr.available()
    found: list[PageLayout | None] = [None] * total
    with formulas_mode(how), mathocr.skipped(skip_formulas), vietocr.skipped(skip_vietocr):
        for k, i in enumerate(order):
            page, kind = doc[i], kinds[i]
            progress(k, total)
            if mode == "ocr-full" or (mode == "ocr" and kind == "scan"):
                layout = ocr_page(page, languages)
            elif kind == "scan" or len(page.get_cdrawings()) > MAX_DRAWINGS:
                # (tens of thousands of vector paths would take the layout analysis hours)
                layout = image_page(page)
            else:
                if budget is not None:
                    budget.check = lambda k=k: progress(k, total)  # a cancelled conversion stops within the page
                with formulas_mode(None):  # its raster pictures read as by default (see above)
                    layout = native_page(page, languages if mode == "ocr" else None, strip=strip, budget=budget)
            found[i] = layout
            # the model is done with this file: the last text page read, or the budget spent or the model failed
            done = k == last_text or (budget is not None and (budget.left <= 0 or budget.failed or budget.ocr_seconds <= 0))
            if done and uses_model and not scans_use_model and model_unused is not None:
                model_unused()
        # the model may have failed to load or broken on a page (mathocr.broken()): the pages after it, and that page
        # again, were read the default way (app/ocr.py), so the result says so
        formulas = formula_reader()
    progress(total, total)
    pages: list[PageLayout] = [p for p in found if p is not None]

    sources = Counter(p.source for p in pages)
    words = sum(p.word_count for p in pages)
    if sources["ocr"] == total and not words and not any(p.pictures for p in pages):
        raise TaskError("no_text_found")
    removed_pictures, removed_lines = strip_watermarks(pages) if strip else (0, 0)
    header, footer = extract_header_footer(pages)
    out = Path(tempfile.mkstemp(suffix=".docx")[1])
    build_docx(pages, out, header, footer)
    return out, {
        "mode": mode,
        "used_ocr": words > 0,
        "ocr_languages": languages if words else [],
        "page_count": total,
        "pages_native": sources["native"],
        "pages_ocr": sources["ocr"],
        "pages_image": sources["image"],
        "ocr_regions": sum(p.ocr_regions for p in pages),
        "legacy_font_spans": sum(p.legacy_spans for p in pages),
        "pictures": sum(len(p.pictures) for p in pages),
        "header_footer": bool(header or footer),
        "removed_watermarks": removed_pictures,
        "removed_lines": removed_lines,
        "word_count": words or None,
        "avg_confidence": round(sum(p.conf_sum for p in pages) / words, 1) if words else None,
        "careful": careful,
        "formulas": formulas if ocr_pages else None,  # how formulas of scanned pages were read: model / scripts / off
        "formulas_skipped": skip_formulas,  # the model was wanted, the file has too many scanned pages for it
        "vietocr_skipped": skip_vietocr,
        # formula regions of scanned pages kept as pictures (fractions, roots: by default always, careful: unread)
        "formula_pictures": sum(p.formula_pictures for p in pages),
        "formulas_read": sum(p.formulas_read for p in pages),  # formula regions of scanned pages written from a reading
        # formulas of text pages: rebuilt as Word equations / kept as pictures (MathType, layouts not read)
        "equations": sum(p.equations for p in pages),
        "native_formula_pictures": sum(p.native_formula_pictures for p in pages),
        "native_picture_pages": sum(1 for p in pages if p.native_readable_pictures),  # pages a careful conversion OCRs
        # of them, those a careful conversion would try to read (not on turned pages, not larger than the model reads)
        "native_readable_pictures": sum(p.native_readable_pictures for p in pages),
        # careful: formula pictures of text pages read with the model / not trusted (kept) / past the budget (kept)
        "native_formulas_read": budget.read if budget else 0,
        "native_formulas_unsure": budget.unsure if budget else 0,
        "native_formulas_over": budget.over if budget else 0,
        "native_formulas_skipped": budget.skipped if budget else 0,  # never tried (turned page, too large, time spent)
        "native_formulas_failed": bool(budget and budget.failed),  # the model could not be used: pictures kept
        "max_formulas": MFR_MAX_FORMULAS,
    }
