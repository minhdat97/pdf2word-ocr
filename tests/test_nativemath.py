"""Word equations rebuilt from the text layer of native PDFs: app/nativemath.py (glyph layout -> LaTeX)
and its use in app/native.py and the .docx. Run: python -m unittest discover -s tests"""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402
from lxml import etree  # noqa: E402

from app import native, nativemath  # noqa: E402
from app.docx_builder import build_docx  # noqa: E402
from app.nativemath import Bar, Glyph, build  # noqa: E402

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
S = 10.0  # line size in the synthetic layouts
BASE = 100.0


class Line:
    """Glyphs set left to right like Word's PDF export: boxes 0.78 em over and 0.22 em under the baseline."""

    def __init__(self, x=50.0, base=BASE):
        self.x, self.base, self.glyphs = x, base, []

    def put(self, text, size=S, dy=0.0, width=None, gap=0.0, **style):
        self.x += gap
        for c in text:
            w = width if width is not None else (0 if c == nativemath.ARROW else 0.5 * size)
            oy = self.base + dy
            style.setdefault("font", "Cambria Math")
            style.setdefault("math_font", style["font"] == "Cambria Math")
            self.glyphs.append(Glyph(c, self.x, oy - 0.78 * size, self.x + w, oy + 0.22 * size, oy, size, **style))
            self.x += w
        return self

    def sup(self, text, **kw):
        return self.put(text, size=0.7 * S, dy=-0.35 * S, **kw)

    def sub(self, text, **kw):
        return self.put(text, size=0.7 * S, dy=0.2 * S, **kw)

    def frac(self, num, den, size=0.72 * S):
        """A fraction at the current position (Word sets an inline fraction's parts at 0.72 of the
        line; a display fraction's at the line's size): returns its bar."""
        width = 0.5 * size * max(len(num), len(den))
        x0, axis = self.x, self.base - 0.25 * S
        for text, dy in ((num, -0.6 * S), (den, 0.7 * S)):
            off = (width - 0.5 * size * len(text)) / 2
            for n, c in enumerate(text):
                gx, oy = x0 + off + n * 0.5 * size, axis + dy
                self.glyphs.append(Glyph(c, gx, oy - 0.78 * size, gx + 0.5 * size, oy + 0.22 * size, oy, size,
                                         font="Cambria Math"))
        self.x += width
        return Bar(x0, x0 + width, axis)


def latex(pieces):
    return [p.text for p in pieces if p.kind == "math"]


class LayoutTest(unittest.TestCase):
    def test_scripts_units_and_typed_spaces(self):
        ln = Line().put("𝑊").sub("𝑑").put("=", gap=2).put("10", gap=2).sup("21").put("kg").put("=", gap=2)
        ln.put("6,06", gap=2).put("J", gap=2.2)
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"W_{d}={10}^{21}\mathrm{kg}=6,06\ \mathrm{J}"])
        # a decimal number is the base of its script; Word spaces after the dot and around fractions itself
        ln = Line().put("1.", width=4).put("4,1").sup("2").put("−", gap=2).put("𝑠𝑖𝑛", gap=2).put("25", gap=1.9).sup("0")
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"1.{4,1}^{2}-sin\ {25}^{0}"])
        ln = Line().put("=")
        bar = ln.frac("1", "2")
        ln.put("𝑚", gap=2)
        self.assertEqual(latex(build(ln.glyphs, [bar], [])), [r"=\frac{1}{2}m"])
        # only a fraction: at its parts' size, its line's size unknown here (native.py settles it)
        ln = Line()
        bar = ln.frac("1", "110", size=7)
        self.assertEqual([(p.text, p.size, p.frac_only) for p in build(ln.glyphs, [bar], [])], [(r"\frac{1}{110}", 7, True)])

    def test_fractions_nested_and_side_by_side(self):
        ln = Line().put("𝑥").put("=", gap=2)
        bars = [ln.frac("1", "2")]
        ln.x += 1
        bars.append(ln.frac("𝑣", "𝑢"))
        self.assertEqual(latex(build(ln.glyphs, bars, [])), [r"x=\frac{1}{2}\frac{v}{u}"])
        # a fraction in a numerator: the shorter bar inside the longer one, every part centred on its bar
        ln = Line()
        outer = Bar(50, 62, BASE - 2.5)
        inner = Bar(51, 56, BASE - 9.5)
        for c, x, oy, sz in (("1", 51, BASE - 12.5, 5), ("2", 51, BASE - 4.5, 5), ("𝑚", 56, BASE - 7.5, 7), ("𝑀", 53.5, BASE + 7, 7)):
            ln.glyphs.append(Glyph(c, x, oy - 0.78 * sz, x + 5, oy + 0.22 * sz, oy, sz, font="Cambria Math"))
        pieces = build(ln.glyphs, [outer, inner], [])
        self.assertEqual([(p.text, p.size, p.frac_only) for p in pieces], [(r"\frac{\frac{1}{2}m}{M}", 7, True)])
        # parts as large as the line: Word's display equation (native.py keeps it alone in its paragraph)
        ln = Line().put("𝑣").put("=", gap=2)
        bar = ln.frac("𝑠", "𝑡", size=S)
        self.assertEqual([(p.text, p.display) for p in build(ln.glyphs, [bar], [])], [(r"v=\frac{s}{t}", True)])
        # a rule under a line of formulas is not a fraction bar: its parts are not centred on it, or it is too long
        ln = Line().put("𝑥").put("=").put("2")
        ln.glyphs += Line(base=BASE + 14).put("𝑦").put("=").put("3").glyphs
        self.assertIsNone(build(ln.glyphs, [Bar(50, 110, BASE + 4)], []))

    def test_vector_arrows_primes_and_greek(self):
        ln = Line().put("𝑚").sub("1").put("𝑣").sub("1")
        v = next(g for g in ln.glyphs if g.c == "𝑣")
        ln.glyphs.append(Glyph(nativemath.ARROW, v.x1, v.y0, v.x1, v.y1, v.oy, S, font="Cambria Math"))
        ln.put("=", gap=2).put("𝑉", gap=2)
        V = ln.glyphs[-1]
        for _ in range(2):  # Word draws the arrow over a capital in two pieces
            ln.glyphs.append(Glyph(nativemath.ARROW, V.x1, V.y0, V.x1, V.y1, V.oy, S, font="Cambria Math"))
        ln.put("+", gap=2).put("𝑡", gap=2).sup("′").put("𝛼").put("∆", gap=1).put("𝑇")
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"m_{1}\vec{v}_{1}=\vec{V}+t'\alpha∆T"])

    def test_prime_then_subscript_greek_and_symbols(self):
        ln = Line().put("𝑣").sup("′").sub("1", gap=0.3).put("=", gap=2).put("2")
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"v_{1}'=2"])
        # ε φ (Word's) and ϵ ϕ are different letters; ∞, ∂ and an upright Ω (ohms) are read too
        ln = Line().put("𝜀").put("𝜑", gap=1).put("𝜖", gap=1).put("𝜙", gap=1).put("∞", gap=1).put("∂", gap=1)
        ln.put("5", gap=2).put("Ω")
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"\varepsilon\varphi\epsilon\phi\infty\partial\ 5\mathrm{Ω}"])

    def test_script_order_does_not_matter(self):
        ln = Line().put("𝑚").sub("𝑡ℎ").put("=", gap=2).put("1")
        ln.glyphs[1], ln.glyphs[2] = ln.glyphs[2], ln.glyphs[1]  # the text layer lists "h" before "t"
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"m_{th}=1"])

    def test_degree_zero_without_base(self):
        # "t( ⁰C)": after a bracket; "80,6 ⁰F": set apart from the number
        ln = Line().put("𝑡").put("(").sup("0", gap=3).put("𝐶").put(")").put("=", gap=2).put("80,6", gap=2)
        ln.sup("0", gap=3).put("𝐹")
        self.assertEqual(latex(build(ln.glyphs, [], [])), [r"t({}^{0}C)=80,6{}^{0}F"])

    def test_words_in_an_equation(self):
        ln = Line().put("∎").put("Hiệu", gap=2.3).put("suất", gap=2.3).put("=", gap=2)
        bar = ln.frac("𝑄", "𝐴")
        self.assertEqual(latex(build(ln.glyphs, [bar], [])), [r"∎\ \text{Hiệu suất}=\frac{Q}{A}"])

    def test_dash_label_and_trailing_text(self):
        ln = Line().put("-").put("∆", gap=2.3).put("𝑈").put("=", gap=2).put("10", gap=2)
        pieces = build(ln.glyphs, [], [])
        self.assertEqual([(p.kind, p.text) for p in pieces], [("text", "- "), ("math", "∆U=10")])
        ln = Line().put("𝒃", color="0000FF").put(")", color="0000FF").put("𝑡", gap=2.3).put("=", gap=2).put("5")
        ln.put(".", font="Times New Roman")
        pieces = build(ln.glyphs, [], [])
        self.assertEqual([(p.kind, p.text) for p in pieces], [("text", "b"), ("text", ") "), ("math", "t=5"), ("text", ".")])
        self.assertTrue(pieces[0].bold and pieces[0].italic)
        self.assertFalse(pieces[1].bold or pieces[1].italic)
        self.assertEqual(pieces[1].color, "0000FF")
        self.assertIsNone(pieces[2].color)
        self.assertEqual(pieces[3].font, "Times New Roman")

    def test_drawn_absolute_value_bars(self):
        ln = Line().put("𝐴", gap=1).put("=", gap=2)
        bars = [(50, BASE - 8, 50.4, BASE + 2), (55.5, BASE - 8, 55.9, BASE + 2)]
        self.assertEqual(latex(build(ln.glyphs, [], bars)), [r"|A|="])


class RefusalTest(unittest.TestCase):
    def assertPicture(self, glyphs, bars=(), uprights=(), others=0):
        self.assertIsNone(build(list(glyphs), list(bars), list(uprights), others),
                          nativemath.why(list(glyphs), list(bars), list(uprights), others))

    def test_unknown_strokes_and_bars(self):
        ln = Line().put("𝑥").put("=").put("2")
        self.assertPicture(ln.glyphs, others=1)  # a radical, a brace, an underline box
        self.assertPicture(ln.glyphs, [Bar(50, 55, BASE - 9)])  # an overline: nothing under it

    def test_two_lines_and_colours(self):
        ln = Line().put("𝑥").put("=").put("2")
        ln.glyphs += Line(base=BASE + 14).put("𝑦").put("=").put("3").glyphs
        self.assertPicture(ln.glyphs)
        self.assertPicture(Line().put("𝑥", color="FF0000").put("=").put("2").glyphs)

    def test_glyphs_it_cannot_read(self):
        self.assertPicture(Line().put("𝑅").put("=").put("!\"").glyphs)  # unmapped glyph codes
        self.assertPicture(Line().put("𝑥").put("⎛").glyphs)  # a piece of a tall bracket
        self.assertPicture(Line().put("𝓐").put("=").put("1").glyphs)  # script letters: not mapped
        lone = Line().put("=")
        lone.glyphs.append(Glyph(nativemath.ARROW, 80, 92, 80, 102, BASE, S, font="Cambria Math"))
        self.assertPicture(lone.glyphs)  # an arrow over nothing

    def test_layouts_it_does_not_read(self):
        # a fraction in an exponent: 2^(−t/T) (its bar sits high over the line's maths axis)
        ln = Line().put("𝑁").put("=", gap=2).put("2", gap=2).sup("−")
        bar = Bar(ln.x, ln.x + 4, BASE - 7.5)
        for c, oy in (("𝑡", BASE - 9.5), ("𝑇", BASE - 4)):
            ln.glyphs.append(Glyph(c, ln.x, oy - 0.78 * 6, ln.x + 3.5, oy + 0.22 * 6, oy, 6, font="Cambria Math"))
        self.assertPicture(ln.glyphs, [bar])
        # limits under "lim" (a script under a symbol), a condition over an arrow
        ln = Line().put("lim")
        ln.glyphs.append(Glyph("𝑥", 52, BASE + 2, 55.5, BASE + 9, BASE + 7, 7, font="Cambria Math"))
        ln.put("𝑓", gap=2)
        self.assertPicture(ln.glyphs)
        ln = Line().put("𝐴").put("→", gap=2, width=10)
        ln.glyphs.append(Glyph("𝑡", 58, BASE - 14, 61.5, BASE - 7, BASE - 9, 7, font="Cambria Math"))
        ln.put("𝐵", gap=2)
        self.assertPicture(ln.glyphs)
        # a coefficient before a nuclide: the stack ends together, so it is the nuclide's
        ln = Line().put("2").put("1", size=7, dy=-3.5).put("𝑛")
        ln.glyphs.insert(2, Glyph("0", 55, BASE - 3.4, 58.5, BASE + 3.6, BASE + 2, 7, font="Cambria Math"))
        self.assertPicture(ln.glyphs)
        # a glyph level with a fraction bar belongs to neither part
        ln = Line().put("𝑥").put("=", gap=2)
        bar = ln.frac("1", "2")
        ln.glyphs.append(Glyph("|", bar.x1 - 1, bar.y - 4, bar.x1, bar.y + 4, bar.y + 1.5, S, font="Cambria Math"))
        self.assertPicture(ln.glyphs, [bar])

    def test_text_a_docx_cannot_hold(self):
        ln = Line().put("\uffff", font="Times New Roman").put("𝑥", gap=2.3).put("=", gap=2).put("1")
        self.assertPicture(ln.glyphs)

    def test_mathtype_and_text_fonts(self):
        self.assertPicture(Line().put("2", font="Times New Roman").put("π", font="Symbol").glyphs)
        self.assertPicture(Line().put("𝑥").put("+", font="Times New Roman").put("1").glyphs)


STIX = "/System/Library/Fonts/Supplemental/STIXTwoMath.otf"


@unittest.skipUnless(Path(STIX).exists(), "needs macOS's STIX Two Math")
class NativePageTest(unittest.TestCase):
    """A page set like Word's export of "Biet W_d = ½mv²", read by native_page() and written to .docx."""

    def page(self, empty_map=False):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=200)
        page.insert_text((40, 100), "Biet", fontsize=11, fontname="Times-Roman")
        page.insert_font(fontname="stix", fontfile=STIX)
        x = 70.0

        def put(text, size=11, dy=0.0, at=None):
            nonlocal x
            px = x if at is None else at
            page.insert_text((px, 100 + dy), text, fontsize=size, fontname="stix")
            width = pymupdf.Font(fontfile=STIX).text_length(text, fontsize=size)
            if at is None:
                x += width
            return width

        put("𝑊")
        put("𝑑", size=8, dy=2.2)
        x += 3
        put("=")
        x += 3
        w = max(put("1", size=8, dy=-6, at=x), 5)
        put("2", size=8, dy=6, at=x)
        page.draw_rect(pymupdf.Rect(x, 97.2, x + w, 97.8), color=None, fill=(0, 0, 0))
        x += w + 0.5
        put("𝑚𝑣")
        put("2", size=8, dy=-4)
        page.insert_text((x + 6, 100), "la dong nang.", fontsize=11, fontname="Times-Roman")
        # PyMuPDF writes no usable map for letters past U+FFFF: write Word's (glyph id -> UTF-16)
        font = pymupdf.Font(fontfile=STIX)
        pairs = "".join(f"<{font.has_glyph(ord(c)):04X}><{c.encode('utf-16-be').hex().upper()}>\n" for c in "𝑊𝑑=12𝑚𝑣")
        body = "" if empty_map else f"7 beginbfchar\n{pairs}endbfchar\n"
        cmap = f"begincmap\n1 begincodespacerange\n<0000><FFFF>\nendcodespacerange\n{body}endcmap\n".encode()
        for xref, *_ in page.get_fonts(full=True):
            kind, tu = doc.xref_get_key(xref, "ToUnicode")
            if kind == "xref":
                doc.update_stream(int(tu.split()[0]), cmap)
        doc = pymupdf.open("pdf", doc.tobytes())  # read back as a file: fonts load with the new map
        return doc, doc[0]

    def test_equation_run_and_docx(self):
        _doc, page = self.page()
        layout = native.native_page(page)
        runs = [r for e in layout.elements for r in getattr(e, "runs", [])]
        maths = [r.math for r in runs if r.math]
        self.assertEqual(maths, [r"W_{d}=\frac{1}{2}mv^{2}"])
        self.assertFalse(any(r.image for r in runs))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx([layout], out)
            root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
        omath = root.findall(f".//{M}oMath")
        self.assertEqual(len(omath), 1)
        self.assertEqual(len(omath[0].findall(f".//{M}f")), 1)
        self.assertIn("Biet", "".join(t.text or "" for t in root.iter(W + "t")))

    def test_font_without_a_text_map_stays_a_picture(self):
        _doc, page = self.page(empty_map=True)
        layout = native.native_page(page)
        runs = [r for e in layout.elements for r in getattr(e, "runs", [])]
        self.assertFalse(any(r.math for r in runs))
        self.assertTrue(any(r.image for r in runs))


class SplitLinesTest(unittest.TestCase):
    """native._split_lines: a formula group spanning two printed lines becomes two formulas only when
    nothing but closeness joins them."""

    @staticmethod
    def glyph(x, base, size=10, kind=2):
        c = {"c": "x", "origin": (x, base), "bbox": (x, base - 0.78 * size, x + 5, base + 0.22 * size)}
        return [pymupdf.Rect(c["bbox"]), kind, size, None, c]

    def test_two_lines(self):
        nodes = [self.glyph(50, 100), self.glyph(56, 103, 7), self.glyph(50, 114), self.glyph(56, 114)]
        parts = native._split_lines(list(range(4)), nodes, [])
        self.assertEqual(sorted(map(sorted, parts)), [[0, 1], [2, 3]])

    def test_kept_whole(self):
        # rows closer than 0.9 em (a system), a row without a formula glyph, a stroke between the lines
        nodes = [self.glyph(50, 100), self.glyph(50, 108)]
        self.assertEqual(native._split_lines([0, 1], nodes, []), [[0, 1]])
        nodes = [self.glyph(50, 100), self.glyph(50, 114, kind=1)]
        self.assertEqual(native._split_lines([0, 1], nodes, []), [[0, 1]])
        nodes = [self.glyph(50, 100), self.glyph(50, 114), [pymupdf.Rect(48, 103, 60, 104), 0, None, 7, None, False]]
        self.assertEqual(native._split_lines([0, 1, 2], nodes, []), [[0, 1, 2]])

    def test_fraction_keeps_its_parts(self):
        # line 1: x = v²/2 (bar node 4 with parts 1, 2; the exponent 3 sits in its box); line 2 under it
        nodes = [self.glyph(50, 100), self.glyph(60, 95), self.glyph(60, 109), self.glyph(66, 89, 7),
                 [pymupdf.Rect(58, 97, 70, 98), 0, None, 3, None, False], self.glyph(50, 122), self.glyph(56, 122)]
        parts = native._split_lines(list(range(7)), nodes, [(4, [1, 2])])
        self.assertEqual(sorted(map(sorted, parts)), [[0, 1, 2, 3, 4], [5, 6]])


class BracketInkTest(unittest.TestCase):
    """native._Extractor._ink_height: a stretched bracket has the text box of a plain one, so its drawn
    height decides whether it encloses rows outside the formula (a matrix, a system)."""

    def test_heights(self):
        doc = pymupdf.open()
        page = doc.new_page(width=300, height=200)
        page.insert_text((50, 100), "(", fontsize=11, fontname="Times-Roman")
        page.draw_line((120, 70), (120, 130), width=0.8)  # a bracket three rows tall
        ex = native._Extractor(page, None)
        plain = Glyph("(", 50, 100 - 0.78 * 11, 54, 100 + 0.22 * 11, 100, 11)
        tall = Glyph("(", 119, 100 - 0.78 * 11, 121.5, 100 + 0.22 * 11, 100, 11)
        self.assertLess(ex._ink_height(plain), 13)
        self.assertGreater(ex._ink_height(tall), 55)
        self.assertIsNone(ex._ink_height(Glyph("(", 200, 92, 204, 102, 100, 11)))  # no ink: unsure


def _unmapped(page):
    fonts = native._math_fonts(page)
    return None if fonts is None else fonts[0]


class UnmappedGlyphsTest(unittest.TestCase):
    """native._math_fonts: glyph codes a Unicode math font gives without a real mapping, read
    within bounds whatever the file holds (the map is the uploader's)."""

    def page(self, cmap: bytes | None = b"", first=33, last=40, stream=True):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 100), "x", fontname="helv")
        xref = next(x[0] for x in page.get_fonts(full=True))
        doc.xref_set_key(xref, "BaseFont", "/AAAAAB+CambriaMath")
        doc.xref_set_key(xref, "Subtype", "/TrueType")
        doc.xref_set_key(xref, "Encoding", "null")
        doc.xref_set_key(xref, "FirstChar", str(first))
        doc.xref_set_key(xref, "LastChar", str(last))
        if cmap is not None:
            if stream:
                ref = doc.get_new_xref()
                doc.update_object(ref, "<<>>")
                doc.update_stream(ref, cmap)
            else:
                ref = doc.get_new_xref()
                doc.update_object(ref, "<< /Foo 1 >>")
            doc.xref_set_key(xref, "ToUnicode", f"{ref} 0 R")
        doc = pymupdf.open("pdf", doc.tobytes())
        return doc, doc[0]

    def test_maps(self):
        _doc, page = self.page(b"begincmap 1 beginbfchar <21><0052> endbfchar 1 beginbfrange <22><23><0041> endbfrange endcmap")
        self.assertEqual(_unmapped(page), set("$%&'("))
        _doc, page = self.page(b"begincmap endcmap")  # an empty map: every code is raw
        self.assertEqual(_unmapped(page), set("!\"#$%&'("))
        _doc, page = self.page(None)  # no map, no standard encoding
        self.assertEqual(_unmapped(page), set("!\"#$%&'("))

    def test_hostile_maps(self):
        import time
        t = time.perf_counter()
        _doc, page = self.page(b"begincmap 1 beginbfrange <00000000><FFFFFFFF><0041> endbfrange endcmap", 0, 10 ** 9)
        self.assertEqual(_unmapped(page), set())  # one-byte codes, all mapped
        _doc, page = self.page(b"begincmap " + b"beginbfchar " * 5000 + b"endcmap")
        self.assertEqual(_unmapped(page), set("!\"#$%&'("))
        self.assertLess(time.perf_counter() - t, 2)
        _doc, page = self.page(b"begincmap endcmap", stream=False)  # not a stream: pictures
        self.assertIsNone(_unmapped(page))
        _doc, page = self.page(b"%" * (4 * native.CMAP_MAX + 1))  # too large to read: formulas stay pictures
        self.assertIsNone(_unmapped(page))


if __name__ == "__main__":
    unittest.main()
