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

    def frac(self, num, den, size=S):
        """A fraction at the current position: returns its bar."""
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
        # only a fraction: the equation takes the size of its line (the caller's)
        ln = Line()
        bar = ln.frac("1", "110", size=7)
        self.assertEqual([(p.text, p.size) for p in build(ln.glyphs, [bar], [])], [(r"\frac{1}{110}", 0.0)])

    def test_fractions_nested_and_side_by_side(self):
        ln = Line().put("𝑥").put("=", gap=2)
        bars = [ln.frac("1", "2")]
        ln.x += 1
        bars.append(ln.frac("𝑣", "𝑢"))
        self.assertEqual(latex(build(ln.glyphs, bars, [])), [r"x=\frac{1}{2}\frac{v}{u}"])
        # a fraction in a numerator: the shorter bar inside the longer one
        ln = Line()
        outer = Bar(50, 70, BASE - 2.5)
        inner = Bar(52, 57, BASE - 9.5)
        for c, x, oy in (("1", 52, BASE - 12.5), ("2", 52, BASE - 4.5), ("𝑚", 58, BASE - 7.5), ("𝑀", 58, BASE + 7)):
            ln.glyphs.append(Glyph(c, x, oy - 0.78 * 7, x + 5, oy + 0.22 * 7, oy, 7, font="Cambria Math"))
        self.assertEqual(latex(build(ln.glyphs, [outer, inner], [])), [r"\frac{\frac{1}{2}m}{M}"])

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


if __name__ == "__main__":
    unittest.main()
