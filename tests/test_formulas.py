"""Formula reading on OCR pages: regions (app/mathregions.py), placeholders and runs (app/ocr.py),
equations in the .docx (app/docx_builder.py). Run: python -m unittest discover -s tests"""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from lxml import etree  # noqa: E402

from app import mathocr, mathregions, ocr  # noqa: E402
from app.docx_builder import FORMULA_ALT, build_docx  # noqa: E402
from app.mathocr import MathDecision  # noqa: E402
from app.model import TAB, PageLayout, Paragraph, Run  # noqa: E402
from app.ocr import OBJ, FormulaPicture, Line, Word  # noqa: E402
from test_omml import M, W, schema_errors  # noqa: E402

EM = 30


def png(h=12, w=20) -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


def document_xml(path: Path):
    return etree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))


def build(runs: list[Run]) -> etree._Element:
    para = Paragraph(text="".join(r.text for r in runs), x0=72, y0=72, x1=300, y1=86, font_pt=12, runs=runs)
    page = PageLayout(width_pt=595, height_pt=842, elements=[para], source="ocr")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.docx"
        build_docx([page], out)
        from docx import Document

        Document(str(out))  # python-docx opens it
        return document_xml(out)


def body_parts(root) -> list[str]:
    """Children of the first body paragraph: "r:<text>", "math" or "drawing"."""
    p = root.find(f"{W}body/{W}p")
    out = []
    for node in p:
        if node.tag == M + "oMath":
            out.append("math")
        elif node.tag == W + "r":
            if node.find(f"{W}drawing") is not None:
                out.append("drawing")
            else:
                out.append("r:" + "".join(t.text or "" for t in node.iter(W + "t")))
    return out


class DocxMathTest(unittest.TestCase):
    def test_math_run_is_an_equation_between_runs(self):
        math = Run(OBJ, size=13, math=r"\frac{1}{2}", math_text="1/2", formula=True)
        runs = [Run("A.", bold=True), Run(" ", formula=True), math, Run(".", formula=True)]
        root = build(runs)
        self.assertEqual(body_parts(root), ["r:A.", "r: ", "math", "r:."])
        maths = list(root.iter(M + "oMath"))
        self.assertEqual(len(maths), 1)
        self.assertEqual(schema_errors(maths[0]), [])
        self.assertEqual({s.get(W + "val") for s in maths[0].iter(W + "sz")}, {"26"})  # the run's size
        self.assertIsNotNone(maths[0].find(f"{M}f"))

    def test_rejected_latex_falls_back_to_the_picture(self):
        picture = dict(image=png(), image_w=20, image_h=12, image_depth=3)
        runs = [Run("A. "), Run(OBJ, math=r"\frac{1}", math_text="1/", **picture)]
        root = build(runs)
        self.assertEqual(body_parts(root), ["r:A. ", "drawing"])
        self.assertEqual(list(root.iter(M + "oMath")), [])
        doc_pr = next(root.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"))
        self.assertEqual(doc_pr.get("descr"), FORMULA_ALT)

    def test_rejected_latex_without_picture_falls_back_to_text(self):
        root = build([Run("A. "), Run(OBJ, math=r"\frac{1}", math_text="1/2", size=12)])
        self.assertEqual(body_parts(root), ["r:A. ", "r:1/2"])
        root = build([Run(OBJ, math=r"\sqrt")])
        self.assertEqual(body_parts(root), [r"r:\sqrt"])


def word(x0, y0, x1, y1, text, conf=95.0, **kw) -> Word:
    return Word(x0, y0, x1, y1, conf, text, 0, EM, **kw)


TEXT = MathDecision("text", parts=[("6,42.10", None), ("-21", "super"), (" J.", None)], conf=0.97)
EQ = MathDecision("eq", latex=r"v=\sqrt{\frac{p}{\rho}}", tail=".", conf=0.95)


class PlaceholderRunsTest(unittest.TestCase):
    def line(self):
        holder_text = word(80, 40, 260, 82, ocr._placeholder_text(TEXT), math=TEXT)
        holder_eq = word(560, 30, 700, 95, ocr._placeholder_text(EQ), math=EQ)
        picture = FormulaPicture(png(), 30.0, 20.0, 4.0)
        holder_pic = word(1060, 30, 1160, 95, OBJ, math=picture)
        words = [word(20, 50, 60, 80, "A.", bold=True), holder_text, word(500, 50, 540, 80, "B.", bold=True), holder_eq,
                 word(1000, 50, 1040, 80, "C.", color="3366FF"), holder_pic, word(1170, 50, 1200, 80, "x")]
        return Line(1, 1, 80, EM, words)

    def test_runs_of_both_decision_kinds_and_a_picture(self):
        runs, pieces = ocr._runs([self.line()])
        self.assertEqual(len(pieces), 3)
        got = [(r.text, r.script, r.math, bool(r.image), r.bold, r.color) for r in runs]
        self.assertEqual(got, [
            ("A.", None, None, False, True, None),
            (" 6,42.10", None, None, False, False, None),
            ("-21", "super", None, False, False, None),
            (" J.", None, None, False, False, None),
            (TAB, None, None, False, False, None),
            ("B.", None, None, False, True, None),
            (" ", None, None, False, False, None),
            (OBJ, None, r"v=\sqrt{\frac{p}{\rho}}", False, False, None),
            (".", None, None, False, False, None),
            (TAB, None, None, False, False, None),
            ("C.", None, None, False, False, "3366FF"),
            (" ", None, None, False, False, None),
            (OBJ, None, None, True, False, None),
            (" x", None, None, False, False, None),  # the next word starts its own run
        ])
        self.assertTrue(all(r.formula for r in runs[1:4] + runs[6:9] + runs[11:13]))
        pic = runs[12]
        self.assertEqual((pic.image_w, pic.image_h, pic.image_depth), (30.0, 20.0, 4.0))

    def test_layout_keeps_formulas_regular(self):
        line = self.line()
        gray = np.full((300, 1300), 255, np.uint8)
        gray[50:80, 20:60] = 0  # ink for the labels only
        with mock.patch.object(ocr, "BOLD_SHARE", 0.0):  # a bold paragraph
            page = ocr.build_layout([line], [], gray, 300, ocr.pymupdf.Rect(0, 0, 312, 72))
        para = page.elements[0]
        self.assertTrue(para.bold)
        self.assertEqual(page.word_count, 7)
        self.assertIn(OBJ, para.text)
        for r in para.runs:
            if r.text != TAB:
                self.assertEqual(r.bold, not r.formula, r)
                self.assertEqual(r.size, para.font_pt)
        self.assertFalse(any(w.bold for w in line.words if w.math is not None))
        self.assertIsNone(ocr._line_width_size(line, 0.24))


def page_image():
    """A synthetic option row: "A." + a number with an exponent touching it, "B." + a fraction
    (numerator, bar, denominator apart), "C." + Vietnamese text; ink only, no letters."""
    gray = np.full((240, 1600), 255, np.uint8)
    gray[100:130, 20:60] = 0  # A.
    gray[100:130, 80:230] = 0  # 6,42.10
    gray[86:104, 232:262] = 0  # exponent, overlapping the number's top rows
    gray[100:130, 500:540] = 0  # B.
    gray[78:102, 575:640] = 0  # numerator
    gray[110:113, 560:660] = 0  # fraction bar (> 1em)
    gray[122:146, 580:640] = 0  # denominator, 9 px under the bar
    gray[100:130, 1000:1040] = 0  # C.
    gray[100:130, 1060:1200] = 0  # "Tăng"
    ink = (gray < 128).astype(np.uint8)
    return gray, ink


def page_lines():
    a, b, c = word(20, 100, 60, 130, "A."), word(500, 100, 540, 130, "B."), word(1000, 100, 1040, 130, "C.")
    num = [word(80, 100, 200, 130, "642.107'", conf=40), word(205, 100, 230, 130, "J.", conf=60)]
    junk = word(575, 100, 640, 113, "E", conf=30)
    tang = word(1060, 100, 1200, 130, "Tăng", conf=93)
    frac_top = word(575, 78, 640, 102, "n,T", conf=20)
    return [Line(1, 1, 130, EM, [a, *num, b, junk, c, tang]), Line(1, 2, 102, 15, [frac_top])]


class RulesDetectorTest(unittest.TestCase):
    def test_options(self):
        gray, ink = page_image()
        lines = page_lines()
        regions = mathregions.find(lines, gray, ink, None, 300, method="rules")
        self.assertEqual(len(regions), 2)  # "C. Tăng" is text
        first, second = regions
        self.assertEqual(first.box, (80, 86, 262, 130))  # grown over the exponent
        self.assertEqual((first.kind, first.ocr_text, first.em_px), ("option", "642.107' J.", EM))
        self.assertIs(first.line, lines[0])
        self.assertEqual([w.text for w in first.words], ["642.107'", "J."])
        self.assertEqual(second.box, (560, 78, 660, 146))  # numerator and denominator across the bar
        self.assertEqual(second.kind, "tall")

    def test_lone_label_and_lowercase_label(self):
        gray, ink = page_image()
        lines = page_lines()
        lines[0].words = [w for w in lines[0].words if w.text not in ("642.107'", "J.", "E")]
        lines[0].words[1].text = "b."  # beside a real label "A."
        regions = mathregions.find(lines, gray, ink, None, 300, method="rules")
        self.assertEqual([r.box for r in regions], [(80, 86, 262, 130), (560, 78, 660, 146)])
        self.assertEqual(regions[0].words, [])  # nothing read on the label's row

    def test_stops_before_the_next_option_and_drops_unsettled(self):
        gray, ink = page_image()
        gray[100:130, 230:480] = 0  # ink running on to 20 px before "B."
        ink = (gray < 128).astype(np.uint8)
        regions = mathregions.find(page_lines(), gray, ink, None, 300, method="rules")
        self.assertEqual([r.kind for r in regions], ["tall"])  # the first one ran into B.

    def test_grow_takes_an_overline_and_a_denominator(self):
        ink = np.zeros((200, 400), np.uint8)
        ink[40:43, 110:140] = 1  # bar of a mean value, 5 px over the numerator
        ink[48:70, 110:140] = 1  # numerator
        ink[75:78, 100:170] = 1  # fraction bar
        ink[78, 100:110] = 1  # its soft lower edge
        ink[86:110, 105:165] = 1  # denominator past a blank gap
        box, settled = mathregions.grow(ink, (100, 74, 170, 79), EM, left_limit=90)
        self.assertEqual((box, settled), ((100, 40, 170, 110), True))
        ink[40:43] = 0
        ink[25:43, 60:300] = 1  # a line of text above instead: left out
        box, settled = mathregions.grow(ink, (100, 74, 170, 79), EM, left_limit=90)
        self.assertEqual((box, settled), ((100, 48, 170, 110), True))
        box, settled = mathregions.grow(ink, (100, 74, 170, 79), EM, left_limit=90, top_limit=50)
        self.assertFalse(settled)  # still on ink at its limit

    def test_off_unknown_and_overlaps(self):
        gray, ink = page_image()
        self.assertEqual(mathregions.find(page_lines(), gray, ink, None, 300, method="off"), [])
        with self.assertLogs("pdf2word.mathregions", "WARNING"):
            self.assertEqual(mathregions.find(page_lines(), gray, ink, None, 300, method="nope"), [])
        R = mathregions.MathRegion
        a, b, c, d = R((0, 0, 10, 10)), R((0, 0, 10, 10)), R((5, 5, 20, 20)), R((30, 0, 40, 10))
        self.assertEqual(mathregions.dedupe([a, b, d]), [a, d])
        self.assertEqual(mathregions.dedupe([a, c, d]), [d])

    def test_detector_registry(self):
        gray, ink = page_image()
        lines = page_lines()
        seen = []

        def mine(lines_, gray_, ink_, rgb_, dpi_):
            seen.append(dpi_)
            return [mathregions.MathRegion((80, 86, 262, 130), [lines_[0].words[1]], "inline", "x", EM, lines_[0])]

        with mock.patch.dict(mathregions.DETECTORS, {"mine": mine}), \
                mock.patch.object(mathregions, "MATH_DETECT", "mine"):
            regions = mathregions.find(lines, gray, ink, None, 300)
        self.assertEqual((seen, [r.kind for r in regions]), ([300], ["inline"]))


class FakeReader:
    def __init__(self, answer):
        self.answer = answer

    def read(self, images):
        return [self.answer(im) for im in images]


class ReadFormulasTest(unittest.TestCase):
    def setUp(self):  # the formula model's path (a careful conversion)
        patch = mock.patch.object(ocr, "FORMULAS", "model")
        patch.start()
        self.addCleanup(patch.stop)

    def run_page(self, answer):
        gray, ink = page_image()
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        lines = page_lines()
        with mock.patch.object(mathocr, "available", return_value=True), \
                mock.patch.object(mathocr, "get_reader", return_value=FakeReader(answer)):
            placed = ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        return placed, lines

    def test_readings_replace_words(self):
        def answer(im):  # crops are cut to the ink: the number is 182 px wide, the fraction 100
            if im.shape[1] < 150:
                return r"\frac { n _ { 1 } } { n _ { 2 } }", 0.95
            return r"6 , 4 2 . 1 0 ^ { - 2 1 } \mathrm { J }", 0.97

        placed, lines = self.run_page(answer)
        self.assertEqual(placed, 2)
        self.assertEqual(len(lines), 1)  # the numerator's line is gone with its word
        texts = [w.text for w in lines[0].words]
        self.assertEqual(texts, ["A.", "6,42.10-21 J.", "B.", OBJ, "C.", "Tăng"])
        first, second = lines[0].words[1], lines[0].words[3]
        self.assertEqual(first.math.parts, [("6,42.10", None), ("-21", "super"), (" J.", None)])
        self.assertEqual((first.x0, first.y0, first.x1, first.y1, first.conf), (80, 86, 262, 130, 97.0))
        self.assertEqual((second.math.kind, second.math.latex), ("eq", r"\frac{n_{1}}{n_{2}}"))
        runs, _ = ocr._runs(lines)
        self.assertEqual(sum(1 for r in runs if r.math), 1)

    def test_unread_regions(self):
        placed, lines = self.run_page(lambda im: (r"\theta", 0.99))
        self.assertEqual(placed, 1)  # the tall fraction becomes a picture, the number keeps its words
        texts = [w.text for w in lines[0].words]
        self.assertEqual(texts, ["A.", "642.107'", "J.", "B.", OBJ, "C.", "Tăng"])
        holder = lines[0].words[4]
        pic = holder.math
        self.assertIsInstance(pic, FormulaPicture)
        self.assertEqual(holder.conf, 25.0)  # the replaced words' mean
        s = 72 / 300
        self.assertAlmostEqual(pic.w_pt, 100 * s)
        self.assertAlmostEqual(pic.h_pt, 68 * s)
        self.assertAlmostEqual(pic.depth_pt, 16 * s)  # 146 - baseline 130
        img = cv2.imdecode(np.frombuffer(pic.png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (68, 100))

    def test_reading_without_vietnamese_marks_is_refused(self):
        gray = np.full((200, 900), 255, np.uint8)
        for x0, x1 in ((20, 60), (80, 100), (110, 150), (160, 190), (500, 540), (560, 600)):
            gray[100:130, x0:x1] = 0
        ink = (gray < 128).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        words = [word(20, 100, 60, 130, "A."), word(80, 100, 100, 130, "5"), word(110, 100, 150, 130, "và", conf=60),
                 word(160, 100, 190, 130, "3."), word(500, 100, 540, 130, "B."), word(560, 100, 600, 130, "4.")]
        lines = [Line(1, 1, 130, EM, words)]
        reader = FakeReader(lambda im: (r"5 \mathrm { v a } 3", 0.99) if im.shape[1] > 60 else ("4", 0.99))
        self.assertIsNotNone(mathocr.decide(r"5 \mathrm { v a } 3", 0.99, "5 và 3."))  # the model's rules let it pass
        with mock.patch.object(mathocr, "available", return_value=True), \
                mock.patch.object(mathocr, "get_reader", return_value=reader):
            placed = ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        self.assertEqual(placed, 1)  # "4." only
        self.assertEqual([w.text for w in lines[0].words], ["A.", "5", "và", "3.", "B.", "4."])
        self.assertIsNotNone(lines[0].words[5].math)

    def test_nothing_when_off_or_in_tables(self):
        gray, ink = page_image()
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        lines = page_lines()
        with mock.patch.object(ocr, "FORMULAS", "off"), mock.patch.object(mathocr, "available", return_value=True):
            self.assertEqual(ocr.read_formulas(lines, [], gray, ink, rgb, 300), 0)
        grid = ocr._Grid(0, 0, 1600, 240, [0, 240], [0, 1600])
        with mock.patch.object(mathocr, "available", return_value=True), \
                mock.patch.object(mathocr, "get_reader", return_value=FakeReader(lambda im: ("1", 0.99))):
            self.assertEqual(ocr.read_formulas(lines, [grid], gray, ink, rgb, 300), 0)
        self.assertEqual([len(ln.words) for ln in lines], [7, 1])


if __name__ == "__main__":
    unittest.main()
