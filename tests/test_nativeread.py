"""Formula pictures of text pages read with the formula model ("Chuyển thành công thức sửa được", app/nativeread.py).
No real model or Tesseract: a fake reader answers every crop, the Tesseract words of the page are given."""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pymupdf
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import mathocr, native, nativeread  # noqa: E402
from app.docx_builder import build_docx  # noqa: E402
from app.model import Cell, Paragraph, Run  # noqa: E402
from app.ocr import Word  # noqa: E402
from tests.test_nativemath import f_over, mathtype_page  # noqa: E402

M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


class FakeReader:
    """Answers every crop with the next of `answers` (cycling), confidence `conf`."""

    def __init__(self, *answers, conf=0.95):
        self.answers, self.conf, self.n = answers, conf, 0

    def read(self, images):
        out = []
        for _ in images:
            out.append((self.answers[self.n % len(self.answers)], self.conf))
            self.n += 1
        return out


def unreadable_page():
    """A MathType object A/A2 cannot rebuild (a bracket piece where λ was): native keeps it as one picture."""
    return mathtype_page(lambda p: f_over(p, letter="\xe6"))


def read_page(page, reader, words=(), budget=None):
    budget = budget or nativeread.FormulaBudget(200, ["vie"])
    with mock.patch.object(nativeread, "_reader", return_value=reader), \
            mock.patch.object(nativeread, "_words", return_value=list(words)):
        layout = native.native_page(page, budget=budget)
    runs = [r for e in layout.elements for r in getattr(e, "runs", [])]
    return layout, runs, budget


def docx_maths(layout):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.docx"
        build_docx([layout], out)
        root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
    return root


class NativePageTest(unittest.TestCase):
    def test_without_budget_the_picture_stays(self):
        _doc, page = unreadable_page()
        layout = native.native_page(page)
        self.assertEqual((layout.equations, layout.native_formula_pictures, layout.native_formulas_read), (0, 1, 0))

    def test_2d_reading_becomes_an_equation_keeping_its_picture(self):
        _doc, page = unreadable_page()
        layout, runs, budget = read_page(page, FakeReader(r"f=\frac{v}{\lambda}"))
        maths = [r for r in runs if r.math]
        self.assertEqual([r.math for r in maths], [r"f=\frac{v}{\lambda}"])
        self.assertTrue(maths[0].image)  # written as the picture should Word's equation fail (docx_builder)
        self.assertEqual(maths[0].font, "Times")  # the text's font around it: an inline equation (_keep_inline)
        self.assertEqual((layout.equations, layout.native_formula_pictures, layout.native_formulas_read), (1, 0, 1))
        self.assertEqual((budget.read_eq, budget.read_text, budget.unsure, budget.left), (1, 0, 0, 199))
        root = docx_maths(layout)
        self.assertEqual(len(root.findall(f".//{M}oMath")), 1)
        self.assertEqual(len(root.findall(f".//{M}oMathPara")), 0)
        self.assertEqual(root.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"), [])

    def test_one_line_reading_becomes_text_with_scripts(self):
        _doc, page = unreadable_page()
        layout, runs, _budget = read_page(page, FakeReader("x_{A}"))
        self.assertFalse(any(r.image or r.math for r in runs))
        i = next(i for i, r in enumerate(runs) if r.text == "x")
        self.assertEqual((runs[i + 1].text, runs[i + 1].script), ("A", "sub"))
        self.assertEqual(runs[i].font, "Times")
        self.assertEqual((layout.equations, layout.native_formula_pictures, layout.native_formulas_read), (0, 0, 1))
        para = next(e for e in layout.elements if isinstance(e, Paragraph) and any(r is runs[i] for r in e.runs))
        self.assertEqual(para.text, "".join(r.text for r in para.runs))  # .text follows the new runs
        self.assertIn("xA", para.text)

    def test_readings_that_disagree_keep_the_picture(self):
        _doc, page = unreadable_page()
        layout, runs, budget = read_page(page, FakeReader(r"\frac{v}{\lambda}", r"\frac{u}{\lambda}", r"\frac{w}{\lambda}"))
        self.assertEqual(sum(1 for r in runs if r.image and not r.math), 1)
        self.assertEqual((budget.read, budget.unsure), (0, 1))
        self.assertEqual((layout.native_formula_pictures, layout.native_formulas_read), (1, 0))

    def test_digits_must_agree_with_tesseract(self):
        _doc, page = unreadable_page()  # the picture at (80, 42, 120, 72) pt: words half inside it are its text
        words = [Word(340, 180, 495, 295, 90.0, "v=7,2", 0, 40.0)]  # px at 300 dpi
        _layout, _runs, budget = read_page(page, FakeReader("v=4,5"), words)  # one-line: its digits are checked
        self.assertEqual((budget.read, budget.unsure), (0, 1))

    def test_budget_is_shared_and_counted(self):
        _doc, page = unreadable_page()
        spent = nativeread.FormulaBudget(0, ["vie"])
        layout, _runs, budget = read_page(page, FakeReader(r"f=\frac{v}{\lambda}"), budget=spent)
        self.assertEqual((budget.over, budget.read), (1, 0))
        self.assertEqual(layout.native_formula_pictures, 1)

    def test_model_missing_keeps_pictures(self):
        _doc, page = unreadable_page()
        layout, _runs, budget = read_page(page, None)
        self.assertTrue(budget.failed)
        self.assertEqual(layout.native_formula_pictures, 1)

    def test_model_failing_turns_it_off(self):
        class Broken:
            def read(self, images):
                raise RuntimeError("past_k")

        _doc, page = unreadable_page()
        with mock.patch.object(mathocr, "broken") as broken:
            layout, _runs, budget = read_page(page, Broken())
        broken.assert_called_once()
        self.assertTrue(budget.failed)
        self.assertEqual(layout.native_formula_pictures, 1)

    def test_cancel_is_checked(self):
        _doc, page = unreadable_page()
        budget = nativeread.FormulaBudget(200, ["vie"], check=mock.Mock(side_effect=RuntimeError("cancelled")))
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            read_page(page, FakeReader("x_{A}"), budget=budget)


def picture_run(text="Ox", size=12.0):
    """A formula picture run as native makes it (RGBA PNG, alpha = ink)."""
    img = np.zeros((60, 160, 4), np.uint8)
    cv2.putText(img, text, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0, 255), 3)
    return Run(text="￼", size=size, image=cv2.imencode(".png", img)[1].tobytes(), image_w=38.4, image_h=14.4)


class HolderTest(unittest.TestCase):
    def read(self, runs, owner, reader, words=()):
        f = {"run": runs[1], "box": pymupdf.Rect(100, 100, 138.4, 114.4), "size": 12.0, "pieces": None}
        h = nativeread.Holder(owner, runs)
        owner.runs = runs
        budget = nativeread.FormulaBudget(200, ["vie"])
        with mock.patch.object(nativeread, "_reader", return_value=reader), \
                mock.patch.object(nativeread, "_words", return_value=list(words)):
            nativeread.read_pictures(pymupdf.open().new_page(), [f], [h], budget)
        return f, budget

    def test_mark_the_line_already_has_is_not_doubled(self):
        runs = [Run("là ", font="Times"), picture_run(), Run(".", font="Times")]
        p = Paragraph(text="", x0=0, y0=0, x1=200, y1=14, font_pt=12, runs=runs)
        # Tesseract's last word ran past the picture ("km." for the picture of "3 km"): the reading took its "."
        words = [Word(420, 420, 450, 470, 90.0, "3", 0, 40.0), Word(470, 420, 575, 470, 90.0, "km.", 0, 40.0)]
        f, _budget = self.read(runs, p, FakeReader(r"3\text{ km}."), words)
        self.assertEqual(p.text, "là 3 km.")
        self.assertEqual(f["read"], "text")

    def test_cell_text_follows(self):
        runs = [Run("v = ", font="Arial"), picture_run(), Run(" m/s", font="Arial")]
        c = Cell()
        _f, _budget = self.read(runs, c, FakeReader("4,5"))
        self.assertEqual(c.text, "v = 4,5 m/s")
        self.assertEqual([r.font for r in c.runs], ["Arial"] * 3)

    def test_plain_letters_agreeing_with_tesseract(self):
        # judge() turns "Ox" down as letters only (OCR reads letters); Tesseract read the same: its text is used
        runs = [Run("trục ", font="Times"), picture_run(), Run(".", font="Times")]
        p = Paragraph(text="", x0=0, y0=0, x1=200, y1=14, font_pt=12, runs=runs)
        words = [Word(415, 415, 580, 480, 90.0, "Ox.", 0, 40.0)]
        _f, budget = self.read(runs, p, FakeReader("O x"), words)
        self.assertEqual(p.text, "trục Ox.")
        self.assertEqual(budget.read_text, 1)

    def test_vietnamese_word_from_tesseract(self):
        runs = [Run("sau ", font="Times"), picture_run("20 phut"), Run(" thì", font="Times")]
        p = Paragraph(text="", x0=0, y0=0, x1=200, y1=14, font_pt=12, runs=runs)
        words = [Word(415, 415, 500, 480, 90.0, "20", 0, 40.0), Word(505, 415, 580, 480, 90.0, "phút", 0, 40.0)]
        _f, _budget = self.read(runs, p, FakeReader(r"2 0 \mathrm{~ p h u t}"), words)
        self.assertEqual(p.text, "sau 20 phút thì")

    def test_tesseract_misreading_a_letter_keeps_the_picture(self):
        # MathType's italic A: Tesseract "4", the model "A" – no agreement, no guess
        runs = [Run("điểm ", font="Times"), picture_run("A"), Run(".", font="Times")]
        p = Paragraph(text="", x0=0, y0=0, x1=200, y1=14, font_pt=12, runs=runs)
        words = [Word(415, 415, 460, 480, 90.0, "4", 0, 40.0)]
        _f, budget = self.read(runs, p, FakeReader("A"), words)
        self.assertEqual((budget.read, budget.unsure), (0, 1))
        self.assertTrue(runs[1].image)


class PlainReadingsTest(unittest.TestCase):
    def test_kinds(self):
        crop = np.full((40, 80), 255, np.uint8)
        crop[10:30, 10:70] = 0
        self.assertEqual(mathocr.plain_readings(FakeReader(r"\mathrm{k m / h}"), crop), ["km/h", "km/h"])
        self.assertEqual(mathocr.plain_readings(FakeReader("x_{A}"), crop), [None, None])  # a script: judge()'s job
        self.assertEqual(mathocr.plain_readings(FakeReader(r"\frac{1}{2}"), crop), [None, None])
        self.assertEqual(mathocr.plain_readings(FakeReader("A", conf=0.5), crop), [None, None])


if __name__ == "__main__":
    unittest.main()
