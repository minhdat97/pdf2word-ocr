"""Tests for app.mathocr on recorded outputs of the formula model (no model needed).

Run: python -m unittest discover -s tests
The readings below are what pix2text-mfr-1.5 wrote for formula regions of two physics
exams (option regions with units and full stops, and tight formula boxes), with the
Tesseract text of the same regions.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import mathocr as M


def as_text(d):
    """A decision as one string: ^{..} / _{..} around scripts; eq as latex + tail."""
    return d.as_latex() if d is not None else None


class TidyTest(unittest.TestCase):
    CASES = (
        ("6 , 4 2 . 1 0 ^ { - 2 1 } ~ \\mathrm { J } .", "6,42.10^{-21}\\ J."),
        ("\\mathrm { 7 , 6 6 . 1 0 ^ { \\mu ^ { 2 2 } } \\, J . }", "7,66.10^{\\mu^{22}}\\ J."),
        ("v _ { { } _ { c q p } } = \\sqrt { \\frac { 3 p } { \\rho } } \\ .", "v_{cqp}=\\sqrt{\\frac{3p}{\\rho}}\\ ."),
        ("x _ { _ A }", "x_{A}"),
        ("\\Delta x = \\left| x _ { \\phantom { } _ { A } } \\cdot x _ { B } \\right|", "\\Delta x=|x_{A}\\cdot x_{B}|"),
        ("\\Delta x = \\left| x _ { \\scriptscriptstyle A } \\cdot x _ { B } \\right|", "\\Delta x=|x_{A}\\cdot x_{B}|"),
        ("8 , 3 \\; \\mathrm { ^ \\circ C } .", "8,3\\ °C."),
        ("2 7 \\, \\, \\mathrm { ^ { o } C . }", "27\\ °C."),
        ("4 0 0 { \\sqrt 2 } ~ \\mathrm { m / s . }", "400\\sqrt{2}\\ m/s."),
        ("V = \\cfrac { 4 } { 3 } \\pi R ^ { 3 }", "V=\\frac{4}{3}\\pi R^{3}"),
        ("V = { \\frac { 4 } { 3 } } \\pi R ^ { 3 }", "V=\\frac{4}{3}\\pi R^{3}"),
        ("\\overline { { L } } = 4 5 , 2 2 \\mathrm { ~ m m }", "\\overline{L}=45,22\\ mm"),
        ("\\textrm { \\psi } v _ { \\textrm { c q p } } = 1", "\\psi v_{cqp}=1"),
        ("{ 1 0 } ^ { - 2 1 }", "10^{-21}"),
        ("^ { 3 \\mathrm { \\ k m } }", "^{3\\ km}"),
        # the root's degree and "\\left ." with the model's blanks between tokens
        ("\\sqrt [ 3 ] { a + b }", "\\sqrt[3]{a+b}"),
        ("x = \\left . \\frac { 1 } { 2 } \\right | _ { 0 } ^ { 1 }", "x=\\frac{1}{2}|_{0}^{1}"),
    )

    def test_cases(self):
        for raw, want in self.CASES:
            with self.subTest(raw=raw):
                self.assertEqual(M.tidy(raw), want)

    def test_idempotent(self):
        for raw, _ in self.CASES:
            with self.subTest(raw=raw):
                once = M.tidy(raw)
                self.assertEqual(M.tidy(once), once)


class DecideTest(unittest.TestCase):
    def test_numbers_with_exponent(self):
        d = M.decide("6 , 4 2 . 1 0 ^ { - 2 1 } ~ \\mathrm { J } .", 0.975, "642.107'J.")
        self.assertEqual(d.kind, "text")
        self.assertEqual(d.parts, [("6,42.10", None), ("-21", "super"), (" J.", None)])

    def test_unit_spacing_and_full_stop_from_ocr(self):
        d = M.decide("6 \\mathrm { ~ k P a }", 0.931, "6 kPa.")
        self.assertEqual(d.parts, [("6 kPa.", None)])
        d = M.decide("\\mathrm { 1 , } 9 2 . 1 0 ^ { 4 } \\mathrm { m / s . }", 0.967, "1,92.106 m/s.")
        self.assertEqual(d.parts, [("1,92.10", None), ("4", "super"), (" m/s.", None)])

    def test_decimal_comma_and_degree(self):
        self.assertEqual(as_text(M.decide("8 , 3 \\; \\mathrm { ^ \\circ C } .", 0.948, "83°C.")), "8,3 °C.")
        self.assertEqual(as_text(M.decide("1 , 2 2 .", 0.91, "122.")), "1,22.")

    def test_subscripts(self):
        d = M.decide("7 2 \\mathrm { v _ { O } } = 6 7 \\mathrm { v _ { H } } .", 0.983, "72vo=67vn.")
        self.assertEqual(d.parts, [("72v", None), ("O", "sub"), ("=67v", None), ("H", "sub"), (".", None)])
        # Tesseract read the subscript A as ",": no comma is added
        self.assertEqual(M.decide("x _ { _ A }", 0.94, "x,").parts, [("x", None), ("A", "sub")])
        self.assertEqual(as_text(M.decide("t _ { 2 } = 0 , 4 0 5 \\mathrm { ~ s }", 0.958, "1, =0,405")), "t_{2}=0,405 s")

    def test_equations(self):
        d = M.decide("v _ { { } _ { c q p } } = \\sqrt { \\frac { 3 p } { \\rho } } \\ .", 0.97, "Vw")
        self.assertEqual((d.kind, d.latex, d.tail), ("eq", "v_{cqp}=\\sqrt{\\frac{3p}{\\rho}}", "."))
        d = M.decide("4 0 0 { \\sqrt 2 } ~ \\mathrm { m / s . }", 0.975, "400/2 m/s.")
        self.assertEqual((d.kind, d.latex, d.tail), ("eq", "400\\sqrt{2}", " m/s."))
        d = M.decide("V = { \\frac { 4 } { 3 } } \\pi R ^ { 3 }", 0.97, "Z7 = 1 ZÑ`)")
        self.assertEqual((d.latex, d.tail), ("V=\\frac{4}{3}\\pi R^{3}", ""))
        # the sentence's closing bracket stays text after the equation; brackets of the formula stay in it
        d = M.decide("V = \\frac { 4 } { 3 } \\pi R ^ { 3 } )", 0.97, "V = 15R) 4")
        self.assertEqual((d.latex, d.tail), ("V=\\frac{4}{3}\\pi R^{3}", ")"))
        d = M.decide("f ( \\frac { 1 } { 2 } ) .", 0.97, "")
        self.assertEqual((d.latex, d.tail), ("f(\\frac{1}{2})", "."))
        d = M.decide("\\overline { { L } } = 4 5 , 2 2 \\mathrm { ~ m m }", 0.993, "J =45,22 mm,")
        self.assertEqual((d.kind, d.latex), ("eq", "\\overline{L}=45,22\\ mm"))

    def test_rejected(self):
        cases = [
            ("6 , 4 2 . 1 0 ^ { - 2 1 } ~ \\mathrm { J } .", 0.80, "642.107'J.", "confidence"),
            ("\\mathrm { 7 , 6 6 . 1 0 ^ { \\mu ^ { 2 2 } } \\, J . }", 0.942, "7,66.1022J.", "symbol \\mu"),
            ("\\mathrm { \\Gamma , 2 3 . 1 0 ^ { - 2 1 } ~ J . }", 0.872, "423.10”J.", "symbol \\Gamma"),
            ("\\ldots ~ 1 ~ \\mathrm { k m / s . }", 0.943, "1 km⁄z.", "symbol \\ldots"),
            ("\\textrm { \\psi } v _ { c q p } = \\sqrt { \\frac { 3 k T } { m _ { 0 } } }", 0.949, "v.= Ea", "symbol \\psi"),
            ("\\mathcal { V } _ { \\omicron }", 0.92, "", "command \\mathcal"),
            ("x _ { \\A }", 0.88, "x,", "symbol \\A"),
            # the whole crop read as an exponent
            ("^ { 3 \\mathrm { \\ k m } }", 0.877, "3 km,", "script without base"),
            # "4v_O = 9v_H" printed with small-capital subscripts
            ("4 _ { \\mathrm { V O } } = 9 _ { \\mathrm { V H } } .", 1.0, "4vo = 9vn.", "subscript after a digit"),
            # Vietnamese words: "20 phút"
            ("2 0 \\mathrm { ~ p h u t }", 0.992, "20 phút", "word phut"),
            ("\\mathrm { t a n g } ~ ~ 2 \\sqrt { 3 } ~ ~ \\mathrm { l \\dot { a } n } .", 0.908, "tăng 2v3 lần.", "command \\dot"),
            ("\\mathrm { t a n g } ~ 2 \\sqrt { 3 } ~ \\mathrm { l a n } .", 0.95, "tăng 2v3 lần.", "word tang"),
            # case: Tesseract saw "v", the model "V"
            ("\\mathrm { V _ { H } }", 0.992, "vụ", "letter case"),
            ("o _ { x }", 0.982, "Ox.", "letter case"),
            # digits that Tesseract read otherwise (one of them is wrong)
            ("\\mathrm { 1 , } 9 2 . 1 0 ^ { 5 } ~ \\mathrm { m / s . }", 0.981, "102.10 m/s.", "digits differ from OCR"),
            ("\\omega", 0.863, "0", "digits only in OCR"),
            # plain letters: the OCR text is as good
            ("O x", 0.869, "Ox.", "letters only"),
            ("x = \\frac { 1 } { 2", 0.95, "", "unbalanced"),
        ]
        for raw, conf, ocr, why in cases:
            with self.subTest(raw=raw):
                d, reason = M.judge(raw, conf, ocr)
                self.assertIsNone(d)
                self.assertEqual(reason, why)

    def test_digits_against_ocr(self):
        # a digit the model left out or added, or read as a letter, is caught
        for raw, ocr in (
            ("2 0 0 \\mathrm { ~ m / s . }", "2000 m/s."),
            ("2 0 0 0 0 \\mathrm { ~ m / s . }", "2000 m/s."),
            ("1 3 2 ~ ^ { \\circ } \\mathrm { C } .", "1323 °C."),
            ("5 0 0 ~ 0 0 ~ \\mathrm { N } .", "500 000 N."),
            ("1 1 1 1 1 ~ \\mathrm { J } .", "1111 J."),
            ("+ 0 , 0 0 + s .", "40,004 s."),
            ("v = 2 \\omega ~ m / s", "v=20 m/s"),
            ("7 \\sqrt { 3 }", "2/3"),  # digits outside a root or fraction are checked too
        ):
            with self.subTest(raw=raw):
                self.assertEqual(M.judge(raw, 0.95, ocr), (None, "digits differ from OCR"))
        # Tesseract's exponents, degree signs and letters read as digits, and its reading of a fraction, are not
        for raw, ocr, want in (
            ("6 , 4 2 . 1 0 ^ { - 2 1 } ~ \\mathrm { J } .", "642.107'J.", "6,42.10^{-21} J."),
            ("7 , 4 3 . 1 0 ^ { - 2 1 } ~ \\mathrm { J } .", "743.10”7.", "7,43.10^{-21} J."),
            ("1 , 1 . 1 0 ^ { 3 } ~ \\mathrm { m / s } .", "1,1.106 m/s.", "1,1.10^{3} m/s."),
            ("t _ { 2 } = 0 , 4 0 5", "1, =0,405", "t_{2}=0,405"),
            ("L = 4 5 , 2 2 \\pm 0 , 0 3 ~ ( \\mathrm { m m } )", "1=45,22+0,03 (mm).", "L=45,22±0,03 (mm)."),
            ("6 ~ \\mathrm { m / s }", "6 ó6ms.", "6 m/s."),
            ("2 7 ~ ^ { \\circ } \\mathrm { C } .", "270C.", "27 °C."),
            ("2 \\frac { 1 } { 3 }", "21 3", "2\\frac{1}{3}"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(as_text(M.decide(raw, 0.95, ocr)), want)

    def test_capital_delta_with_a_letter(self):
        self.assertEqual(as_text(M.decide("\\Delta x", 0.99, "Ax")), "Δx")
        self.assertEqual(as_text(M.decide("\\Delta A .", 0.99, "Aa.")), "ΔA.")

    def test_label_left_in_ocr_text(self):
        # "A. " in front of the OCR text does not count as digits of the reading
        d = M.decide("1 , 9 . 1 0 ^ { 3 } ~ \\mathrm { m / s . }", 0.98, "D. 1,9.106 m/s.")
        self.assertEqual(as_text(d), "1,9.10^{3} m/s.")


class VoteTest(unittest.TestCase):
    def test_spacing_differences_agree(self):
        readings = [("2 7 \\, \\, \\mathrm { ^ { o } C . }", 0.859), ("2 7 \\mathrm { ^ \\circ C . }", 0.96)]
        self.assertEqual(as_text(M.vote(readings, "27°C.")), "27 °C.")

    def test_two_of_three(self):
        readings = [
            ("t _ { 5 } = 0 , 4 0 0 \\ \\mathrm { s }", 0.895),
            ("t _ { s } = 0 , 4 0 0 \\ \\mathrm { s }", 0.879),
            ("t _ { 5 } = 0 , 4 0 0 \\ \\mathrm { s }", 0.888),
        ]
        self.assertEqual(as_text(M.vote(readings, "f, 0,400 s.")), "t_{5}=0,400 s.")
        self.assertIsNone(M.vote(readings[:2], "f, 0,400 s."))

    def test_one_reading_is_not_enough(self):
        self.assertIsNone(M.vote([("6 , 4 2 . 1 0 ^ { - 2 1 }", 0.99)], ""))
        self.assertIsNone(M.vote([("6 , 4 2 . 1 0 ^ { - 2 1 }", 0.99), ("6 , 4 2 . 1 0 ^ { - 2 1 }", 0.7)], ""))

    def test_junk_exponent_counts_for_a_minus(self):
        junk = [
            ("\\mathrm { 7 , 6 6 . 1 0 ^ { \\mu ^ { 2 2 } } \\, J . }", 0.942),
            ("7 , 6 6 . 1 0 ^ { ' 2 2 } ~ J .", 0.93),
        ]
        self.assertEqual(M.repaired(junk[0][0]), "7,66.10^{-22}\\ J.")
        self.assertIsNone(M.vote(junk, "7,66.1022J."))  # nobody read the "-"
        d = M.vote(junk + [("7 , 6 6 . 1 0 ^ { - 2 2 } ~ \\mathrm { J } .", 0.9)], "7,66.1022J.")
        self.assertEqual(as_text(d), "7,66.10^{-22} J.")
        # a repair never makes other digits
        d = M.vote([junk[0], ("7 , 6 6 . 1 0 ^ { - 2 1 } ~ \\mathrm { J } .", 0.9)], "7,66.1022J.")
        self.assertIsNone(d)


class CoefficientTest(unittest.TestCase):
    def test_letter_read_into_the_subscript_of_a_digit(self):
        self.assertEqual(M.coefficients("4 _ { \\mathrm { V O } } = 9 _ { \\mathrm { V H } } ."), "4v_{O}=9v_{H}.")
        self.assertEqual(M.coefficients("8 _ { \\mathrm { V _ { O } } } = 3 \\mathrm { v _ { H } } ."), "8v_{O}=3v_{H}.")
        self.assertIsNone(M.coefficients("1 0 _ { 2 }"))  # a digit: no letter to take out
        self.assertIsNone(M.coefficients("4 _ { A B }"))  # its case is not told by its height
        self.assertIsNone(M.coefficients("7 2 v _ { O }"))

    def test_ink_of_letters_with_subscripts(self):
        gray = np.full((60, 200), 255, np.uint8)
        for x in (10, 100):
            gray[10:42, x : x + 18] = 0  # a digit
            gray[21:42, x + 22 : x + 40] = 0  # "v" at x-height on the baseline
            gray[26:46, x + 43 : x + 60] = 0  # its subscript, lowered
        self.assertEqual(M._letter_subscripts(gray), 2)
        gray[42:46, :] = 255  # nothing lowered
        self.assertEqual(M._letter_subscripts(gray), 0)


class FakeReader:
    """Stands in for FormulaReader: answers by crop width (one script per region) and counts passes."""

    def __init__(self, by_width):
        self.by_width = by_width
        self.passes = []

    def read(self, images):
        self.passes.append(len(images))
        out = []
        for im in images:
            width = next(w for w in self.by_width if abs(im.shape[1] - w) <= 16)
            answers = self.by_width[width]
            out.append(answers[min(len(self.passes) - 1, len(answers) - 1)])
        return out


class ReadRegionsTest(unittest.TestCase):
    def test_votes_and_third_pass(self):
        gray = np.full((200, 700), 255, np.uint8)
        gray[50:80, 20:120] = 0
        gray[50:80, 200:400] = 0
        gray[50:80, 450:690] = 0
        reader = FakeReader(
            {
                100: [("1 0 ^ { 5 }", 0.99)],  # agree at once
                200: [("1 , 2 2", 0.95), ("1 2 2", 0.95), ("1 , 2 2", 0.9)],  # third pass decides
                240: [("\\theta", 0.95)],  # never usable
            }
        )
        boxes = [(10, 40, 130, 90), (190, 40, 410, 90), (440, 40, 695, 90)]
        ds = M.read_regions(gray, boxes, ["105", "122", ""], reader=reader)
        self.assertEqual([as_text(d) for d in ds], ["10^{5}", "1,22", None])
        self.assertEqual(reader.passes, [3, 3, 1])

    def test_minus_read_as_a_symbol_confirmed_without_the_unit(self):
        gray = np.full((100, 300), 255, np.uint8)
        for x0, x1 in ((20, 60), (64, 100), (104, 160), (172, 190)):  # a number, a blank gap, the unit
            gray[30:60, x0:x1] = 0
        crop = M.cut(gray, (10, 20, 200, 70))
        self.assertEqual(M._without_unit(crop).shape, (30, 140))
        self.assertIsNone(M._without_unit(crop[:, :140]))  # no unit after the number
        junk = [("\\mathrm { 7 , 6 6 . 1 0 ^ { \\mu ^ { 2 2 } } \\, J . }", 0.94),
                ("\\mathrm { 7 , 6 6 . 1 0 ^ { \\theta ^ { 2 2 } } \\, J . }", 0.97)]
        for head, want in (("7 , 6 6 . 1 0 ^ { - 2 2 }", "7,66.10^{-22} J."), ("7 , 6 6 . 1 0 ^ { 2 2 }", None)):
            reader = FakeReader({176: junk, 146: [(head, 0.97)]})
            ds = M.read_regions(gray, [(10, 20, 200, 70)], ["7,66.1022J."], reader=reader)
            self.assertEqual([as_text(d) for d in ds], [want])
            self.assertEqual(reader.passes, [1, 1, 1, 1])

    def test_reading_without_the_final_mark_confirms(self):
        gray = np.full((100, 200), 255, np.uint8)
        gray[30:62, 20:50] = 0  # T
        gray[50:70, 52:58] = 0  # a thin subscript 1
        gray[40:66, 70:76] = 0  # ";"
        read = {62: [("T _ { 1 } ;", 0.9), ("T _ { \\i } ;", 0.95)], 44: [("T _ { 1 }", 0.99)]}
        reader = FakeReader(read)
        self.assertEqual([as_text(d) for d in M.read_regions(gray, [(10, 20, 90, 80)], ["T,;"], reader=reader)], ["T_{1};"])
        self.assertEqual(reader.passes, [1, 1, 1, 1])
        # the ";" only the model saw: not confirmed
        self.assertEqual(M.read_regions(gray, [(10, 20, 90, 80)], ["T,"], reader=FakeReader(read)), [None])

    def test_coefficient_and_case_from_the_ink(self):
        gray = np.full((100, 300), 255, np.uint8)
        for x in (20, 110):
            gray[30:62, x : x + 18] = 0
            gray[41:62, x + 22 : x + 40] = 0
            gray[46:66, x + 43 : x + 60] = 0
        coef = ("4 _ { \\mathrm { V O } } = 9 _ { \\mathrm { V H } } .", 1.0)
        reader = FakeReader({158: [coef]})
        ds = M.read_regions(gray, [(10, 20, 200, 80)], ["4vo = 9vH."], reader=reader)
        self.assertEqual([as_text(d) for d in ds], ["4v_{O}=9v_{H}."])
        gray[62:66, :] = 255  # the ink shows no subscripts: the reading stays refused
        ds = M.read_regions(gray, [(10, 20, 200, 80)], ["4vo = 9vH."], reader=FakeReader({158: [coef]}))
        self.assertEqual(ds, [None])
        # "V_{H}" for v_H: the glyph's case when it was measured, else refused
        small = np.full((100, 100), 255, np.uint8)
        small[41:62, 20:38] = 0
        small[46:66, 41:58] = 0
        read = {44: [("\\mathrm { V _ { H } }", 0.99)]}
        ds = M.read_regions(small, [(10, 20, 70, 80)], ["vh"], reader=FakeReader(read), case_from_ink=[True])
        self.assertEqual([as_text(d) for d in ds], ["v_{H}"])
        self.assertEqual(M.read_regions(small, [(10, 20, 70, 80)], ["vh"], reader=FakeReader(read)), [None])

    def test_reading_equal_to_ocr_read_once(self):
        gray = np.full((100, 400), 255, np.uint8)
        gray[30:60, 20:120] = 0
        gray[30:60, 200:380] = 0
        boxes = [(10, 20, 130, 70), (190, 20, 390, 70)]
        read = {100: [("6 \\mathrm { ~ k P a } .", 0.97)], 180: [("1 0 ^ { 5 }", 0.99)]}
        for same_stop, passes in ((False, [2, 2]), (True, [2, 1])):
            reader = FakeReader(read)
            with mock.patch.object(M, "SAME_STOP", same_stop):
                ds = M.read_regions(gray, boxes, ["6 kPa.", "105"], reader=reader)
            self.assertEqual([as_text(d) for d in ds], ["6 kPa.", "10^{5}"])
            self.assertEqual(reader.passes, passes)  # "10^{5}" has a script: Tesseract cannot confirm it
        # a plain reading Tesseract does not agree with is read again
        reader = FakeReader({100: [("8 \\mathrm { ~ k P a } .", 0.97)], 180: [("1 0 ^ { 5 }", 0.99)]})
        with mock.patch.object(M, "SAME_STOP", True):
            self.assertEqual(M.read_regions(gray, boxes[:1], ["6 kPa."], reader=reader), [None])
        self.assertEqual(reader.passes, [1, 1])

    def test_model_off(self):
        with mock.patch.object(M, "get_reader", return_value=None):
            self.assertEqual(M.read_regions(np.zeros((10, 10), np.uint8), [(0, 0, 5, 5)], [""]), [None])


class CutTest(unittest.TestCase):
    def test_label_dot_removed_and_trimmed(self):
        gray = np.full((100, 300), 255, np.uint8)
        gray[60:66, 10:16] = 0  # the "." of a label, 6 px
        gray[30:70, 40:250] = 0  # the formula
        crop = M.cut(gray, (0, 0, 300, 100), em=50)
        self.assertEqual(crop.shape, (40, 210))
        self.assertEqual(int(crop.min()), 0)

    def test_minus_sign_kept(self):
        gray = np.full((100, 300), 255, np.uint8)
        gray[48:52, 10:30] = 0  # a minus sign: wide
        gray[30:70, 40:250] = 0
        self.assertEqual(M.cut(gray, (0, 0, 300, 100), em=50).shape[1], 240)

    def test_empty(self):
        self.assertIsNone(M.cut(np.full((50, 50), 255, np.uint8), (0, 0, 50, 50)))


class ModelPartsTest(unittest.TestCase):
    def test_vocab_decode(self):
        vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "\\": 5, "frac": 6, "Ġ": 7, "{": 8, "1": 9, "}": 10, "Ã©": 11}
        data = {"model": {"type": "BPE", "vocab": vocab},
                "added_tokens": [{"id": i, "content": c, "special": True} for c, i in vocab.items() if i < 3]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            v = M.Vocab(path)
        self.assertEqual(v.decode([1, 5, 6, 7, 8, 9, 10, 2, 0]), "\\frac {1}")
        self.assertEqual(v.decode([11]), "é")

    def test_resize_bicubic(self):
        flat = np.full((17, 900), 200, np.uint8)
        self.assertTrue((M.resize_bicubic(flat, 384, 384) == 200).all())
        img = (np.arange(384 * 384) % 251).astype(np.uint8).reshape(384, 384)
        self.assertTrue((M.resize_bicubic(img, 384, 384) == img).all())

    def test_resize_filters_stay_small(self):
        # cached for the process's life: per output pixel, not a dense [384, width] matrix (13.8 MB at 4500 px)
        starts, taps = M._weights(4500, 384)
        self.assertLess(starts.nbytes + taps.nbytes, 200_000)
        img = (np.arange(40 * 4500) % 253).astype(np.uint8).reshape(40, 4500)
        dense = np.zeros((384, 4500))
        for row, (x, t) in enumerate(zip(starts, taps)):
            n = min(len(t), 4500 - x)
            dense[row, x : x + n] = t[:n]
        want = np.clip(np.floor((img.astype(np.float64) @ dense.T + (1 << 21)) / (1 << 22)), 0, 255).astype(np.uint8)
        self.assertTrue((M._resample(img, (starts, taps), 1) == want).all())

    @unittest.skipUnless(__import__("importlib").util.find_spec("PIL"), "Pillow not installed")
    def test_resize_matches_pillow(self):
        from PIL import Image

        rng = np.random.RandomState(0)
        for h, w in [(5, 3), (60, 700), (400, 1200), (384, 384)]:
            img = (rng.rand(h, w) * 255).astype(np.uint8)
            want = np.asarray(Image.fromarray(img).resize((384, 384), Image.BICUBIC))
            self.assertTrue((M.resize_bicubic(img, 384, 384) == want).all())

    def test_split_decoder_reads_as_the_whole_one(self):
        """cross_kv + decoder_step (scripts/split_mfr_decoder.py) decode as decoder_model does, rows that end
        dropping out of the batch."""
        rng = np.random.RandomState(0)
        proj = rng.rand(4, 3).astype(np.float32)
        emb = rng.rand(30, 3).astype(np.float32)

        class Run:
            def __init__(self, fn, outputs=()):
                self.fn, self.calls = fn, []
                self.get_outputs = lambda: [SimpleNamespace(name=n) for n in outputs]

            def run(self, _names, feed):
                self.calls.append({k: v.shape for k, v in feed.items()})
                return self.fn(feed)

        def logits(ids, kv):  # next token from the last token and the crop's keys; EOS (2) after 3-6 steps
            n = ids.shape[1]
            score = (kv.sum(axis=(1, 2))[:, None] * emb[ids[:, -1]].sum(1, keepdims=True)) % 1
            out = np.zeros((len(ids), n, 30), np.float32)
            nxt = np.where(n >= 3 + (score[:, 0] * 4).astype(int), 2, 3 + (score[:, 0] * 20).astype(int))
            out[np.arange(len(ids)), -1, nxt] = 10.0
            return [out]

        encoder = Run(lambda f: [f["pixel_values"][:, :1, :5, :4].reshape(len(f["pixel_values"]), 5, 4)])
        whole = Run(lambda f: logits(f["input_ids"], f["encoder_hidden_states"] @ proj))
        cross = Run(lambda f: [f["encoder_hidden_states"] @ proj], ["kv0"])
        step = Run(lambda f: logits(f["input_ids"], f["kv0"]), ["logits"])
        vocab = mock.Mock(decode=lambda ids: " ".join(str(int(i)) for i in ids))
        crops = [(rng.rand(20 + 7 * k, 60) * 255).astype(np.uint8) for k in range(5)]
        plain = M.FormulaReader.__new__(M.FormulaReader)
        plain.encoder, plain.decoder, plain.cross, plain.vocab = encoder, whole, None, vocab
        split = M.FormulaReader.__new__(M.FormulaReader)
        split.encoder, split.decoder, split.cross, split.step, split.vocab = encoder, None, cross, step, vocab
        with mock.patch.object(M, "ENCODE_ONE", False):
            want = plain.read(crops)
        self.assertEqual(len(encoder.calls), 1)
        with mock.patch.object(M, "ENCODE_ONE", True):  # one crop per encoder run: the same readings
            self.assertEqual(split.read(crops), want)
        self.assertEqual(len(encoder.calls), 1 + len(crops))
        self.assertEqual(len({t for t, _ in want}), 5)  # rows of several lengths
        self.assertEqual(len(cross.calls), 1)
        self.assertEqual(len(step.calls), len(whole.calls))
        self.assertEqual([c["kv0"][0] for c in step.calls], [c["input_ids"][0] for c in step.calls])

    def test_split_files_count_as_the_model(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M, "MFR_DIR", Path(tmp)):
            for name in ("encoder_model.onnx", "tokenizer.json") + M.SPLIT_FILES:
                (Path(tmp) / name).write_bytes(b"")
            self.assertTrue(M.available())
            (Path(tmp) / M.SPLIT_FILES[0]).unlink()
            self.assertFalse(M.available())

    def test_no_model_files(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(M, "MFR_DIR", Path(tmp)):
            self.assertFalse(M.available())


if __name__ == "__main__":
    unittest.main()
