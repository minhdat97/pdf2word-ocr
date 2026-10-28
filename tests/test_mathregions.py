"""The "rules" formula detector (app/mathregions.py) on synthetic pages: scripts in running text, 2-D formulas
(fraction bar, bar over a letter, radical), answer labels Tesseract garbled, and how app/ocr.py writes such a
label. Glyphs are filled rectangles; 1em = 30 px, x-height 14 px, baseline at y=100.
Run: python -m unittest discover -s tests"""

import sys
import unittest
from itertools import pairwise
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import mathocr, mathregions, ocr  # noqa: E402
from app.mathocr import MathDecision  # noqa: E402
from app.ocr import Line, Word  # noqa: E402

EM, XH, BASE = 30, 14, 100
BLUE = (0x33, 0x66, 0xFF)


class Canvas:
    def __init__(self, w=1600, h=400):
        self.rgb = np.full((h, w, 3), 255, np.uint8)

    def block(self, x0, y0, x1, y1, colour=(0, 0, 0)):
        self.rgb[y0:y1, x0:x1] = colour

    def letters(self, x0, n, base=BASE, high=XH, step=14):
        """n letters of x-height (or `high`) standing on the baseline; returns the right edge."""
        for k in range(n):
            self.block(x0 + k * step, base - high, x0 + k * step + 10, base)
        return x0 + (n - 1) * step + 10

    def arrays(self):
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        return gray, (gray < 128).astype(np.uint8), self.rgb


def word(x0, y0, x1, y1, text, conf=95.0, **kw) -> Word:
    return Word(x0, y0, x1, y1, conf, text, 0, EM, **kw)


def detect(canvas, lines):
    gray, ink, rgb = canvas.arrays()
    return mathregions.find(lines, gray, ink, rgb, 300, method="rules")


def prose_line(canvas, words_at, base=BASE):
    """Prose words ("khí", "là", ...) of x-height letters at the given x; returns their Words."""
    out = []
    for x, text in words_at:
        right = canvas.letters(x, len(text), base)
        out.append(word(x, base - XH, right, base, text))
    return out


class ScriptsTest(unittest.TestCase):
    def page(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (700, "và"), (800, "của")])
        # "10^5": two digits at cap height, then a small 5 raised over the caps
        c.block(200, 79, 210, 100)
        c.block(214, 79, 224, 100)
        c.block(226, 74, 234, 86)
        words.append(word(200, 74, 234, 100, "105", conf=40))
        # "n_1": a letter, then a small 1 lowered under the baseline
        c.block(300, 86, 312, 100)
        c.block(314, 96, 319, 108)
        words.append(word(300, 86, 319, 108, "n,", conf=30))
        # "x,": a comma is lower than a subscript
        c.block(400, 86, 412, 100)
        c.block(414, 95, 418, 103)
        words.append(word(400, 86, 418, 103, "x,", conf=90))
        # "°C": a degree sign stops at the caps' height
        c.block(500, 79, 508, 88)
        c.block(511, 79, 524, 100)
        words.append(word(500, 79, 524, 100, "°C", conf=93))
        # "ế"-like: an accent just over a letter is no superscript
        c.block(600, 86, 612, 100)
        c.block(603, 75, 609, 82)
        words.append(word(600, 75, 612, 100, "ế", conf=60))
        return c, [Line(1, 1, BASE, EM, words)]

    def test_superscript_and_subscript_words(self):
        c, lines = self.page()
        regions = detect(c, lines)
        self.assertEqual([(r.kind, r.ocr_text) for r in regions], [("inline", "105"), ("inline", "n,")])
        self.assertEqual(regions[0].box, (200, 74, 234, 100))  # tight on the ink with its exponent
        self.assertEqual(regions[1].box, (300, 86, 319, 108))
        self.assertIs(regions[0].line, lines[0])
        self.assertEqual(regions[0].words, [lines[0].words[4]])

    def test_exponent_split_off_joins_its_number(self):
        c, lines = self.page()
        number = next(w for w in lines[0].words if w.text == "105")
        lines[0].words.remove(number)
        lines[0].words += [word(200, 79, 224, 100, "10", conf=93), word(226, 74, 234, 86, "5", conf=50)]
        regions = detect(c, lines)
        self.assertEqual(regions[0].box, (200, 74, 234, 100))
        self.assertEqual(regions[0].ocr_text, "10 5")

    def test_low_confidence_number_with_a_mark(self):
        c, lines = self.page()
        c.block(900, 79, 910, 100)
        c.block(912, 97, 915, 103)
        c.block(918, 79, 928, 100)
        lines[0].words.append(word(900, 79, 928, 103, "8,31", conf=42))
        lines[0].words.append(word(1000, 79, 1010, 100, "7", conf=42))  # no ink: nothing to read
        c.block(1100, 79, 1110, 100)
        lines[0].words.append(word(1100, 79, 1110, 100, "8", conf=40))  # a plain integer stays OCR's
        texts = [r.ocr_text for r in detect(c, lines)]
        self.assertIn("8,31", texts)
        self.assertNotIn("8", texts)

    def test_delta_read_as_a_letter(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (700, "và")])
        cv2.polylines(c.rgb, [np.array([[900, 98], [910, 78], [920, 98]], np.int32)], True, (0, 0, 0), 2)  # Δ
        c.block(924, 86, 936, 100)
        words.append(word(900, 78, 936, 100, "Ax", conf=91))
        cv2.polylines(c.rgb, [np.array([[1000, 99], [1010, 78], [1020, 99]], np.int32)], False, (0, 0, 0), 2)
        cv2.line(c.rgb, (1005, 91), (1015, 91), (0, 0, 0), 2)  # a capital A: its counter ends half way
        c.block(1024, 86, 1036, 100)
        words.append(word(1000, 78, 1036, 100, "Ax", conf=91))
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.ocr_text, r.box[0]) for r in regions], [("Ax", 899)])

    def test_case_of_the_first_letter_from_its_height(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (700, "và")])
        c.block(300, 86, 312, 100)  # "v" at x-height, read "V"
        c.block(314, 90, 326, 104)  # a small capital H sitting low
        words.append(word(300, 86, 326, 104, "Vh", conf=80))
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([r.ocr_text for r in regions], ["vh"])
        self.assertEqual([r.case_from_ink for r in regions], [True])


class TwoDimensionalTest(unittest.TestCase):
    def test_fraction_in_running_text_stops_at_prose(self):
        c = Canvas()
        words = prose_line(c, [(20, "tăng"), (100, "là"), (400, "lần")])
        c.block(250, 66, 270, 84)  # numerator
        c.block(240, 89, 290, 91)  # bar
        c.block(250, 96, 270, 116)  # denominator
        c.block(215, 86, 228, 100)  # "x" before it, joined
        words += [word(215, 86, 228, 100, "x", conf=50), word(240, 66, 290, 116, "E", conf=20)]
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.kind, r.box) for r in regions], [("tall", (215, 66, 290, 116))])
        self.assertEqual([w.text for w in regions[0].words], ["x", "E"])

    def test_bar_over_a_letter(self):
        c = Canvas()
        words = prose_line(c, [(20, "giá"), (100, "trị"), (400, "và")])
        c.block(250, 72, 266, 75)  # the bar
        c.block(250, 79, 266, 100)  # "A"
        words.append(word(250, 79, 266, 100, "A", conf=89))
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.kind, r.box, r.ocr_text) for r in regions], [("tall", (250, 72, 266, 100), "A")])

    def test_equals_and_minus_signs_are_no_bars(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (400, "và")])
        c.block(250, 88, 270, 90)
        c.block(250, 94, 270, 96)  # "="
        c.block(300, 92, 320, 94)  # "-"
        words += [word(250, 88, 270, 96, "="), word(300, 92, 320, 94, "-")]
        self.assertEqual(detect(c, [Line(1, 1, BASE, EM, words)]), [])

    def test_radical(self):
        c = Canvas()
        words = prose_line(c, [(20, "tăng"), (400, "lần")])
        c.block(200, 79, 210, 100)  # "2"
        x0, top, bottom = 214, 64, 104
        pts = [(x0, 86), (x0 + 6, bottom), (x0 + 16, top), (x0 + 60, top)]
        for a, b in pairwise(pts):
            cv2.line(c.rgb, a, b, (0, 0, 0), 2)
        c.block(240, 79, 252, 100)  # "3" under it
        words.append(word(200, 64, 276, 105, "2/3", conf=30))
        _, ink, rgb = c.arrays()
        page = mathregions._Page([Line(1, 1, BASE, EM, words)], ink, rgb)
        self.assertIn("root", [k for k, _, _ in mathregions._seeds(page)])
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.kind, r.ocr_text) for r in regions], [("tall", "2/3")])
        self.assertEqual(regions[0].box[0], 200)  # the "2" before the sign joins

    def test_fraction_goes_into_its_own_column_line(self):
        # two columns: "... tỉ số A/Q" ends the left line, the right column starts 2.2em after the fraction; the
        # right line's baseline may be the nearer one to the fraction's
        for right_base in (198, 200, 201, 202):
            with self.subTest(right_base=right_base):
                c = Canvas(1400, 400)
                left = prose_line(c, [(100 + 80 * i, t) for i, t in enumerate(["hiệu", "suất", "tỉ", "số"])], base=200)
                c.block(400, 187, 460, 190)  # bar
                c.block(418, 155, 442, 182)  # numerator
                c.block(418, 195, 442, 222)  # denominator
                right = prose_line(c, [(526 + 80 * i, t) for i, t in enumerate(["nhiệt", "lượng", "tỏa"])], base=right_base)
                lines = [Line(1, 1, 200, EM, left), Line(2, 2, right_base, EM, right)]
                regions = detect(c, lines)
                self.assertEqual([(r.kind, r.box) for r in regions], [("tall", (400, 155, 460, 222))])
                self.assertIs(regions[0].line, lines[0])

    def labelled_fraction(self, label_text):
        """ "C. 3/2" after prose: a capital letter and its dot, blank paper, then a fraction; the label read as given."""
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (600, "và")])
        c.block(200, 79, 216, 100)  # "C"
        c.block(219, 96, 223, 100)  # "."
        c.block(250, 66, 270, 84)  # numerator
        c.block(240, 89, 290, 91)  # bar
        c.block(250, 96, 270, 116)  # denominator
        if label_text == "c.x=—":  # glued to the fraction
            words.append(word(200, 66, 290, 116, label_text, conf=40))
        else:
            words += [word(200, 79, 223, 100, label_text, conf=40), word(240, 66, 290, 116, "E", conf=20)]
        return c, [Line(1, 1, BASE, EM, words)]

    def test_garbled_label_stops_a_fraction_and_is_kept(self):
        for text in ("c.x=—", "C"):
            with self.subTest(label=text):
                c, lines = self.labelled_fraction(text)
                regions = detect(c, lines)
                self.assertEqual([(r.kind, r.box) for r in regions], [("tall", (240, 66, 290, 116))])
                self.assertEqual((regions[0].label.text, regions[0].label.box), ("C.", (200, 79, 223, 100)))
                self.assertNotIn("c.", regions[0].ocr_text)

    def test_label_read_into_an_equation_is_written_as_text(self):
        c, lines = self.labelled_fraction("p")  # not taken for a label: the fraction grows over it
        self.assertEqual([r.box for r in detect(c, lines)], [(200, 66, 290, 116)])
        gray, ink, rgb = c.arrays()

        class Reader:
            def read(self, images):
                return [("C . \\frac { 3 } { 2 }", 0.99) for _ in images]

        with mock.patch.object(mathocr, "available", return_value=True), ocr.formulas_mode("model"), \
                mock.patch.object(mathocr, "get_reader", return_value=Reader()), \
                mock.patch.object(mathregions, "MATH_DETECT", "rules"):
            ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        row = [w for w in sorted(lines[0].words, key=lambda w: w.x0) if 150 < w.x0 < 300]
        self.assertEqual([(w.text, w.math is None) for w in row][0], ("C.", True))
        self.assertEqual(row[1].math.latex, "\\frac{3}{2}")
        self.assertNotIn("p", [w.text for w in lines[0].words])

    def test_split_label(self):
        d = mathocr.MathDecision("eq", latex="C.\\ W=\\frac{m}{2v}", tail=" J.", conf=0.9)
        label, rest = mathocr.split_label(d)
        self.assertEqual((label, rest.latex, rest.tail), ("C.", "W=\\frac{m}{2v}", " J."))
        self.assertIsNone(mathocr.split_label(mathocr.MathDecision("eq", latex="\\frac{1}{2}")))
        self.assertIsNone(mathocr.split_label(mathocr.MathDecision("text", parts=[("A. 3", None)])))


class GarbledLabelsTest(unittest.TestCase):
    def page(self):
        """Two option rows at columns 20, 400, 800, 1200: the first read well, the second with "C." glued to
        its option ("c.9") and "D." read as "p.9"."""
        c = Canvas()
        lines = []
        for row, base in enumerate((100, 200)):
            words = []
            for k, x in enumerate((20, 400, 800, 1200)):
                c.block(x, base - 21, x + 16, base, BLUE)  # letter
                c.block(x + 19, base - 4, x + 23, base, BLUE)  # dot
                c.block(x + 40, base - 21, x + 50, base)  # "9"
                c.letters(x + 60, 2, base)  # "km"
                letter = "ABCD"[k]
                if row == 1 and k >= 2:
                    words.append(word(x, base - 21, x + 50, base, ("c" if k == 2 else "p") + ".9", conf=40))
                else:
                    words += [word(x, base - 21, x + 23, base, letter + ".", color="3366FF"),
                              word(x + 40, base - 21, x + 50, base, "9", conf=60)]
                words.append(word(x + 60, base - XH, x + 84, base, "km.", conf=60))
            lines.append(Line(1, row + 1, base, EM, words))
        return c, lines

    def test_label_found_on_the_glyphs(self):
        c, lines = self.page()
        regions = [r for r in detect(c, lines) if r.line is lines[1]]
        labelled = [r for r in regions if r.label is not None]
        self.assertEqual([r.label.text for r in labelled], ["C.", "D."])
        self.assertEqual(labelled[0].label.box, (800, 179, 823, 200))
        self.assertEqual(labelled[0].label.color, "3366FF")
        self.assertEqual(labelled[0].box, (840, 179, 884, 200))  # the option, not the label
        self.assertEqual(labelled[0].ocr_text, "9 km.")  # the label's text left out

    def test_label_in_black_found_by_its_weight(self):
        c, lines = self.page()
        c.rgb[(c.rgb == BLUE).all(axis=2)] = 0
        for ln in lines:
            for w in ln.words:
                w.color = None
        thin = c.rgb.copy()
        thin[180:200, 1203:1213] = 255  # "D" of row 2 drawn with thin strokes: not a label
        c.rgb = thin
        regions = [r for r in detect(c, lines) if r.line is lines[1] and r.label is not None]
        self.assertEqual([r.label.text for r in regions], ["C."])

    def test_numbers_at_the_label_columns_are_no_labels(self):
        c, lines = self.page()
        c.rgb[(c.rgb == BLUE).all(axis=2)] = 0  # black labels: found by their weight
        for ln in lines:
            for w in ln.words:
                w.color = None
        base, words = 300, []
        for x, n in ((20, "2"), (400, "3")):  # "2. 2,5 km | 3. 4,0 km": a numbered answer line
            c.block(x, base - 21, x + 16, base)  # the digit, as high and bold as a label's letter
            c.block(x + 19, base - 4, x + 23, base)  # its dot
            c.block(x + 40, base - 21, x + 50, base)
            c.letters(x + 60, 2, base)
            words += [word(x, base - 21, x + 50, base, n + ".2,5" if x == 20 else n + ".", conf=60),
                      word(x + 60, base - XH, x + 84, base, "km", conf=60)]
            if x == 400:
                words.insert(-1, word(x + 40, base - 21, x + 50, base, "4,0", conf=60))
        lines.append(Line(1, 3, base, EM, words))
        self.assertEqual([r for r in detect(c, lines) if r.line is lines[2] and r.label is not None], [])

    def test_a_lone_glyph_takes_no_letter_from_its_column(self):
        style = mathregions._Style(21.0, 0.5, None, [(20.0, {"A": 2}), (400.0, {"B": 2})])
        self.assertEqual(mathregions._letters_of([(20.0, None)], style, EM), [None])
        self.assertEqual(mathregions._letters_of([(20.0, None), (400.0, None)], style, EM), ["A", "B"])

    def test_label_is_written_when_read(self):
        c, lines = self.page()
        gray, ink, rgb = c.arrays()

        class Reader:
            def read(self, images):
                return [("9 \\mathrm { k m } .", 0.99) for _ in images]

        with mock.patch.object(mathocr, "available", return_value=True), ocr.formulas_mode("model"), \
                mock.patch.object(mathocr, "get_reader", return_value=Reader()), \
                mock.patch.object(mathregions, "MATH_DETECT", "rules"):
            ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        row = sorted(lines[1].words, key=lambda w: w.x0)
        texts = [w.text for w in row]
        self.assertEqual(texts.count("C."), 1)
        self.assertNotIn("c.9", texts)
        label = next(w for w in row if w.text == "C.")
        self.assertEqual((label.x0, label.color), (800, "3366FF"))
        nxt = row[row.index(label) + 1]
        self.assertIsInstance(nxt.math, MathDecision)

    def test_label_is_written_when_not_read(self):
        # the default way reads no option without a superscript or subscript, and a reading may be refused: the
        # garbled label is written all the same, the option keeps its OCR words ("c.9" keeps its "9")
        for how in ("scripts", "model"):
            c, lines = self.page()
            gray, ink, rgb = c.arrays()

            class Refused:
                def read(self, images):
                    return [("", 0.0) for _ in images]

            with mock.patch.object(mathocr, "available", return_value=True), ocr.formulas_mode(how), \
                    mock.patch.object(mathocr, "get_reader", return_value=Refused()), \
                    mock.patch.object(mathregions, "MATH_DETECT", "rules"):
                ocr.read_formulas(lines, [], gray, ink, rgb, 300)
            row = sorted(lines[1].words, key=lambda w: w.x0)
            self.assertEqual([w.text for w in row], ["A.", "9", "km.", "B.", "9", "km.", "C.", "9", "km.", "D.", "9", "km."], how)
            label = next(w for w in row if w.text == "C.")
            self.assertEqual((label.x0, label.color), (800, "3366FF"))
            nine = row[row.index(label) + 1]
            self.assertEqual((nine.x0, nine.x1, nine.math), (824, 850, None))

    def test_label_left_alone_when_the_word_over_it_does_not_split(self):
        c, lines = self.page()
        gray, ink, rgb = c.arrays()
        glued = next(w for w in lines[1].words if w.text == "c.9")
        glued.text = "e9"  # no mark after the letter: where the label ends in the text is not known
        with ocr.formulas_mode("scripts"), mock.patch.object(mathregions, "MATH_DETECT", "rules"):
            ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        texts = [w.text for w in sorted(lines[1].words, key=lambda w: w.x0)]
        self.assertIn("e9", texts)
        self.assertNotIn("C.", texts)
        self.assertIn("D.", texts)


if __name__ == "__main__":
    unittest.main()
