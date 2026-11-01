"""PDFs printed through Adobe's PDF Library from Word (a file of four exam versions, a key and a landscape matrix):
glyph boxes several em tall, a math font mapping each glyph to its letter twice, lines cut at every script, each
version numbering its own pages, landscape pages after portrait ones. Run: python -m unittest discover -s tests"""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lxml import etree  # noqa: E402

from app import native  # noqa: E402
from app.convert import split_versions  # noqa: E402
from app.docx_builder import build_docx  # noqa: E402
from app.model import TAB, PageLayout, Paragraph, Run  # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def span(text, x0, x1, base, size=12.0, font="TimesNewRomanPSMT"):
    """A rawdict span: one char box per character across x0..x1, an em around the baseline."""
    step = (x1 - x0) / max(len(text), 1)
    chars = [{"c": c, "origin": (x0 + k * step, base), "bbox": (x0 + k * step, base - 0.8 * size, x0 + (k + 1) * step,
                                                                base + 0.2 * size)} for k, c in enumerate(text)]
    return {"text": text, "font": font, "size": size, "flags": 0, "origin": (x0, base), "ascender": 0.8,
            "descender": -0.2, "bbox": (x0, base - 0.8 * size, x1, base + 0.2 * size), "chars": chars}


def line(*spans):
    x0 = min(s["bbox"][0] for s in spans)
    return {"dir": (1.0, 0.0), "wmode": 0, "spans": list(spans),
            "bbox": (x0, min(s["bbox"][1] for s in spans), max(s["bbox"][2] for s in spans),
                     max(s["bbox"][3] for s in spans))}


class TightBoxesTest(unittest.TestCase):
    def test_tall_boxes_cut_to_the_em(self):
        s = span("E", 85, 93, 193.2)
        s["chars"][0]["bbox"] = (85, 155.8, 93, 222.8)  # 67 pt on a 12 pt line (Cambria Math by Adobe)
        s["bbox"] = (85, 155.8, 93, 222.8)
        ln = line(s)
        ln["bbox"] = s["bbox"]
        block = {"type": 0, "bbox": s["bbox"], "lines": [ln]}
        native._tight_boxes([block])
        self.assertEqual([round(v, 1) for v in s["chars"][0]["bbox"]], [85, 183.6, 93, 195.6])
        self.assertEqual(round(block["bbox"][3] - block["bbox"][1], 1), 12.0)

    def test_ordinary_boxes_untouched(self):
        s = span("Câu 1", 56, 90, 300)
        before = s["chars"][0]["bbox"]
        native._tight_boxes([{"type": 0, "bbox": s["bbox"], "lines": [line(s)]}])
        self.assertEqual(s["chars"][0]["bbox"], before)


class BrokenMapTest(unittest.TestCase):
    def chars(self, *items):
        return [{"c": c, "bbox": (x0, 0, x1, 10)} for c, x0, x1 in items]

    def test_letter_mapped_twice(self):
        # one glyph of E written "𝐸𝐸": the second without width
        self.assertTrue(native._broken_map(self.chars(("𝐸", 85.4, 92.9), ("𝐸", 92.9, 92.9))))
        self.assertTrue(native._broken_map(self.chars(("�", 0, 5))))

    def test_doubled_letters_drawn_twice_are_fine(self):
        self.assertFalse(native._broken_map(self.chars(("𝑥", 0, 5), ("𝑥", 5, 10))))


class SplitLinesTest(unittest.TestCase):
    def test_script_and_meeting_pieces_join_their_line(self):
        # "Câu 5: … có " | "³" | "2 dm" " hỗn hợp …" (Adobe cuts the line at the superscript)
        first = line(span("Câu 5: Trong xilanh có ", 56.6, 303.5, 657.2))
        sup = line(span("3", 333.3, 337.2, 651.4, size=7.6))
        rest = line(span("2 dm", 305.6, 333.0, 657.1), span(" hỗn hợp khí", 339.5, 483.5, 657.2))
        out = native._join_interleaved([first, sup, rest])
        self.assertEqual(len(out), 1)
        self.assertEqual("".join(s["text"] for s in out[0]["spans"]), "Câu 5: Trong xilanh có 2 dm3 hỗn hợp khí")

    def test_subscript_before_its_line(self):
        # p₁'s "1" set in a line of its own before the "p"
        sub = line(span("1", 93.0, 96.4, 289.6, size=7.1))
        main = line(span("p", 87.3, 93.3, 286.1, size=12.3))
        out = native._join_interleaved([sub, main])
        self.assertEqual([("".join(s["text"] for s in ln["spans"])) for ln in out], ["p1"])

    def test_gap_between_pieces_is_a_space(self):
        out = native._join_interleaved([line(span("p1", 87.3, 96.4, 286.1)), line(span("<", 99.7, 106.2, 286.1))])
        self.assertEqual("".join(s["text"] for s in out[0]["spans"]), "p1 <")

    def test_formula_glyphs_stay_apart(self):
        # MathType glyphs without text ("!" at 7.9 pt) are grouped by _find_formulas, not joined to the text
        text = line(span("Kết quả là ", 42.5, 483.7, 499.7, size=13.0))
        glyph = line(span("!", 489.7, 493.6, 503.1, size=7.9))
        self.assertEqual(len(native._join_interleaved([text, glyph])), 2)

    def test_two_columns_stay_apart(self):
        left = line(span("SỞ GIÁO DỤC", 56, 200, 100))
        right = line(span("ĐỀ KIỂM TRA", 330, 520, 100))
        self.assertEqual(len(native._join_interleaved([left, right])), 2)


def footer(code, n, total):
    return Paragraph(f"Mã đề {code}\tTrang {n}/{total}", 56, 806, 566, 818, 12, zone="bottom",
                     runs=[Run(f"Mã đề {code}"), Run(TAB), Run(f"Trang {n}/{total}")], tab_stops=[(480.0, "left")])


def body(text, y=60):
    return Paragraph(text, 56, y, 566, y + 14, 12, runs=[Run(text)])


class VersionsTest(unittest.TestCase):
    def pages(self):
        pages = []
        for code in (121, 122):
            for n in (1, 2):
                pages.append(PageLayout(595, 842, elements=[body(f"Câu {code}-{n}"), footer(code, n, 2)]))
        pages.append(PageLayout(595, 842, elements=[body("ĐÁP ÁN")]))
        pages.append(PageLayout(792, 612, elements=[body("MA TRẬN")]))
        return pages

    def test_each_version_numbers_its_pages(self):
        pages = self.pages()
        versions = split_versions(pages)
        self.assertEqual([(v.start, v.end) for v in versions], [(0, 1), (2, 3)])
        self.assertEqual([(r.text, r.field) for r in versions[1].line.runs if r.field or "Mã" in r.text],
                         [("Mã đề 122", None), ("1", "PAGE"), ("2", "SECTIONPAGES")])
        self.assertTrue(all(len(p.elements) == 1 for p in pages))  # the footer lines left their pages

    def test_one_numbering_is_not_versions(self):
        pages = [PageLayout(595, 842, elements=[body("x"), footer(121, n, 3)]) for n in (1, 2, 3)]
        self.assertEqual(split_versions(pages), [])

    def test_sections(self):
        pages = self.pages()
        versions = split_versions(pages)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out, versions=versions)
            z = zipfile.ZipFile(out)
            root = etree.fromstring(z.read("word/document.xml"))
            sects = list(root.iter(W + "sectPr"))
            footers = {n: z.read(n).decode() for n in z.namelist() if n.startswith("word/footer")}
        # 2 versions, the key, the landscape matrix
        self.assertEqual(len(sects), 4)
        sizes = [(s.find(W + "pgSz").get(W + "w"), s.find(W + "pgSz").get(W + "orient")) for s in sects]
        self.assertEqual(sizes[-1], ("15840", "landscape"))
        self.assertEqual([s.find(W + "pgNumType") is not None for s in sects[:2]], [True, True])
        texts = sorted(f for f in footers.values() if "SECTIONPAGES" in f)
        self.assertEqual(len(texts), 2)
        self.assertTrue(any("Mã đề 121" in f for f in texts) and any("Mã đề 122" in f for f in texts))
        # no paragraph of its own for a section break after text: it sits in the section's last paragraph
        self.assertEqual(sum(1 for p in root.iter(W + "p") if p.find(f"{W}pPr/{W}sectPr") is not None
                             and not "".join(p.itertext()).strip()), 0)

    def test_one_section_stays_as_before(self):
        pages = [PageLayout(595, 842, elements=[body("a")]), PageLayout(595, 842, elements=[body("b")])]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out)
            root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
        sects = list(root.iter(W + "sectPr"))
        self.assertEqual(len(sects), 1)
        self.assertIsNone(sects[0].find(W + "pgSz").get(W + "orient"))


if __name__ == "__main__":
    unittest.main()
