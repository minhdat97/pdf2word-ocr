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
from app.convert import _move_repeated, split_versions  # noqa: E402
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

    def test_a_glyph_mapped_to_several_signs_is_fine(self):
        # "…" written "...", ″ written ′′: not letters, the font's map is not broken
        self.assertFalse(native._broken_map(self.chars(("′", 0, 5), ("′", 5, 5))))

    def test_ligature_ff_is_fine(self):
        # a math font (STIX) writing its ﬀ ligature as "ff", the second without width
        self.assertFalse(native._broken_map(self.chars(("f", 0, 5), ("f", 5, 5))))

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

    def test_script_inside_a_span_goes_in_its_place(self):
        # "₂" drawn before its line, the line one span running across it ("khí H trong khí O dư"), with room for it
        sub = line(span("2", 102.5, 106.5, 103, size=7))
        s = span("khí H trong khí O dư", 60, 230, 100)  # "H" ends at 102.5
        for c in s["chars"][5:]:
            c["bbox"] = (c["bbox"][0] + 4.5, c["bbox"][1], c["bbox"][2] + 4.5, c["bbox"][3])
            c["origin"] = (c["origin"][0] + 4.5, c["origin"][1])
        s["bbox"] = (60, s["bbox"][1], 234.5, s["bbox"][3])
        main = line(s)
        out = native._join_interleaved([sub, main])
        text = "".join(s["text"] for s in out[0]["spans"])
        self.assertEqual(text.index("2"), text.index("H") + 1)

    def test_short_line_beside_a_heading_joins_the_line_it_runs_into(self):
        # Word's export: "… 27 C" then "⁰ )" starting inside it; a 16 pt heading in the block makes "⁰ )" look small
        head = line(span("ĐỀ THI THỬ", 200, 300, 60, size=16))
        main = line(span("Nhiệt lượng (t – 27 C", 56, 200, 100))
        sup = line(span("0", 190, 194, 95, size=7), span(" )", 194, 205, 100))
        out = native._join_interleaved([head, main, sup])
        self.assertEqual(len(out), 2)
        self.assertEqual("".join(s["text"] for s in out[1]["spans"]).replace(" ", ""), "Nhiệtlượng(t–27C0)")

    def test_no_space_where_a_script_goes(self):
        # Adobe's order: the right piece, then the script, then the left piece; nothing between "27", "⁰" and "C"
        right = line(span("C, áp suất", 119.6, 170, 100))
        sup = line(span("0", 115.3, 119.1, 95.4, size=7.1))
        left = line(span("nhiệt độ 27", 56, 115, 100))
        out = native._join_interleaved([right, sup, left])
        self.assertEqual(len(out), 1)
        self.assertEqual("".join(s["text"] for s in out[0]["spans"]), "nhiệt độ 270C, áp suất")

    def test_stacked_scripts_lower_first(self):
        sup = line(span("2", 70, 74, 95, size=7))
        sub = line(span("1", 70, 74, 103, size=7))
        host = line(span("trị", 56, 70, 100))
        out = native._join_interleaved([sup, sub, host])
        self.assertEqual("".join(s["text"] for s in out[0]["spans"]), "trị12")

    def test_wide_script_at_the_end(self):
        # "Fe" "³⁺": the script runs more than half an em past its line's end
        host = line(span("Fe", 56, 70, 100))
        sup = line(span("3+", 70.3, 78.5, 95, size=7))
        out = native._join_interleaved([host, sup])
        self.assertEqual(len(out), 1)
        self.assertEqual("".join(s["text"] for s in out[0]["spans"]), "Fe3+")

    def test_option_label_keeps_its_tab(self):
        # "… D." set 4 pt after the option before it: two options, not one run of text
        a = line(span("A. p1 < p2.", 70, 124, 286))
        b = line(span("B. p1 > p2.", 128, 182, 286))
        self.assertEqual(len(native._join_interleaved([a, b])), 2)

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

    def test_no_header_part_when_versions_use_footers(self):
        # an empty unlinked header (one 12.6 pt paragraph 36 pt from the edge) would push the body of every page
        pages = self.pages()
        versions = split_versions(pages)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out, versions=versions)
            z = zipfile.ZipFile(out)
            names = z.namelist()
            empties = [z.read(n).decode() for n in names if n.startswith("word/footer") and "SECTIONPAGES" not in z.read(n).decode()]
            root = etree.fromstring(z.read("word/document.xml"))
        self.assertFalse(any(n.startswith("word/header") for n in names))
        self.assertTrue(empties and all('w:lineRule="exact"' in f for f in empties))  # the key's and matrix's
        restarts = [s.find(W + "pgNumType") is not None for s in root.iter(W + "sectPr")]
        self.assertEqual(restarts, [True, True, False, False])  # not copied into the key and the matrix

    def build(self, pages, versions):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out, versions=versions)
            z = zipfile.ZipFile(out)
            parts = {n: etree.fromstring(z.read(n)) for n in z.namelist() if n.endswith(".xml") and n.startswith("word/")}
        return parts

    def test_empty_story_in_schema_order(self):
        # w:pStyle first in w:pPr (Word may refuse or drop the rest otherwise)
        pages = self.pages()
        parts = self.build(pages, split_versions(pages))
        for name, root in parts.items():
            for ppr in root.iter(W + "pPr"):
                tags = [c.tag for c in ppr]
                if W + "pStyle" in tags:
                    self.assertEqual(tags.index(W + "pStyle"), 0, name)

    def test_versions_in_two_zones(self):
        # 121 numbered in the footer, 122 in the header (a part from another source): each in its own story
        pages = [PageLayout(595, 842, elements=[body(f"Câu {n}"), footer(121, n, 2)]) for n in (1, 2)]
        for n in (1, 2):
            t = f"Mã đề 122 - Trang {n}/2"
            pages.append(PageLayout(595, 842, elements=[Paragraph(t, 56, 20, 566, 32, 12, zone="top", runs=[Run(t)]),
                                                        body(f"Câu {n}")]))
        parts = self.build(pages, split_versions(pages))
        doc = parts["word/document.xml"]
        sects = list(doc.iter(W + "sectPr"))
        self.assertEqual(len(sects), 2)
        text = {n: "".join(r.itertext()) for n, r in parts.items() if "/header" in n or "/footer" in n}
        self.assertTrue(any("122" in t for n, t in text.items() if "/header" in n))
        self.assertFalse(any("122" in t for n, t in text.items() if "/footer" in n))
        self.assertTrue(all(int(s.find(W + "pgMar").get(W + "footer")) < 1000 for s in sects))  # not 805 pt

    def test_scan_sizes_are_one_paper(self):
        pages = [PageLayout(w, h, elements=[body("x")]) for w, h in ((595, 842), (589, 835), (601, 849))]
        parts = self.build(pages, [])
        self.assertEqual(len(list(parts["word/document.xml"].iter(W + "sectPr"))), 1)

    def test_footer_distance_from_the_lines_page(self):
        # the fullest line is on the version's portrait page 2; its first page is landscape
        pages = []
        for code in (101, 102):
            pages.append(PageLayout(842, 595, elements=[body("Bảng"), Paragraph("Trang 1/2", 400, 559, 440, 571, 12,
                                                                               zone="bottom", runs=[Run("Trang 1/2")])]))
            t = f"Mã đề {code} - Trang 2/2"
            pages.append(PageLayout(595, 842, elements=[body("Câu 1"), Paragraph(t, 56, 806, 566, 818, 12, zone="bottom",
                                                                                runs=[Run(t)])]))
        versions = split_versions(pages)
        self.assertEqual([v.page for v in versions], [1, 3])
        parts = self.build(pages, versions)
        self.assertTrue(all(s.find(W + "pgMar").get(W + "footer") == "480"
                            for s in parts["word/document.xml"].iter(W + "sectPr")))  # 842 - 818 = 24 pt

    def test_empty_page_opening_a_section_is_kept(self):
        pages = self.pages()[:4]
        pages.insert(2, PageLayout(595, 842))  # blank page between the versions
        pages += [PageLayout(595, 842), PageLayout(595, 842, elements=[body("ĐÁP ÁN")])]  # blank page, then the key
        doc = self.build(pages, split_versions(pages))["word/document.xml"]
        key = next(p for p in doc.iter(W + "p") if "ĐÁP ÁN" in "".join(p.itertext()))
        before = key.getprevious()
        self.assertEqual("".join(before.itertext()), "")
        self.assertIsNone(before.find(f"{W}pPr/{W}sectPr"))  # the blank page's own paragraph, not a section's end

    def test_repeated_lines_go_with_versions(self):
        # a school's name on every page goes; a title on each version's first page (pages 1, 3, 5) stays
        pages = []
        for code in (121, 122, 123):
            for n in (1, 2):
                top = [Paragraph("Sở GD&ĐT Hà Nội", 56, 20, 200, 32, 12, zone="top", runs=[Run("Sở GD&ĐT Hà Nội")])]
                if n == 1:
                    top.append(Paragraph("ĐỀ THI THỬ", 250, 36, 350, 48, 12, zone="top", runs=[Run("ĐỀ THI THỬ")]))
                pages.append(PageLayout(595, 842, elements=top + [body(f"Câu {n}", 120), footer(code, n, 2)]))
        self.assertEqual(len(split_versions(pages)), 3)
        _move_repeated(pages, {}, two_sided=False)
        texts = [[e.text for e in p.elements] for p in pages]
        self.assertFalse(any("Sở GD&ĐT Hà Nội" in t for t in texts))
        self.assertEqual(sum("ĐỀ THI THỬ" in t for t in texts), 3)

    def test_label_inside_a_piece(self):
        for text in ("Mã đề {c} - Trang {n}/2", "Trang {n}/2 - Mã đề thi {c}", "Page {n} of 2 – {c}"):
            pages = []
            for code in (121, 122):
                for n in (1, 2):
                    t = text.format(c=code, n=n)
                    pages.append(PageLayout(595, 842, elements=[
                        body(f"Câu {n}"), Paragraph(t, 56, 806, 566, 818, 12, zone="bottom", runs=[Run(t)])]))
            with self.subTest(text):
                versions = split_versions(pages)
                self.assertEqual([(v.start, v.end) for v in versions], [(0, 1), (2, 3)])
                fields = [(r.text, r.field) for r in versions[0].line.runs if r.field]
                self.assertEqual(fields, [("1", "PAGE"), ("2", "SECTIONPAGES")])  # never the code's 121
                self.assertIn("121", "".join(r.text for r in versions[0].line.runs if not r.field))

    def test_version_over_a_landscape_page(self):
        # A4, A4 landscape, A4 in one version: three sections with its line, numbered on, the PDF's total
        pages = []
        for code in (121, 122):
            for n, (w, h) in enumerate(((595, 842), (842, 595), (595, 842)), 1):
                f = footer(code, n, 3)
                pages.append(PageLayout(w, h, elements=[body(f"Câu {n}"), f]))
        versions = split_versions(pages)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out, versions=versions)
            z = zipfile.ZipFile(out)
            root = etree.fromstring(z.read("word/document.xml"))
            footers = [z.read(n).decode() for n in z.namelist() if n.startswith("word/footer")]
        sects = list(root.iter(W + "sectPr"))
        self.assertEqual(len(sects), 6)
        self.assertEqual([s.find(W + "pgNumType") is not None for s in sects], [True, False, False] * 2)
        self.assertTrue(all("Mã đề" in f for f in footers))
        self.assertFalse(any("SECTIONPAGES" in f for f in footers))

    def test_footer_distance_on_another_paper(self):
        # one numbering, an A3 page in an A4 file: the footer stays as far from the edge as on page 1
        pages = [PageLayout(595, 842, elements=[body("a")]), PageLayout(842, 1191, elements=[body("b")])]
        line = Paragraph("Trang 1/2", 280, 806, 320, 818, 12, zone="bottom", runs=[Run("Trang 1/2")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out, footer=[line])
            root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
        distances = [int(s.find(W + "pgMar").get(W + "footer")) for s in root.iter(W + "sectPr")]
        self.assertEqual(distances, [480, 480])  # 842 - 818 = 24 pt

    def test_near_sizes_are_one_paper(self):
        pages = [PageLayout(595.28, 841.89, elements=[body("a")]), PageLayout(595.5, 842.5, elements=[body("b")])]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx(pages, out)
            root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
        self.assertEqual(len(list(root.iter(W + "sectPr"))), 1)

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
