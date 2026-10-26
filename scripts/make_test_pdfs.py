"""Generate sample PDFs (born-digital, scanned, mixed) into samples/.

Scanned files are produced by rendering a text PDF to a grayscale bitmap with a
little noise and re-embedding only the image, so they carry no text layer.
"""

from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REGULAR = str(FONT_DIR / "Times New Roman.ttf")
BOLD = str(FONT_DIR / "Times New Roman Bold.ttf")

A4 = pymupdf.paper_rect("a4")
MARGIN = 64

VI_PAGES = [
    {
        "title": "Báo cáo hoạt động quý III năm 2026",
        "paragraphs": [
            "Trong quý III, công ty đã hoàn thành việc triển khai hệ thống quản lý tài liệu "
            "điện tử cho ba chi nhánh tại Hà Nội, Đà Nẵng và Thành phố Hồ Chí Minh. Doanh thu "
            "tăng trưởng ổn định nhờ mở rộng tệp khách hàng doanh nghiệp vừa và nhỏ.",
            "Bảng dưới đây tổng hợp kết quả kinh doanh theo từng khu vực trong kỳ báo cáo.",
        ],
        "table": [
            ["Khu vực", "Doanh thu (triệu đồng)", "Tăng trưởng"],
            ["Miền Bắc", "12.500", "14%"],
            ["Miền Trung", "6.800", "9%"],
            ["Miền Nam", "15.200", "17%"],
        ],
        "after": [
            "Ghi chú: số liệu đã được phòng kế toán đối chiếu và xác nhận vào ngày 30/09/2026.",
        ],
    },
    {
        "title": "Kế hoạch quý IV",
        "paragraphs": [
            "Ưu tiên hàng đầu là nâng cấp tính năng nhận dạng ký tự quang học (OCR) để xử lý "
            "chính xác các văn bản tiếng Việt có dấu, bao gồm hợp đồng, hóa đơn và biên bản họp.",
            "Đồng thời, đội ngũ kỹ thuật sẽ tối ưu thời gian chuyển đổi tài liệu PDF sang Word, "
            "giảm độ trễ trung bình xuống dưới năm giây cho mỗi trang.",
            "Người phụ trách: Phòng Công nghệ thông tin.",
        ],
        "table": None,
        "after": [],
    },
]

EN_PAGES = [
    {
        "title": "Quarterly Report 2026",
        "paragraphs": [
            "This document is a scanned test page used to check how a PDF to Word converter "
            "handles optical character recognition. The page contains only an image, without "
            "any embedded text layer.",
        ],
        "table": [
            ["Region", "Revenue", "Growth"],
            ["North", "$1,200,000", "12%"],
            ["South", "$950,000", "8%"],
        ],
        "after": ["Prepared by the finance team. Contact: finance@example.com"],
    }
]


def _textbox(page, y, text, size, font, fontname, align=pymupdf.TEXT_ALIGN_LEFT):
    rect = pymupdf.Rect(MARGIN, y, A4.width - MARGIN, A4.height - MARGIN)
    unused = page.insert_textbox(
        rect, text, fontsize=size, fontname=fontname, fontfile=font, align=align
    )
    return y + (rect.height - unused) + size * 0.6


def _table(page, y, rows, size=11):
    col_w = (A4.width - 2 * MARGIN) / len(rows[0])
    row_h = size * 2.2
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x0 = MARGIN + c * col_w
            cell_rect = pymupdf.Rect(x0, y + r * row_h, x0 + col_w, y + (r + 1) * row_h)
            page.draw_rect(cell_rect, color=(0, 0, 0), width=1)
            font, name = (BOLD, "tnrb") if r == 0 else (REGULAR, "tnr")
            page.insert_text(
                (x0 + 6, cell_rect.y1 - row_h / 2 + size / 3),
                cell,
                fontsize=size,
                fontname=name,
                fontfile=font,
            )
    return y + len(rows) * row_h + size * 1.2


def draw_page(doc, spec):
    page = doc.new_page(width=A4.width, height=A4.height)
    y = _textbox(page, MARGIN, spec["title"], 20, BOLD, "tnrb", pymupdf.TEXT_ALIGN_CENTER)
    y += 6
    for para in spec["paragraphs"]:
        y = _textbox(page, y, para, 12, REGULAR, "tnr", pymupdf.TEXT_ALIGN_JUSTIFY)
    if spec["table"]:
        y = _table(page, y + 4, spec["table"])
    for para in spec["after"]:
        y = _textbox(page, y, para, 12, REGULAR, "tnr")
    return page


def text_pdf(pages):
    doc = pymupdf.open()
    for spec in pages:
        draw_page(doc, spec)
    return doc


def rasterize(doc, dpi=200, seed=7, skew=0.0, turns=0):
    """Return a new PDF whose pages are noisy grayscale images of doc's pages,
    optionally skewed by `skew` degrees or turned by `turns` quarter turns."""
    import cv2

    rng = np.random.default_rng(seed)
    out = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        img = np.ascontiguousarray(img[:, : pix.width])
        if skew:
            m = cv2.getRotationMatrix2D((pix.width / 2, pix.height / 2), skew, 1)
            img = cv2.warpAffine(img, m, (pix.width, pix.height), borderValue=255)
        img = np.ascontiguousarray(np.rot90(img, turns))
        noisy = np.clip(img.astype(np.int16) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
        h, w = noisy.shape
        image = pymupdf.Pixmap(pymupdf.csGRAY, w, h, noisy.tobytes(), False)
        target = out.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        target.insert_image(target.rect, stream=image.tobytes("jpeg", jpg_quality=80))
    return out


BLUE, RED = (0, 0, 1), (1, 0, 0)


def _put(page, x, y, text, size=12, bold=False, italic=False, color=(0, 0, 0)):
    """Insert one run at baseline y; returns the x where it ends."""
    style = " Bold" if bold else (" Italic" if italic else "")
    fontfile = str(FONT_DIR / f"Times New Roman{style}.ttf")
    name = "tnr" + ("b" if bold else "i" if italic else "")
    page.insert_text((x, y), text, fontsize=size, fontname=name, fontfile=fontfile, color=color)
    return x + pymupdf.Font(fontfile=fontfile).text_length(text, size)


def _put_runs(page, x, y, runs):
    """runs: (text, options) where options may hold bold/italic/color and
    script="super"|"sub" for 8pt text raised or lowered off the baseline."""
    for text, opts in runs:
        opts = dict(opts)
        script = opts.pop("script", None)
        if script:
            x = _put(page, x, y + (-4.5 if script == "super" else 1.5), text, size=8, **opts)
        else:
            x = _put(page, x, y, text, **opts)
    return x


def _figure_png() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 240, 200), True)
    pix.set_rect(pix.irect, (255, 255, 255, 0))  # transparent background
    pix.set_rect(pymupdf.IRect(40, 60, 200, 80), (90, 90, 90, 255))  # lever
    pix.set_rect(pymupdf.IRect(110, 80, 130, 170), (150, 110, 60, 255))  # pivot
    return pix.tobytes("png")


def exam_pdf():
    """An exam-style sheet with the features the converter must keep:
    two-column letterhead, a VNI-encoded title (as left by legacy fonts),
    colours, super/subscripts, answer options on tab stops, a labelled
    figure, and a repeated header/footer with "Trang x/2"."""
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=A4.width, height=A4.height)
        _put(page, 250, 22, "Trường THPT Mẫu", size=11, italic=True)
        # Pale diagonal stamp, as test-sharing sites add.
        stamp = pymupdf.Point(150, 700)
        page.insert_text(stamp, "BẢN DÙNG THỬ", fontsize=40, fontname="tnrb", fontfile=BOLD,
                         color=(0, 0, 0), fill_opacity=0.12, morph=(stamp, pymupdf.Matrix(35)))
        _put(page, 60, 826, "Đề thi thử")
        _put(page, 500, 826, f"Trang {n}/2")
        # What sharing sites add: a translucent name stamp and a footer link.
        page.insert_text((420, 22), "Nguồn: kho đề mẫu", fontsize=10, fontname="tnr",
                         fontfile=REGULAR, color=(0, 0, 0), fill_opacity=0.5)
        end = _put(page, 230, 826, "https://dethi.example.vn")
        page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(229, 815, end + 1, 830),
                          "uri": "https://dethi.example.vn"})
        if n == 1:
            # Letterhead: two blocks side by side, each centred on its own axis.
            for text, y, opts in (("SỞ GD&ĐT HÀ NỘI", 60, {"bold": True}),
                                  ("ÑEÀ KIEÅM TRA SOÁ 1", 78, {"bold": True, "color": RED}),
                                  ("(Đề gồm 02 trang)", 95, {"italic": True})):
                w = pymupdf.Font(fontfile=REGULAR).text_length(text, 12)
                _put(page, 140 - w / 2, y, text, **opts)
            for text, y, opts in (("KỲ THI THỬ TỐT NGHIỆP NĂM 2026", 60, {"bold": True}),
                                  ("Môn: VẬT LÍ", 78, {"bold": True})):
                w = pymupdf.Font(fontfile=BOLD).text_length(text, 12)
                _put(page, 400 - w / 2, y, text, **opts)
            y = 140
            _put_runs(page, 64, y, [("Câu 1: ", {"bold": True, "color": BLUE}),
                                    ("Hằng số Avogadro N", {}), ("A", {"script": "sub"}),
                                    (" = 6,02.10", {}), ("23", {"script": "super"}),
                                    (" mol", {}), ("-1", {"script": "super"}), (". Giá trị đúng là", {})])
            for x, label, value in ((80, "A.", "1"), (200, "B.", "2"), (320, "C.", "3"), (440, "D.", "4")):
                _put_runs(page, x, y + 18, [(label, {"bold": True, "color": BLUE}),
                                            (f" {value}.10", {}), ("23", {"script": "super"})])
            _put_runs(page, 64, y + 44, [("Câu 2: ", {"bold": True, "color": BLUE}),
                                         ("Đòn bẩy trong hình bên cân bằng khi nào?", {})])
            # A pale vector watermark band under the text, crossing the figure
            # (the figure image has a transparent background, so it shows through).
            band = page.new_shape()
            band.draw_quad(pymupdf.Quad((60, 420), (90, 440), (560, 200), (530, 180)))
            band.finish(fill=(0.87, 0.92, 0.97), color=None, fill_opacity=0.5)
            band.commit(overlay=False)
            page.insert_image(pymupdf.Rect(400, 190, 520, 290), stream=_figure_png())
            _put(page, 470, 285, "Hình 1", size=10, italic=True)
            for i, text in enumerate(("Khi mômen lực hai bên bằng nhau.", "Khi hai lực bằng nhau.")):
                _put_runs(page, 80, y + 64 + 18 * i, [("AB"[i] + ".", {"bold": True, "color": BLUE}),
                                                      (" " + text, {})])
        else:
            _put_runs(page, 64, 60, [("Câu 3: ", {"bold": True, "color": BLUE}),
                                     ("Khối lượng riêng của nước là 1000 kg/m", {}),
                                     ("3", {"script": "super"}), (".", {})])
            _put(page, 250, 100, "--- HẾT ---", bold=True)
    return doc


def main():
    OUT.mkdir(exist_ok=True)
    vi = text_pdf(VI_PAGES)
    vi.save(OUT / "vi-text.pdf")
    rasterize(vi).save(OUT / "vi-scan.pdf")
    rasterize(text_pdf(EN_PAGES)).save(OUT / "en-scan.pdf")

    mixed = pymupdf.open()
    mixed.insert_pdf(text_pdf(VI_PAGES[:1]))
    mixed.insert_pdf(rasterize(text_pdf(VI_PAGES[1:])))
    mixed.save(OUT / "mixed.pdf")

    exam = exam_pdf()
    exam.save(OUT / "exam-like.pdf")
    rasterize(exam).save(OUT / "exam-like-scan.pdf")
    rasterize(vi, skew=2.0).save(OUT / "vi-scan-skewed.pdf")
    rasterize(vi, turns=1).save(OUT / "vi-scan-turned.pdf")

    for f in sorted(OUT.glob("*.pdf")):
        print(f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
