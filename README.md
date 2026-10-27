# pdf2word-ocr

Prototype chuyển PDF sang Word có OCR, viết cho đề thi tiếng Việt: upload file → task
`analyse-document` → task `pdf-to-word` (cơ bản hoặc OCR) → poll kết quả → tải `.docx`.

Trang đã có lớp text được **dựng lại từ chính lớp text đó** chứ không nhận dạng lại từ ảnh;
OCR chỉ dùng cho trang scan và ảnh lớn chứa chữ.

## Chạy

```bash
brew install tesseract                       # engine OCR
cd pdf2word-ocr
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000   # mở http://127.0.0.1:8000
```

`tessdata/` đã có model `vie` và `eng` bản tessdata_best (4.1.0), `osd` và `configs`.
Muốn thêm ngôn ngữ, tải `<lang>.traineddata` từ
https://github.com/tesseract-ocr/tessdata_best vào `tessdata/`; giao diện tự hiện thêm.

VietOCR (nên có cho file scan tiếng Việt) đọc lại từng từ Tesseract tìm được, sửa dấu và chữ
Tesseract đọc sai. Mô hình (`vgg_transformer` của VietOCR chuyển sang ONNX, trọng số float16,
76 MB, chạy bằng onnxruntime, không cần PyTorch) không nằm trong git, lấy về `models/vietocr/`:

```bash
.venv/bin/python scripts/fetch_models.py                   # tải theo scripts/models.json
.venv/bin/python scripts/fetch_models.py --from <thư mục>  # hoặc chép từ thư mục có sẵn các file
.venv/bin/python scripts/fetch_models.py --check           # chỉ kiểm tra
.venv/bin/python scripts/fetch_models.py --model mfr       # chỉ một mô hình
```

VietOCR chưa có địa chỉ tải trong `scripts/models.json`: dựng `models/vietocr/` bằng
`scripts/export_vietocr.py` trên máy có PyTorch rồi đưa lên máy chủ bằng `--from <thư mục>`.
Script kiểm SHA-256 từng file; các file mới của một mô hình chỉ thay file cũ khi đã đủ và đúng
cả (tải hỏng giữa chừng thì mô hình giữ nguyên bộ file cũ, không lẫn cũ với mới). Dịch vụ đang
chạy tự thấy mô hình mới chép vào, không cần restart; mô hình hỏng khi đọc (file lệch nhau) thì
tắt tới lần restart sau, trang vẫn chuyển bằng chữ Tesseract. Không có mô hình (hoặc
`P2W_VIETOCR=0`) thì công cụ chạy như trước, chỉ Tesseract.

**Đọc công thức** (nên có cho đề Toán/Lí scan): mô hình pix2text-mfr-1.5 (giấy phép MIT, ONNX
~120 MB, chạy bằng onnxruntime) đọc vùng công thức của trang OCR thành LaTeX. Kết quả tin được
thì thay chữ Tesseract đọc sai: số mũ/chỉ số dưới thành chữ chỉ số trên/dưới ("6,42.10⁻²¹ J."),
căn, phân số thành Equation sửa được trong Word; kết quả không chắc thì giữ chữ OCR. Tìm vùng ở
**phương án trả lời** (sau nhãn "A."–"D.", kể cả nhãn Tesseract đọc sai hay bỏ sót), **công thức hai
tầng** ở bất cứ đâu (phân số, căn, gạch trên) và **chữ có số mũ/chỉ số dưới trong câu dẫn**
("6,02.10²³ mol⁻¹", "g/m³", "n₁", "x_A"); chữ cái đứng một mình ("p", "k", "T") vẫn là chữ OCR. Lấy mô hình
về `models/mfr/` bằng cùng script (tải từ địa chỉ ghim theo phiên bản trong `scripts/models.json`);
không có mô hình, `P2W_MFR=0` hoặc `P2W_MATH_DETECT=off` thì trang OCR như trước.

Biến môi trường tuỳ chọn: `P2W_DATA_DIR`, `P2W_FILE_TTL` (giây, mặc định 3600),
`P2W_MAX_UPLOAD_MB` (50), `P2W_MAX_PAGES` (50), `P2W_WORKERS` (2), `P2W_MAX_STORAGE_MB` (4096),
`P2W_MAX_PENDING` (20), `P2W_RATE_UPLOADS` (40), `P2W_RATE_TASKS` (120), `TESSERACT_CMD`,
`P2W_VIETOCR` (`0` để tắt), `P2W_VIETOCR_DIR` (`models/vietocr`), `P2W_VIETOCR_THREADS` (2),
`P2W_MFR` (`0` để tắt đọc công thức), `P2W_MFR_DIR` (`models/mfr`), `P2W_MFR_THREADS` (2),
`P2W_MATH_DETECT` (cách tìm vùng công thức: `rules` mặc định, `off` để tắt).

Bảo vệ khi chạy công khai:
- Giới hạn theo IP người dùng (`CF-Connecting-IP` do Cloudflare Tunnel đặt; máy chủ chỉ nghe
  127.0.0.1): 40 lần tải lên, 120 lần xử lý mỗi 10 phút. Hàng đợi tối đa 20 việc; mỗi file một
  việc chạy cùng lúc, tối đa 6 lần. Tổng file lưu tối đa 4 GB. Trang dài hơn 3000pt bị từ chối;
  trang hơn 20.000 nét vẽ giữ thành ảnh.
- Header: CSP chỉ cho script/CSS của chính trang (`static/app.js`, `app.css`), cấm nhúng iframe,
  nosniff, HSTS; `/v1/*` có `Cache-Control: no-store` (Cloudflare mặc định lưu đệm .pdf/.docx).
  Không có `/docs`, `/openapi.json`. File lưu với quyền 600.
- `deploy/pdf2word.service`: hộp cát systemd (chỉ ghi được `data/`, `PrivateTmp`,
  `NoNewPrivileges`…) và `MemoryMax=80%`; `scripts/deploy.sh` cài lại file này khi nó đổi và tự trả
  lại bản cũ nếu dịch vụ mới không lên.

Cập nhật bản chạy trên máy chủ (dịch vụ systemd `pdf2word`): `./scripts/deploy.sh`. Script
`git pull`, cài lại thư viện khi `requirements.txt` đổi, cài lại dịch vụ khi
`deploy/pdf2word.service` đổi và chỉ restart khi `app/` hoặc dịch vụ đổi (giao diện `static/` đọc
thẳng từ ổ đĩa); `--restart` để restart dù không có gì đổi.

## Luồng API

| Bước | Request |
|---|---|
| Xin URL upload | `POST /v1/upload-url` → URL ký HMAC, hết hạn 15 phút (kiểu presigned URL của object store) |
| Upload | `PUT /v1/files/<token>?expires&sig` |
| Phân tích | `POST /v1/tasks` `{"tool":"analyse-document"}` |
| Convert | `POST /v1/tasks` `{"tool":"pdf-to-word","options":{"ocr_type":"ocr","languages":["vie"]}}` |
| Poll | `GET /v1/tasks/<task_id>` → `status`, `progress`, `result.output_files`, `result.error_fe` |
| Tải về | `GET /v1/files/<token>?name=...` |

Giao diện hỏi trạng thái mỗi 0,7 giây và chỉ bỏ cuộc ("Xử lý quá lâu…") khi 10 phút liền không có
trang nào xong thêm; thời gian chờ trong hàng đợi không tính, nên file scan dài có đọc công thức vẫn
nhận được kết quả.

`task_id` do client sinh (32 ký tự hex). Mã lỗi `error_fe` cho giao diện: `password`,
`corrupt_pdf`, `no_pages_in_pdf`, `too_many_pages`, `no_text_found`,
`internal_server_error`. Mở trang với `?debug` để thấy mục "Nhật ký API" (toàn bộ request/response).

## Chế độ (`ocr_type`)

| `ocr_type` | Trang có text | Trang scan | Ảnh lớn chứa chữ trên trang có text |
|---|---|---|---|
| `""` (cơ bản) | dựng từ lớp text | giữ dạng ảnh | giữ dạng ảnh |
| `"ocr"` (cũng nhận `"solid-ocr"`) | dựng từ lớp text | OCR | OCR (nếu ≥15 từ, tin cậy ≥70%) |
| `"ocr-full"` | OCR (bỏ qua lớp text, dùng khi text bị lỗi font) | OCR | – |

Thêm `"strip_watermarks": true` vào `options` (ô "Bỏ watermark…" trên giao diện, bật sẵn) để bỏ:

- ảnh nền nằm sau chữ là watermark: lớn (từ 2% trang), chứa chữ xoay, hay vẽ màu nhạt / trong
  suốt (`Picture.watermark`); nét mảnh màu đậm sau chữ (gạch phân số, kẻ dòng) giữ lại;
- chữ bán trong suốt (dấu đóng của các trang chia sẻ tài liệu); trên trang scan là chữ xám
  nhạt không sắc độ, và chỉ xét ở vùng header/footer vì chữ màu trong scan xám trông giống hệt;
- link ở header/footer: chữ có link annotation bắt đầu tại nó, hoặc chữ dạng URL/tên miền;
- phần watermark lộ qua hình minh hoạ có nền trong suốt: ảnh watermark được bỏ khỏi trang khi
  vẽ hình (trừ ảnh chính là hình); watermark vector và chữ xoay thì biết màu và độ mờ nên chỉ các
  điểm ảnh nằm đúng trên đường pha trắng → màu watermark mới được tẩy trắng; trên hình cắt từ
  trang scan thì tẩy trắng các điểm ảnh nhạt.

Số trang ("Trang 1/2") luôn được giữ; hình minh hoạ và nội dung đề không bị đụng tới. Mặc định
tắt.

## Pipeline

- **analyse-document** (`app/analyze.py`): PyMuPDF đếm ký tự text và diện tích ảnh từng
  trang, phân loại `text` / `scan` / `mixed`. Giao diện tự chọn OCR khi file có trang `scan` (hoặc không có chữ); đề có chữ kèm hình
  (`mixed`) giữ chữ sẵn có. Ngôn ngữ OCR và OCR cả trang nằm trong "Tuỳ chọn nâng cao".
- **Trang có text** (`app/native.py`), đọc `rawdict` của PyMuPDF:
  - Mỗi span thành một run giữ font, cỡ, đậm, nghiêng, màu; số mũ/chỉ số dưới nhận theo
    cỡ chữ và độ lệch baseline. Gạch chân và tô nền xét **từng ký tự**: gạch chân là nét kẻ
    mảnh (cả hình chữ nhật tô đen cao ~0,8pt như Word vẽ) dưới tâm ký tự – Word gạch riêng
    "C." trong "C. 273 K." –, tô nền là hình chữ nhật màu đặc cao khoảng một dòng sau chữ
    (thành `w:highlight` nếu đúng màu highlight của Word, không thì `w:shd`; ô tô màu của bảng
    kẻ khung không tính).
  - Font tiếng Việt cũ (`app/vnfont.py`): VNI và TCVN3/ABC nhận theo tên font (`VNI-*`,
    `.Vn*`), VNI còn nhận theo ký tự đặc trưng; chuyển sang Unicode và đổi font.
  - Các mục cùng baseline (đáp án A. B. C. D.) thành tab stop; tab của Word có khi chỉ là một
    khoảng nhảy hẹp hơn 1em (phương án B dài, "C." đứng sát) hay một dấu cách rất rộng – nhảy
    ≥1,5em, hoặc ≥0,4em ngay trước nhãn "C." / "c)", hay dấu cách rộng gấp 2,5 lần dấu cách
    thường của dòng thì cũng tách mục. Dòng xuống dòng được nối lại thành đoạn (dòng ngắn
    lại vì chừa chỗ cho hình bên cạnh vẫn nối; dòng mở đầu bằng "Câu 3." / "B. " luôn mở đoạn
    mới); khối đặt cạnh nhau (đầu trang hai cột) – chung ít nhất nửa dòng – thành bảng không viền;
    đầu trang mà mỗi cặp dòng là một cặp khối riêng ("SỞ GIÁO DỤC…" | "ĐỀ KIỂM TRA…") vẫn gộp
    thành một bảng.
  - Bảng kẻ khung: trước hết đọc thẳng từ nét kẻ (nét ngang/dọc, kể cả hình chữ nhật tô dày 0pt
    Word hay vẽ, nối nhau thành lưới có khung kín; có chữ trong ít nhất một ô; không có đường
    cong/xiên bên trong – đó là đồ thị). Thiếu nét giữa hai ô thì là **ô gộp** (bảng Đúng/Sai có
    "Câu 1" gộp 5 hàng thành `vMerge` của Word). Chỗ khác dùng `page.find_tables()` (PyMuPDF bỏ
    sót bảng đáp án một hàng và đọc nhầm lưới Đúng/Sai thành bảng 1×3), khung nới ra theo nét kẻ
    bên trong bảng (PyMuPDF có khi cắt ô ngoài theo chữ: kẻ 63–532pt mà chỉ báo 76–339pt, phần nét
    kẻ còn lại thành ảnh bảng thứ hai). Màu nền ô là hình tô màu phủ ≥60% ô; ô căn giữa khi mọi
    dòng chữ cách hai viền như nhau và có dòng chừa trống >15% bề ngang mỗi bên, căn phải khi chữ
    sát viền phải còn bên trái trống >30% (dòng đầy ô không cho biết gì nên để trái). Lề trong ô
    lấy theo khoảng viền → chữ của ô canh trái (0,5–5,4pt; mặc định 5,4pt của Word làm cột hẹp bẻ
    "Đúng" thành "Đún/g"). Nét kẻ của bảng không bị coi là gạch chân chữ.
  - Công thức không có chữ dùng được – MathType/Equation mà Word xuất ra PDF đánh số glyph
    "!", "\"", "#"…, chữ toán Unicode (𝑥) – được cắt thành **ảnh nội dòng** 300 DPI đúng chỗ trong
    câu, hạ theo baseline (alt "Công thức (ảnh)"). Vùng công thức là nhóm glyph công thức cùng
    glyph Symbol/Cambria Math kề bên (cùng dòng: cách ≤0,35em; chồng nhau như cận tích phân: chỉ
    glyph nhỏ hơn nằm giữa glyph lớn) và nét vẽ dính vào (gạch phân số, căn, mũi tên, vạch trị
    tuyệt đối "|x|"). Gạch phân số giữ tử và mẫu đặt gọn trên/dưới nó thành một khối, kể cả chữ
    thường trong phương trình Word ("Năng lượng nhận được", "1/110" Cambria Math); gạch chân thì
    cắt qua hộp chữ nên không bị coi là gạch phân số. Hai dòng công thức xếp sát nhau vẫn là hai
    ảnh. Khối chồng lên nhau gộp làm một, chữ nằm gần trọn trong khối thuộc về khối (không viết lại
    lần nữa cạnh ảnh). Ảnh chỉ có nét của chính công thức trên nền trong suốt: watermark, nền ô,
    chữ xoay chéo phía sau và mép dòng bên cạnh đều không lọt vào. Dấu câu ngay sau công thức
    (")?", ".") nằm ngoài khối chữ của công thức vẫn là chữ; số và nhãn "(1)", "20%", "(1) 6273." là chữ. Dòng
    bị PDF cắt quanh công thức được nối lại; hai dòng của PDF chen nhau trên cùng đường chân chữ (số
    mũ "⁰" cùng ") được" là một dòng riêng bắt đầu giữa "J/kg" và "C") được ghép theo thứ tự ngang,
    không thành tab; mảnh sát nhau ghép thẳng, không chèn tab. Dòng công
    thức rộng cạnh hình không bị coi là nhãn của hình.
  - Ảnh raster và hình vector thành ảnh nổi đúng vị trí trên trang; nhãn chữ trong hình hoặc sát
    mép (tên trục "s (km)", vạch chia "O", "2 4 5"; ở trên/dưới hình thì phải nằm trong bề ngang
    hình – "PHẦN I." trên bảng là tiêu đề) được render cùng hình; hình lớn đè lên chữ
    (watermark) đặt sau chữ; dòng chỉ chạm mép trên/dưới hình (phương án "A. … D." ngay dưới đồ
    thị) không làm hình thành nền, trừ khi trong hình có cả câu dài; đường kẻ mỏng (kẻ trên chân
    trang) chỉ nhận chữ sát nó (số trang), không nhận dòng cuối bài ngay trên. Hình (có chữ bao
    quanh) neo vào đoạn đầu tiên nằm ngang mép trên của nó (chung ít nhất 8pt, hoặc nửa đoạn, và
    bắt đầu không thấp hơn mép trên quá 8pt – dòng phương án dưới đồ thị không làm neo), không có
    thì đoạn ngay trên nó, và đặt theo **đoạn đó** (không theo trang): hình đi theo câu hỏi cả khi chữ
    dời sang trang khác. Khoảng trống chứa hình giữ đủ chiều cao (khoảng khác tối đa 72pt). Ảnh nền
    và nét mảnh (không có chữ bao quanh) vẫn neo vào đoạn đầu trang, đặt theo trang. Hình lấn ≤4pt
    vào khung dòng ngay trên/dưới được dời (hay thu nhỏ ≤15%) cho vừa khe, nếu không Word bắt dòng
    đó chạy quanh hình và đẩy chữ xuống. Mấy ảnh xếp sát nhau trên một dải (dòng chữ WordArt xuất
    thành từng mẩu, chữ thiếu dấu để là chữ thật đè lên) gộp thành một hình. Chữ dưới 4pt (chữ
    trắng 3pt để đánh dấu) bị bỏ.
  - Chữ xoay (watermark là tên trang chia sẻ đề) được set lại đúng gốc, góc, độ dài, độ mờ
    rồi đưa vào ảnh nền sau chữ.
  - Chữ bán trong suốt (`alpha` của span) được trộn với nền trắng thành màu tương ứng. Font
    Type3 (không có tên, không có cờ đậm): độ đậm suy ra từ độ dày nét so với chữ thường và
    chữ đậm cùng trang.
  - **Đầu/chân trang chỉ giữ "Trang x/y"**: mọi thứ lặp lại ở dải đầu/cuối trang (8% chiều cao)
    bị bỏ – dòng chữ và ảnh (logo, khẩu hiệu, dòng liên hệ, tên trang web) có trên hầu hết các
    trang, hoặc trên mọi trang lẻ / mọi trang chẵn của sách in hai mặt (logo trang lẻ, khẩu hiệu
    trang chẵn, số trang ở mép ngoài). Thay vào là một dòng "Trang " + trường PAGE + "/" + trường
    NUMPAGES ở chỗ PDF đặt số trang (giữa chân trang nếu PDF không có hay số trang đổi bên); y
    giữ số của PDF khi nhãn ghi tổng khác số trang ("Trang 1/4" trong file 6 trang có 2 trang đáp
    án). Thêm/bớt trang trong Word thì số trang tự cập nhật. Đoạn đứng ngay sau một dòng bị bỏ
    (đầu/chân trang, chữ watermark, link) lấy khoảng trước đoạn từ phần còn lại phía trên nó (0 ở
    đầu trang), không từ dòng đã bỏ: chữ chạy liền nên khoảng đó sẽ thành khe hở giữa trang.
  - Khoảng dòng và khoảng cách đoạn lấy từ baseline thật. Độ rộng "tự nhiên" của mỗi dòng
    (tổng bề rộng ký tự) cho biết dòng nào nhà sản xuất PDF đã nén khoảng trắng; đoạn đó
    được thụt lề phải âm vừa đủ để Word ngắt dòng y như bản gốc.
- **Trang scan / OCR** (`app/ocr.py`):
  1. Render 300 DPI; xoay trang về đúng chiều (Tesseract OSD, 90°/180°/270°) và chỉnh nghiêng
     (projection profile, ±5°, chính xác ~0,1°).
  2. OpenCV tìm bảng có kẻ khung, xoá nét kẻ và mực nhạt (watermark, nhiễu) khi chữ đủ đậm.
     Màu nền ô đo trên ảnh màu: màu giấy của ô (phần sáng chiếm đa số, bỏ mực) so với màu giấy
     trong dải quanh bảng, nên giấy ngả vàng hay xám vẫn ra ô trắng; ô tô màu khi đậm hơn giấy
     ≥10/255 ở một kênh (xám nhạt nhất của Word F2F2F2 là 13) – ảnh chụp sáng tối không đều thì
     ngưỡng cộng thêm độ chênh giữa các phía của dải. Căn giữa/phải trong ô như với trang có text.
  3. Tesseract xuất hOCR: block, đoạn, dòng (baseline, `x_size`, ascender/descender), từ;
     bỏ từ rác (dòng chấm dưới watermark, vệt watermark đọc thành chuỗi ký tự lặp).
     Có mô hình VietOCR và ngôn ngữ có `vie` (`app/vietocr.py`): mỗi từ cắt theo cột của từ, hàng
     của dòng, đệm 1/4 chiều cao dòng, VietOCR đọc lại (từ cùng chiều rộng đọc chung một lượt).
     Cùng chữ cái (bỏ qua dấu thanh) thì lấy dấu của VietOCR, giữ chữ hoa/thường của Tesseract,
     trừ khi VietOCR bỏ dấu của từ Tesseract tin ≥90 ("khí" → "khi" ở từ ngắn); khác chữ thì lấy
     chữ VietOCR khi Tesseract tin <92, VietOCR ≥0,5, không thêm "?" và giữ đủ ký hiệu
     `= + ° % / < > ( ) ^` và dấu trừ của Tesseract (VietOCR không học công thức: "R=" → "?";
     từ điển của nó không có "°", nên "?" đúng chỗ Tesseract có "°" được đọc là "°"). Ô cắt một
     từ không cho thấy đầu câu hay cuối câu ("lần." → "Lần"): chữ VietOCR giữ hoa/thường của các
     chữ cái trùng với chữ Tesseract tin ≥60, không bỏ dấu câu hay ngoặc Tesseract đọc được
     ("T,;" → "T,": giữ chữ Tesseract; dấu đọc thành dấu khác hay thành chữ, "0.25" → "0,25", thì
     được), dấu . , ; : cuối từ VietOCR bỏ sót thì thêm lại ("Lần" → "lần.", "Nan3" → "NaN3.").
     Ô cắt rộng hơn khoảng cách giữa hai từ nên có mẩu chữ của từ bên cạnh: VietOCR đọc thêm một từ
     ở đầu hay cuối mà Tesseract không đọc thấy chữ nào, ở phía có từ khác của dòng lấn vào ô cắt
     ("chuyển" → "ố chuyên", "8m" → "8 m về" khi khung "8m" chồng lên "về"), thì giữ chữ Tesseract;
     không có từ nào ở phía đó thì đó là nét trong khung Tesseract bỏ sót ("B." → "B. v"), vẫn lấy;
     tách chữ dính ("B.343m/s." → "B. 343 m/s.") cũng lấy.
     Sửa lỗi theo cấu trúc đề (`app/ocr_fix.py`), chỉ chỗ cấu trúc tự quyết được, không chọn giữa
     hai từ có thật ("chuyên"/"chuyển"):
     - nhãn phương án dính chữ "A.0K." → "A. 0K.", ".." → ".";
     - số đọc thành chữ "I0O" → "100" (cần ≥3 ký tự hoặc mở đầu bằng số: "O2" giữ);
     - số dính đơn vị/chữ "2mol", "4giờ" → "2 mol", "4 giờ" ("2NaN3", "3x" giữ);
     - âm tiết dính nhau "phântử", "sựnóngchảy", "nhiệtđộcủakhí" (tới 5 âm tiết) → tách theo cấu
       tạo âm tiết tiếng Việt, chỉ khi mọi phần đều có dấu tiếng Việt (từ mượn "canô", "nitơ",
       "vectơ", "điốt" giữ nguyên);
     - "PHẦN" + số La Mã theo thứ tự các phần ("PHẦN L." → "PHẦN I.", "PHAN H." → "PHẦN II.");
     - thẻ lặp nhiều lần "[ĐNH]" sửa bản lệch một chữ "[ĐNHỊ]";
     - bảng: cột số thứ tự (1, 2, 3…) sửa ô lệch ("1\"" → "11", "l" → "1"), cột đáp án A–D và
       cột Đ/S ("SDĐĐ" → "SĐĐĐ").
     Dòng Tesseract tách khỏi một hàng in ("A." thành khối riêng, đứng sau hàng "B. … C. … D.")
     được nhập lại theo thứ tự x: hai dòng cạnh nhau (không chồng theo chiều ngang, chung ≥ 1/2
     chiều cao dòng thấp và ≥ 40% dòng cao, đường chân chữ lệch ≤ 0,15em), một dòng là cả khối của
     nó và nằm trong cột của dòng kia (khối dòng kia trải qua nó, hay cũng chỉ một dòng: cột chữ là
     khối nhiều dòng); dải đầu/chân trang giữ nguyên.
     Có mô hình công thức: tìm vùng công thức (`app/mathregions.py`, cách `rules`, theo thứ tự ưu tiên):
     - phương án: phần sau nhãn "A."–"D." của dòng có phương án, kể cả nhãn đứng một mình; nới theo
       mực tới khoảng trắng, qua gạch phân số tới tử/mẫu; bỏ vùng có chữ tiếng Việt Tesseract tin
       ≥80 và vùng tràn sang phương án hay dòng khác. Nhãn Tesseract đọc sai ("c.v=a.", "p.v=—.")
       hay bỏ sót được tìm trên nét chữ: ở cột nhãn của trang, chữ cao bằng nhãn kèm dấu chấm, trước
       nó là giấy trắng, cùng màu nhãn (nhãn đen: cùng độ đậm); chữ của nhãn lấy theo thứ tự trong
       hàng (hàng có từ hai nhãn trở lên thì nhãn chưa rõ chữ lấy chữ cột của nó luôn có). Từ Tesseract
       đọc mở đầu bằng chữ số ("2." của dòng đánh số, "1.2,3" của bảng đáp số) không phải nhãn. Vùng
       đọc được thì nhãn được viết lại đúng ("C.", cả "c." viết thường);
     - công thức hai tầng: gạch phân số (nét mảnh, phẳng, có chữ ngay trên và dưới), dấu căn, gạch
       trên chữ (Ā, ΔL̄); nới theo mực sang hai bên tới khoảng trắng 0,6em, dừng ở nhãn và từ
       Tesseract tin là chữ thường (tiếng Việt, từ ≥3 chữ cái không phải đơn vị), cả nhãn Tesseract
       đọc sai hay thiếu dấu chấm ("c.x=—", "D": chữ hoa kèm dấu chấm sau giấy trắng) – nhãn đó được
       viết lại thành chữ trước Equation; mô hình vẫn đọc nhãn vào kết quả ("D.\frac{3}{2}") thì nhãn
       cũng tách ra thành chữ, không nằm trong Equation; kết quả vào dòng ngang tầm gạch phân số có chữ
       gần vùng nhất theo chiều ngang (trang hai cột: không vào dòng của cột bên kia dù đường chân chữ
       của dòng đó sát hơn); luôn là vùng "cao": không đọc được thì thành ảnh công thức;
     - chữ trong câu: từ có số mũ hay chỉ số dưới đo theo hình (nét nhỏ cao hơn chữ hoa hay thấp
       hơn đường chân chữ ngay sau nét chữ khác, không phải dấu thanh, dấu phẩy hay dấu độ), từ chỉ
       có số mũ ghép vào từ trước; từ Tesseract tin <50 có số thập phân hay dấu toán trên dòng chữ;
       chữ Δ kèm một chữ cái mà Tesseract đọc thành "Ax", "At", "Aa." (nét đầu là tam giác cao bằng
       chữ hoa, lỗ trong rộng và kín tới gần đáy; lỗ của chữ A, số 4 chỉ tới nửa chiều cao).
       Hoa/thường của chữ đầu vùng lấy theo chiều cao nét ("Vh" → "vh") ở các chữ chỉ khác nhau
       về cỡ (c o s u v w x z; không có k: nét lên của k cao bằng chữ hoa).
     Mô hình đọc mỗi vùng 2–3 lần với lề cắt khác nhau, chỉ lấy kết quả
     các lần đọc khớp nhau và qua luật an toàn (`app/mathocr.py`: không ký hiệu lạ, chữ số khớp
     Tesseract theo thứ tự – mô hình bớt hay thêm một chữ số ("200" cho "2000", "500 00" cho
     "500 000", "2ω" cho "20"), kể cả chữ số ngoài phân số/căn ("7√3" cho "2/3"), là bỏ; số mũ
     Tesseract đọc thành chữ số thường, dấu độ đọc thành "0" thì không…; mô hình không có chữ tiếng Việt nên kết quả viết một từ của vùng không dấu, "và" →
     "va", cũng bị bỏ). Các lần cắt chỉ khác lề nên mô hình có thể sai giống nhau ở mọi lần: dấu
     trừ đầu số mũ thành ký hiệu lạ ("7,66.10^{\mu^{22}} J."), chỉ số "1" mảnh cạnh dấu ";" thành
     "\i" ("T_{\i};"). Vùng chưa có kết quả mà có lần đọc dùng được (hay sửa được: ký hiệu lạ → "-")
     thì đọc thêm phần trước từ ngắn cuối cùng (đơn vị, dấu câu): lần đọc đó trùng lần đọc kia tới
     chỗ đơn vị/dấu, và Tesseract cũng đọc được đơn vị/dấu đó ở cuối, thì tính là một phiếu. Trước
     khi bỏ phiếu, mỗi lần đọc nhận điều nét chữ cho biết rõ hơn mô hình (mô hình thấy vùng cắt bị
     kéo thành hình vuông): hệ số mà mô hình đọc chữ sau nó vào chỉ số ("4_{VO}") được viết lại
     "4v_{O}" khi nét chữ có đúng chừng ấy cụm số, chữ cao bằng chữ thường đứng trên đường chân chữ,
     nét hạ thấp; chữ đầu vùng mà chiều cao nét đã cho biết hoa/thường lấy theo nét ("V_{H}" →
     "v_{H}"), không bị bỏ vì khác Tesseract. Vùng đọc được: bỏ mọi từ OCR có tâm trong vùng (giữ
     nhãn), đặt kết quả vào chỗ đó (chữ thường + chỉ số trên/dưới, hay Equation + đơn vị, dấu chấm,
     dấu ngoặc đóng của câu sau nó); vùng cao (phân số, căn) không đọc được thành ảnh công thức
     trên dòng chữ như trang có text; vùng khác giữ chữ OCR.
     Chữ của công thức không đậm, không màu theo đoạn.
  4. Giữ hình: mực không phải chữ đáng tin, không phải nét kẻ mảnh (hình vẽ, ảnh, con dấu,
     chữ ký) thành ảnh nổi cắt từ bản màu; nhãn ngắn sát hình đi cùng hình.
  5. Dựng layout: tách đoạn theo dòng kết thúc sớm / thụt đầu dòng / hết câu + chữ hoa; khoảng
     trống ≥1em thành tab stop (hàng đáp án). Nhãn dính phương án ("A.6kPa." một từ Tesseract)
     tách tại khoảng trống rộng cỡ dấu cách hoặc chỗ mực màu chuyển đen, để đo riêng đậm/màu.
     Màu chữ theo từng từ, lấy từ phần mực đậm nhất (mép chữ pha giấy); màu gần nhau trong trang
     gộp một ("Câu" 0000FF, "[ĐNH]" FF0000, nhãn 3366FF như bản gốc). Chữ đậm theo từng từ (tỉ
     lệ nét còn lại sau erosion ~0,11em, cỡ chữ không nhỏ hơn 0,9 lần cỡ thân bài – Tesseract đo
     dòng "A. 6 kPa." nhỏ đi nên nét thường còn lại như chữ đậm); nhãn "A."–"D." và "Câu" của một
     trang đậm theo số đông, số trong câu thường ("7,5", "200") không đậm trừ khi cả dòng đậm, số
     của "Câu 3:" đậm theo "Câu"; cỡ chữ thân bài tính từ bề rộng dòng (thứ quyết định ngắt dòng trong
     Word), đối chiếu với chiều cao glyph; dòng lặp lại ở đầu/cuối trang bị bỏ như trang có text
     (so khớp bỏ qua khoảng trắng và ký tự gạch chéo OCR đọc lệch như "1⁄2"). Số trang OCR đọc
     sai khác nhau giữa các trang ("Trang" / "Trang 2/2") vẫn được nhận: chữ tĩnh bị bỏ, vị trí
     của nó là chỗ đặt "Trang x/y". Đề in hai mặt (tiêu đề trang lẻ khác trang chẵn): khi một loại
     trang chỉ có một trang, chỉ bỏ dòng nằm trong dải tiêu đề/chân trang đã tìm thấy – đề 3 trang
     OCR đọc tiêu đề trang 2 khác đi từng làm câu đầu trang 2 bị coi là "tiêu đề trang chẵn".
- **Ghi DOCX** (`app/docx_builder.py`): khổ giấy, lề, header/footer, ảnh nổi (`wp:anchor`),
  bảng, tab stop, khoảng dòng "at least"/"exactly", tắt widow/orphan control. Lề dưới chừa 10pt
  dưới dòng thấp nhất của trang đầy nhất (Word dàn cao hơn PDF vài pt là dòng cuối đã sang trang
  mới). Lề phải là mép xa nhất mà ít nhất hai khối chữ chạm tới (một dòng PDF lấn lề không làm
  rộng mọi dòng).
  **Sang trang:** chỉ trang PDF kết thúc sớm (cuối trang trống hơn 6 dòng), trang trống hay trang
  mở đầu bằng hình mới có "ngắt trang trước đoạn". Trang chỉ sang vì trang trước đã kín thì chữ
  chạy liền, nên sửa hay "Chuẩn hóa đề" không để lại trang gần trống. Để bản chưa sửa vẫn sang
  trang đúng chỗ PDF, vài đoạn mở trang được "giữ với đoạn sau" / "giữ các dòng cùng nhau" vừa đủ
  không lọt vào chỗ trống cuối trang trước (cộng cả 10pt lề chừa). Word bỏ chuỗi giữ khi đoạn giữa
  chuỗi mang ảnh nổi, nên hình đó neo vào đoạn đầu chuỗi. Khi mép trên một hình rơi vào khoảng
  hở giữa hai đoạn, khoảng hở ghi thành "khoảng sau" của đoạn trên (Word kéo cả "khoảng trước"
  của dòng chạm hình xuống ngang mép hình).

## Kiểm thử

```bash
.venv/bin/python scripts/make_test_pdfs.py   # sinh samples/ (cần font Times New Roman của macOS)
.venv/bin/python scripts/smoke_test.py       # cần server đang chạy
.venv/bin/python -m unittest discover -s tests  # kiểm thử đơn vị (VietOCR, công thức, tải mô hình)
```

`exam-like.pdf` kiểm tra các định dạng phải giữ: dòng VNI, số mũ/chỉ số dưới, màu, tab đáp
án, hình nổi, khối đầu trang hai cột, header và trường số trang.

Kết quả hiện tại (M4 Pro):
- File đề thi thật 2 trang (có font VNI, watermark, hình): ~0,7 giây; khớp 98,4% ký tự /
  99,3% từ với lớp text gốc; Word render đúng 2 trang, ngắt dòng
  như bản gốc.
- File scan tiếng Việt 2 trang: ~2 giây, độ tin cậy Tesseract ~94%, khớp ~93% theo từ;
  bản nghiêng 2° và bản xoay 90° cho kết quả tương đương.
- Bản scan của file đề thi (200 dpi, có watermark): khớp 94,3% ký tự / 88,7% từ, giữ hình cờ lê
  kèm nhãn, Word render đúng 2 trang.
- Đề Vật lí 10 gõ MathType, Word xuất PDF 6 trang: ~1,2 giây, ~145 ảnh công thức; khớp 98,7% từ
  với file Word gốc (công thức tính là một từ); Word render đúng 6 trang; trang đọc đề đọc đủ 28/28
  đáp án.

Chọn model: trên mẫu test, `tessdata_best` cho ~99% ký tự, còn model chuẩn ghép `vie+eng`
chỉ ~87–90% (mất dấu). Khử nhiễu, nhị phân hoá hay tăng DPI không cải thiện thêm. Lọc mực
nhạt ở mức 215 tăng ~1,3% độ khớp theo từ trên bản scan có watermark; mức 170 làm giảm.

Sửa lỗi sau OCR (2026-10-26), đo trên 30 trang scan giả từ 5 đề (300 dpi tốt / 200 dpi kém),
khớp theo từ: Tesseract 89,6% / 88,5% → 92,2% / 91,5%. Bảng đáp án scan đúng 62/62 ô (trước
61/62); đề scan kém đọc lại đúng tổng điểm (trước 6–7/10 vì tiêu đề phần đọc sai).

VietOCR đọc lại từ (2026-10-27, mô hình ONNX float16): cùng 30 trang, khớp theo từ
92,2% / 91,5% → 96,5% / 95,7%, mọi file tăng 1,6–5,6 điểm; đề Vật lí scan 3 trang (300 dpi):
chữ 96,9% → 98,4%, số 67,5% → 78,1%, không thêm dấu "?" nào. Hai điều kiện an toàn (không "?",
giữ ký hiệu) bớt ~0,5 điểm: từ điển VietOCR không có số mũ nên "g/cm³" thành "g/cm?". Bản ONNX
đọc như VietOCR chạy PyTorch: cùng chữ 9325/9329 từ thử (bản float32: 9329/9329). Chi phí trên
CPU M4 Pro, 2 luồng: thêm ~2 giây/trang (trang dày ~400 từ: ~3 giây), RAM thêm ~400 MB.
Giữ hoa/thường và dấu câu của Tesseract (2026-10-27, có đọc công thức): 30 trang 97,56% /
96,96% → 97,68% / 97,06% ("lần", "NaN3"), không trang nào mất từ đúng.

Đọc công thức (2026-10-27, cách tìm vùng `rules`, có VietOCR): đề Vật lí scan 3 trang: chữ
98,4% → 99,3%, số 78,1% → 92,1%, Equation đúng 0 → 13/24 công thức của bản chuẩn (15/24 khi coi
dấu trừ Word viết "−" là "-"); 41/51 công thức và số mũ/chỉ số của bộ thử đúng (chỉ tìm ở phương
án: 27/51). Còn thiếu: chữ cái đứng riêng (p, ρ, k, T), ba phương án "4v_O = 9v_H" mô hình không
đọc được, v_H và T_1 mô hình đọc sai nên giữ chữ OCR. 30 trang scan giả: khớp theo từ 96,45% /
95,70% (chỉ VietOCR) → 97,56% / 96,96% (chỉ tìm ở phương án: 96,72% / 96,20%), không trang nào
mất từ đúng. Trên đề đó và hai bản scan giả của một đề Vật lí khác: 192 kết quả đặt vào đề, 147
ứng với công thức của bộ thử: 142 đúng hẳn, 5 đúng một phần ("t_2=0,405" còn " s" là chữ ngay
sau; "V=4/3πR³)" lấy cả dấu ngoặc đóng của câu); các kết quả còn lại là số kèm đơn vị
("594 m/s.") cũng đúng. Chi phí trên CPU M4 Pro, 2 luồng: ~0,2 giây mỗi vùng (đề Vật lí 86 vùng:
16,5 → 34 giây cho 3 trang khi máy bận), RAM thêm ~1 GB.

Sửa các chỗ sai thấy khi mở trong Word (2026-10-27): đề Vật lí scan 3 trang: chữ 99,3% → 99,6%,
số 92,1% → 93,4%, bộ thử 41/51 → 47/51 ("7,66.10^{-22} J.", "v_H", ba phương án "4v_O = 9v_H",
"T_1"; còn thiếu chữ cái đứng riêng p, ρ, k, T), 47/47 kết quả ứng với bộ thử đúng; Câu 6 "lần."
và hàng "A. … D." của Câu 19 đúng. Hai bản scan giả của đề Vật lí kia: kết quả ứng với bộ thử 101
đúng, 3 đúng một phần, 2 sai (")" trong Equation) → 111 đúng (thêm Δx, ΔA., Δt, Δs), 3 đúng một
phần, 0 sai. 30 trang scan giả 97,56% / 96,96% → 97,73% / 97,09%, không trang nào mất từ đúng;
PDF có text không đổi; tắt mô hình thì VL1212 chỉ khác main ở nhãn "A." vào lại hàng của nó.

Sửa theo rà soát mã (2026-10-27): kết quả trên đề Vật lí scan và hai bản scan giả như trước (chữ
99,6%, số 93,4%, 47/51; 47 + 111 kết quả ứng với bộ thử đúng, 0 sai), 30 trang scan giả 97,73% /
97,09%, không trang nào mất từ đúng, bớt chữ thừa VietOCR đọc từ từ bên cạnh ("về về", "PHẦN I. I.",
"ố chuyên"); PDF có text và VL1212 tắt mô hình không đổi. Trên trang thử dựng riêng: nhãn đọc sai
không còn nằm trong Equation, số thứ tự ở cột nhãn không thành "A.", căn bậc ba ra đúng, mô hình bớt
hay thêm một chữ số so với Tesseract thì giữ chữ OCR (bộ thử: 393/522 kết quả nhận, 0 sai).

## Giới hạn

- Bảng phải có kẻ khung đầy đủ; chưa hỗ trợ bảng không viền; màu viền chưa giữ (luôn đen); bảng
  scan (OCR) chưa có ô gộp.
- Mọi thông tin đầu/chân trang trừ số trang đều bị bỏ (cố ý): tên trường, tên đề in lặp mỗi
  trang cũng mất; dòng chỉ có ở trang đầu thì vẫn là nội dung.
- Bảng scan: màu nền ô lệch vài đơn vị so với bản gốc (ảnh chụp điện thoại lệch nhiều hơn); ô tô
  màu chỉ đậm hơn giấy dưới 10/255 (hay dưới độ chênh sáng của ảnh chụp) thì thành ô trắng.
- Công thức thành ảnh: xem và in đúng nhưng không sửa được trong Word, công cụ đọc đề cũng không
  đọc được nội dung (chưa dựng lại thành Equation). Công thức có chữ thật (PDF từ LaTeX, chỉ
  dùng font Symbol) vẫn là chữ.
- Đọc công thức (trang scan): chữ cái đứng riêng (p, ρ, k, T) và số thường trong câu dẫn vẫn
  là chữ OCR; công thức trong bảng cũng vậy. Không đọc phương án có chữ tiếng Việt Tesseract tin
  ≥80, vùng chạm dòng trên/dưới hay phương án bên cạnh, phương án mà một từ OCR chạy sang phương
  án sau ("C. 273 K. D. 373"); kết quả mô hình không chắc thì giữ chữ OCR (thường sai:
  "642.107'J."), phân số/căn/gạch trên không đọc được thành ảnh. Tesseract để nhãn đứng riêng
  một dòng mà đường chân chữ lệch hàng của nó quá 0,15em thì phương án đó vẫn thành đoạn riêng
  ngay sau hàng. Số mũ chỉ
  nhận theo hình khi tách khỏi nét chữ trước nó (scan nhoè dính liền thì không thấy).
- Trang scan: luôn dùng Times New Roman, chưa nhận in nghiêng/gạch chân; số mũ/chỉ số dưới chỉ có
  ở chỗ mô hình công thức đọc được;
  chữ đậm theo từng từ thỉnh thoảng sai ở từ rời; Tesseract không đọc được dòng gạch dài kiểu
  "------HẾT------"; hình mờ, nét rất mảnh có thể bị cắt sát.
- Watermark chữ xoay được set lại bằng font serif, không phải font gốc; font Type3 dùng font
  mặc định (màu và độ đậm thì giữ).
- Ảnh nền/watermark (khi không xoá) vẫn đặt theo trang, neo vào đoạn đầu trang PDF: sửa nhiều làm
  đoạn đó sang trang khác thì ảnh nền đi theo trang mới.
- Chữ chạy liền giữa các trang: Word dàn mỗi trang hơi khác PDF (dòng có công thức, chỗ xuống
  dòng), trang đầy nhất có thể tràn vài dòng sang trang sau (4 đề thử: mọi trang mở đúng chỗ như
  PDF; đề Vật lí trang 2 chỉ khác chỗ xuống dòng của câu cuối).
- Tẩy watermark trong hình: viền răng cưa của chữ watermark có thể còn vệt rất nhạt; trên hình
  scan, mảng xám nhạt của chính hình (bóng, tô nền) cũng bị tẩy theo.
- Danh sách a) b) c) cố ý giữ dạng chữ (dễ chuẩn hoá về sau), không chuyển thành danh sách
  tự đánh số của Word.
- Text nhiều cột thật sự được đặt vào bảng không viền, chưa dùng section nhiều cột.
- Chữ có dấu kép (ế, ồ, ấ...) là chỗ Tesseract hay sai nhất; sửa lỗi sau OCR không đụng tới
  (đọc nhầm thành một từ có thật khác). VietOCR sửa phần lớn khi có mô hình; từ Tesseract đã tin
  ≥92 mà sai chữ cái thì vẫn giữ.
- Dấu phẩy mất hẳn khi nhận dạng ("8,5" → "85") không khôi phục được (VietOCR lấy lại được một
  phần khi có mô hình).
- Đọc công thức: chỉ số bậc căn mô hình bỏ sót thì căn bậc ba thành căn bậc hai ("√(a−b)" cho
  "∛(a−b)"); chữ tiếng Việt Tesseract tin <80 sát phân số vẫn có thể bị gộp vào vùng công thức
  (vùng đó thành ảnh). Chữ số trong phân số không đối chiếu với Tesseract (Tesseract đọc tử/mẫu
  lộn xộn).
- File scan dài có đọc công thức chạy lâu (trên M4 Pro ~9 giây/trang khi chỉ có mô hình công thức):
  máy chủ chạy hết việc dù người dùng đã đóng trang; chưa giới hạn thời gian mỗi việc.
- Hàng đợi chạy trong process (ThreadPool), trạng thái task nằm trong RAM, file lưu ổ
  local: mất hết khi restart. Chưa có xác thực hay giới hạn tần suất (rate limit).

## Hướng lên production

- `storage.py` → presigned URL của S3/R2/MinIO (giữ nguyên contract HTTP).
- Worker tách riêng (Celery/RQ/Arq + Redis), trạng thái task trong Redis/DB.
- Engine OCR tốt hơn cho tiếng Việt: VietOCR đã đọc lại từng từ, còn tìm dòng/từ vẫn là
  Tesseract (thay bằng PaddleOCR), hoặc dịch vụ (Azure Document Intelligence, Google Document
  AI, ABBYY), hoặc một SDK chuyển đổi thương mại.
- Giới hạn CPU/RAM/thời gian cho mỗi task, quét virus file upload, xoá file theo TTL.

## Giấy phép

AGPL-3.0 (xem `LICENSE`), vì công cụ dùng PyMuPDF (AGPL-3.0). Chạy công cụ thành dịch vụ cho
người khác dùng qua mạng thì phải cung cấp mã nguồn (kể cả phần đã sửa) cho người dùng đó.

Thành phần khác: model trong `tessdata/` lấy từ
[tesseract-ocr/tessdata_best](https://github.com/tesseract-ocr/tessdata_best) (Apache-2.0);
Tesseract và OpenCV (Apache-2.0), python-docx và FastAPI (MIT), NumPy (BSD), ONNX Runtime
(MIT). Mô hình trong `models/mfr/` là pix2text-mfr-1.5 (MIT), dùng nguyên các file ONNX đã công
bố. Mô hình trong `models/vietocr/` là `vgg_transformer` của
[VietOCR](https://github.com/pbcquoc/vietocr) (Apache-2.0), chuyển sang ONNX bằng
`scripts/export_vietocr.py` (giữ nguyên trọng số, lưu float16). File PDF mẫu
không nằm trong repo: `scripts/make_test_pdfs.py` tạo lại vào `samples/`.
