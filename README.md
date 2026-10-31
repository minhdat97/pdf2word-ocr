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

VietOCR (nên có cho file scan tiếng Việt) đọc lại các từ Tesseract chưa chắc, sửa dấu và chữ
Tesseract đọc sai. Mô hình (`vgg_transformer` của VietOCR chuyển sang ONNX, trọng số float16,
76 MB, chạy bằng onnxruntime, không cần PyTorch) và mô hình công thức không nằm trong git, lấy về
`models/`:

```bash
.venv/bin/python scripts/fetch_models.py                   # tải theo scripts/models.json
.venv/bin/python scripts/fetch_models.py --from <thư mục>  # hoặc chép từ thư mục có sẵn các file
.venv/bin/python scripts/fetch_models.py --check           # chỉ kiểm tra
.venv/bin/python scripts/fetch_models.py --model mfr       # chỉ một mô hình
```

Các file tải từ bản phát hành GitHub `models-2026-10-28` của repo này (VietOCR float16 và int8, hai
nửa bộ giải mã công thức; trang phát hành ghi cách làm và giấy phép) và nơi công bố gốc của mô hình
công thức (ghim theo bản). `--from` nhận thư mục có các file dưới tên trong `scripts/models.json`
hay tên file của bản phát hành (`vietocr-encoder.onnx`…). Script kiểm SHA-256 từng file; file một
mô hình khác đã có (VietOCR int8 dùng chung `decoder.onnx`, `config.json` với bản float16) được chép,
không tải lại; các file mới của một mô hình chỉ thay file cũ khi đã đủ và đúng cả (tải hỏng giữa
chừng thì mô hình giữ nguyên bộ file cũ, không lẫn cũ với mới). Dịch vụ đang chạy tự thấy mô hình
mới chép vào, không cần restart; mô hình hỏng khi đọc (file lệch nhau) thì tắt tới lần restart sau,
trang vẫn chuyển bằng chữ Tesseract. Không có mô hình (hoặc `P2W_VIETOCR=0`) thì công cụ chạy như
trước, chỉ Tesseract. Đổi mô hình: đăng file lên một bản phát hành mới (`models-<ngày>`), đổi địa chỉ
và SHA-256 trong `scripts/models.json`; `scripts/deploy.sh` tải về.

Từ được đọc lại (`P2W_VIETOCR_WORDS`): mặc định `unsure` – từ Tesseract tin dưới 92 và từ có chữ
â ă ê ô không dấu thanh hay dấu huyền (Tesseract chắc mà vẫn hay sót dấu ở đó: "chuyên" thay "chuyển",
"thăng" thay "thẳng"), khoảng 30 % số từ; `all`: mọi từ. Trên 30 trang scan thử `unsure` cho số từ đúng
như `all` (từ chắc khác gần như không bao giờ được VietOCR sửa đúng) mà VietOCR chỉ tốn ~40 % thời gian.
Bản int8 (cho máy chủ nhỏ, `models/vietocr-int8`, `fetch_models.py` tải cùng bản float16): các lớp tích
chập của bộ mã hoá đổi sang int8 (`scripts/quantize_vietocr.py`, hiệu chỉnh trên chữ của các trang scan).
Mặc định dùng bản này khi thư mục đủ file (không thì bản float16; `P2W_VIETOCR_INT8=0` luôn dùng bản
float16; `P2W_VIETOCR_DIR` đặt thì luôn dùng thư mục đó). Số từ đúng như bản float16, 2 % số từ đọc khác.
Đo trên máy chủ 2 lõi ARM (2026-10-28, một luồng): 98 thay vì 211 ms mỗi từ.

**Công thức trên trang scan** có hai cách đọc; vùng công thức tìm như nhau (`app/mathregions.py`):
**phương án trả lời** (sau nhãn "A."–"D.", kể cả nhãn Tesseract đọc sai hay bỏ sót), **công thức hai
tầng** ở bất cứ đâu (phân số, căn, gạch trên) và **chữ có số mũ/chỉ số dưới trong câu dẫn**
("6,02.10²³ mol⁻¹", "g/m³", "n₁", "x_A"); chữ cái đứng một mình ("p", "k", "T") vẫn là chữ OCR.

- **Mặc định, không cần mô hình** (`app/scriptocr.py`): số mũ và chỉ số dưới của số, đơn vị và chữ
  ("6,42.10⁻²¹ J.", "mol⁻¹", "g/cm³", "4v_O = 9v_H", "n₁") được đọc bằng Tesseract: hình học tách đoạn
  nâng lên/hạ xuống, Tesseract đọc lại (hai lần, chỉ chữ số, chữ Latinh và vài dấu) rồi từng ký tự được
  đối chiếu với nét chữ; không chắc thì giữ chữ OCR. Chữ trong chỉ số có thể là chữ số hay Đ/đ (Tesseract
  đọc ở đây không có chữ Việt: R_Đ ra "R_O", U_Đ ra "U_b", F_đh ra "F_dh"; "1" mất chân ra "t") phải có trong
  chữ OCR của trang, "D" không bao giờ nhận; chữ hoa/thường của chỉ số (x/X, v/V, y/Y…) theo chiều cao nét,
  không rõ thì giữ chữ OCR; hai lần đọc khác nhau ở chữ số thì không lấy lần nào. Phân số, căn thành ảnh trong
  dòng. Nhãn A.–D. Tesseract đọc sai hay bỏ sót ("c.9") được viết lại cả khi vùng không đọc. Cần Tesseract
  4.1 trở lên (danh sách ký tự có dấu cách, `hocr_char_boxes`): chạy `tests/test_scriptocr.py` trên máy
  chủ, phép thử in một trang rồi đọc lại. `P2W_SCRIPTS_LANG` (`eng`): mô hình Tesseract đọc các dòng chỉ số.
- **"Chuyển thành công thức sửa được"** (trước gọi "Đọc kỹ"; ô của giao diện cho trang scan và nút sau lần
  chuyển nhanh; tuỳ chọn `"careful": true` của tác vụ, xem Luồng API; nó cũng đọc công thức dạng ảnh của
  trang có chữ, xem Pipeline):
  mô hình pix2text-mfr-1.5 (giấy phép MIT, ONNX ~120 MB khi chạy, onnxruntime) đọc vùng công thức thành
  LaTeX. Kết quả tin được thì thay chữ Tesseract đọc sai: số mũ/chỉ số dưới thành chữ chỉ số trên/dưới,
  căn, phân số thành Equation sửa được trong Word; kết quả không chắc thì giữ chữ OCR. `fetch_models.py`
  lấy về `models/mfr/` ba file đã công bố và hai nửa của bộ giải mã (`cross_kv.onnx`: khoá/giá trị
  cross-attention của một vùng, tính một lần; `decoder_step.onnx`; tách bằng `scripts/split_mfr_decoder.py`,
  tải từ bản phát hành): kết quả như bộ giải mã nguyên khối, phần đọc công thức nhanh khoảng gấp đôi.
  Thiếu một trong hai file thì dùng `decoder_model.onnx` (máy chủ chỉ có ba file cũ vẫn chạy).
  `P2W_MFR_SPLIT=0` (đặt trong drop-in systemd) quay về bộ giải mã nguyên khối dù có hai file tách; đừng
  xoá hai file: `deploy.sh` thấy thiếu là tải lại. Chậm (xem Máy chủ nhỏ) nên chỉ cho file có tối đa
  `P2W_MFR_MAX_PAGES` (5) trang scan và mỗi lúc chỉ một lần chuyển công thức (`P2W_CAREFUL_WORKERS`): lần
  khác chờ trong hàng đợi (trạng thái `queued`, giao diện ghi "Đang chờ đến lượt chuyển công thức…"), không
  giữ luồng nào, lần chuyển thường xếp sau nó vẫn chạy; tối đa `P2W_CAREFUL_QUEUE` (3) lần chờ, thêm nữa thì
  bị từ chối (`careful_busy`, giao diện: "Đang có nhiều người chuyển công thức. Hãy thử lại sau ít phút."),
  nên lần chuyển công thức không lấp hàng đợi chung. Lần chuyển không dùng mô hình (file quá số trang scan và
  không có trang có chữ) trả lượt ngay khi bắt đầu; trang scan không đọc bằng mô hình thì lượt được trả sau
  trang có chữ cuối cùng, hay khi đã đọc đủ `P2W_MFR_MAX_FORMULAS` ảnh công thức hoặc mô hình hỏng. Ảnh có chữ trên trang có chữ đọc theo cách mặc định. Mô
  hình không tải được hay hỏng giữa chừng thì trang đó và các trang sau đọc theo cách mặc định, kết quả ghi
  `formulas: "scripts"`. Giao diện chỉ hiện ô khi sẽ đọc chữ từ trang scan và máy chủ dùng được mô hình (có
  mô hình, không tắt, không hỏng; không hiện khi `P2W_FORMULAS=model`); file quá số trang scan thì ô tắt kèm
  lý do; ô ghi thời gian ước ("chậm hơn: khoảng X phút", `P2W_SECONDS_PER_SCAN_PAGE` mỗi trang scan). Lần
  chuyển nhanh còn công thức dạng ảnh mà mô hình đọc được (trang có chữ; trang scan khi ô dùng được) thì
  ghi chú một dòng "N công thức đang là ảnh – [Chuyển thành công thức sửa được] (khoảng X phút)": X là thời
  gian lần vừa rồi cộng `P2W_SECONDS_PER_FORMULA` mỗi công thức (tối đa `P2W_MFR_MAX_FORMULAS`),
  `P2W_SECONDS_PER_PICTURE_PAGE` mỗi trang có công thức dạng ảnh (Tesseract đọc lại trang) và mỗi trang scan
  như trên (lần chuyển lại đọc cả trang scan bằng mô hình khi ô dùng được); bấm là chuyển lại với mô hình,
  file Word vừa chuyển vẫn tải được trong lúc chờ, cả khi lần này không xong; "Chuyển file khác" lúc đó huỷ
  lần chuyển lại (kết quả của nó không hiện cho file mới). Xong thì ghi chú mỗi ý một dòng: số công thức đã
  chuyển ("nên soát lại"; trang scan: vùng viết từ kết quả mô hình, `formulas_read`), đọc chưa chắc nên giữ
  ảnh, quá giới hạn mỗi file, máy chủ chưa chuyển được. Hàng đợi đầy khi đã chọn ô: lời báo nhắc bỏ chọn ô
  để chuyển ngay. Qua API, file dài hơn, máy không có mô hình hay mô hình hỏng (cả khi hỏng sau lúc trang
  hiện nút) thì đọc kiểu mặc định (kết quả ghi `formulas_skipped` / `formulas` / `native_formulas_failed`).

`P2W_FORMULAS`: `scripts` (mặc định), `model` (mọi lần OCR đều đọc công thức trang scan bằng mô hình),
`off` (không đọc công thức; "Chuyển thành công thức sửa được" vẫn dùng mô hình); `P2W_MFR=0` tắt mô hình, `P2W_MATH_DETECT=off` tắt cả việc tìm vùng.

Biến môi trường tuỳ chọn: `P2W_DATA_DIR`, `P2W_FILE_TTL` (giây, mặc định 3600),
`P2W_MAX_UPLOAD_MB` (50), `P2W_MAX_PAGES` (50), `P2W_WORKERS` (2), `P2W_MAX_STORAGE_MB` (4096),
`P2W_MAX_PENDING` (20), `P2W_RATE_UPLOADS` (40), `P2W_RATE_TASKS` (120), `TESSERACT_CMD`,
`P2W_VIETOCR` (`0` để tắt), `P2W_VIETOCR_DIR` (`models/vietocr`), `P2W_VIETOCR_INT8` (`1`: bản int8
`models/vietocr-int8` khi đủ file; `0`: bản float16), `P2W_VIETOCR_MAX_PAGES` (30),
`P2W_VIETOCR_WORDS` (`unsure` mặc định, `all`), `P2W_FORMULAS` (`scripts` mặc định, `model`, `off`),
`P2W_SCRIPTS_LANG` (`eng`),
`P2W_MFR` (`0` để tắt mô hình công thức), `P2W_MFR_DIR` (`models/mfr`), `P2W_MFR_MAX_PAGES` (5),
`P2W_MFR_MAX_FORMULAS` (200: công thức dạng ảnh của trang có chữ đọc tối đa mỗi file),
`P2W_SECONDS_PER_FORMULA` (0,7), `P2W_SECONDS_PER_PICTURE_PAGE` (3), `P2W_SECONDS_PER_SCAN_PAGE` (70: giây
giao diện tính cho "khoảng X phút"),
`P2W_MFR_SPLIT` (`0`: bộ giải mã nguyên khối),
`P2W_CAREFUL_WORKERS` (1: số lần chuyển công thức chạy cùng lúc), `P2W_CAREFUL_QUEUE` (3: số lần chuyển công
thức chờ tối đa), `P2W_ABANDON_SECONDS` (600: việc chờ không ai hỏi chừng ấy giây bị bỏ),
`P2W_VIETOCR_THREADS`, `P2W_MFR_THREADS` (số lõi chia cho `P2W_WORKERS`, tối đa 2),
`P2W_MATH_DETECT` (cách tìm vùng công thức: `rules` mặc định, `off` để tắt).

Máy chủ nhỏ (2 lõi ARM Neoverse-N1, 12 GB, `P2W_WORKERS=2`): mỗi lần chuyển dùng 1 luồng cho mỗi mô hình,
nên hai lần chuyển cùng lúc không giành lõi của nhau (đo: hai lần chuyển VL1212 cùng lúc mỗi lần chậm như
chạy một mình, cả process ~2,5 GB); mỗi lúc chỉ một lần chuyển công thức, nên lõi kia luôn còn cho lần
chuyển thường. File hơn `P2W_MFR_MAX_PAGES` (5) trang scan không đọc công thức trang scan bằng mô hình, file hơn `P2W_VIETOCR_MAX_PAGES` (30)
trang scan không dùng VietOCR (giao diện ghi một dòng), để một file dài không giữ máy chủ quá lâu. Thời gian
mỗi trang scan trên máy chủ, giây (ước từ CPU đo trên Mac không dùng KleidiAI, một luồng: Tesseract ×2,3, mô
hình ×2,9 – khớp trong 10 % với lần đo thật đề VL1212 trên máy chủ: 438 giây có cả hai mô hình, 17 giây
không mô hình; 2026-10-28):

| cách chuyển | đề Vật lí scan 3 trang (VL1212): trung bình (từng trang) | bản scan giả 6 trang của đề Vật lí khác | 30 trang scan thử: trung bình / trung vị / trang lâu nhất |
|---|---|---|---|
| chỉ Tesseract (`P2W_VIETOCR=0`, `P2W_FORMULAS=off`) | 5,4 (6 / 7 / 3) | 4,7 | 4,4 / 5,4 / 9 |
| mặc định: VietOCR int8 `unsure` + công thức không mô hình (tỉ lệ int8 / float16 đo trên máy chủ: 0,46, trên Mac 0,44) | 12,4 (16 / 16 / 6) | 8,3 | 8,2 / 9,7 / 21 |
| như trên, VietOCR float16 (`P2W_VIETOCR_INT8=0`) | 19,7 (25 / 26 / 8) | 12,4 | 12,2 / 14,7 / 33 |
| chuyển công thức: thêm mô hình công thức, bộ giải mã tách đôi | 73,9 (105 / 105 / 12) | 30,2 | 34,6 / 21,7 / 114 |
| bản trước (mọi lần chuyển dùng cả hai mô hình, VietOCR đọc mọi từ) | 155,7 (203 / 239 / 25) | 65,9 | 66,3 / 44,1 / 233 |

Trang scan dày nhất (~2 phút) còn xa mức 10 phút giao diện bỏ cuộc; một file 5 trang dày giữ lượt chuyển
công thức ~9 phút, lần sau chờ trong lúc đó. Công thức dạng ảnh của trang có chữ: đề Vật lí MathType 6 trang
(158 ảnh) thêm 36 giây trên Mac một luồng (2,0 → 37,7 giây): ~0,85 giây mỗi trang Tesseract đọc các dải quanh
ảnh cho phép so chữ số, ~0,2 giây mỗi ảnh mô hình; trên máy chủ theo các hệ số trên ~2 giây mỗi trang
(`P2W_SECONDS_PER_PICTURE_PAGE` 3), ~0,58 giây mỗi ảnh (`P2W_SECONDS_PER_FORMULA` 0,7, để dư); 200 công thức
trên 10 trang ~3 phút. 50 trang A4 dày chữ, mỗi trang một ảnh: 29 giây trên Mac (đọc cả trang: 292 giây).

Bảo vệ khi chạy công khai:
- Giới hạn theo IP người dùng (`CF-Connecting-IP` do Cloudflare Tunnel đặt; máy chủ chỉ nghe
  127.0.0.1): 40 lần tải lên, 120 lần xử lý mỗi 10 phút. Hàng đợi tối đa 20 việc, trong đó chuyển công
  thức tối đa 1 chạy + 3 chờ; mỗi file một việc chạy cùng lúc, tối đa 6 lần. Trang đóng hay tải lại thì việc
  của nó bị huỷ (đang chờ: bỏ ngay; đang chạy: dừng ở trang sau); việc đang chờ mà 10 phút không ai hỏi
  bị bỏ khi tới lượt. Tổng file lưu tối đa 4 GB; file của việc chưa xong không bị xoá khi quá 1 giờ (xoá ở
  lần dọn sau khi việc xong). Trang dài hơn 3000pt bị từ chối; trang hơn 20.000 nét vẽ giữ thành ảnh.
- Header: CSP chỉ cho script/CSS của chính trang (`static/app.js`, `app.css`), cấm nhúng iframe,
  nosniff, HSTS; `/v1/*` có `Cache-Control: no-store` (Cloudflare mặc định lưu đệm .pdf/.docx).
  Không có `/docs`, `/openapi.json`. File lưu với quyền 600.
- `deploy/pdf2word.service`: hộp cát systemd (chỉ ghi được `data/`, `PrivateTmp`,
  `NoNewPrivileges`…) và `MemoryMax=80%`; `scripts/deploy.sh` cài lại file này khi nó đổi và tự trả
  lại bản cũ nếu dịch vụ mới không lên.

Cập nhật bản chạy trên máy chủ (dịch vụ systemd `pdf2word`): `./scripts/deploy.sh`. Script `git pull` rồi
so với bản đã triển khai lần trước (ghi trong `.git/pdf2word-deployed` sau mỗi lần xong; chưa có thì coi
như mọi thứ đổi) và với bản có trên máy trước khi pull, nên `git pull` tay trước đó hay quay lại bản cũ
bằng tay (`git reset --hard <commit>` rồi restart) không làm sót gì: cài lại thư viện khi `requirements.txt`
đổi, tải mô hình khi `scripts/models.json` đổi hoặc `fetch_models.py --check` thấy thiếu/hỏng (tải hỏng
thì lần chạy sau tải tiếp), cài lại dịch vụ khi bản đã cài khác `deploy/pdf2word.service`, restart khi
`app/`, mô hình hay dịch vụ đổi (giao diện `static/` đọc thẳng từ ổ đĩa). Dịch vụ không lên thì bản đó
không được ghi, lần chạy sau làm lại. Cuối cùng nhắc một dòng nếu một drop-in systemd
(`/etc/systemd/system/pdf2word.service.d/*.conf`) đặt `P2W_VIETOCR=0`, `P2W_MFR=0`, `P2W_MFR_SPLIT=0`
hay `P2W_FORMULAS=model|off`: lệnh xoá file khi file không đặt gì khác, không thì lệnh sửa file
(`sudoedit`) để bỏ các biến đó. `--restart` để restart dù không có gì đổi.

## Luồng API

| Bước | Request |
|---|---|
| Xin URL upload | `POST /v1/upload-url` → URL ký HMAC, hết hạn 15 phút (kiểu presigned URL của object store) |
| Upload | `PUT /v1/files/<token>?expires&sig` |
| Phân tích | `POST /v1/tasks` `{"tool":"analyse-document"}` |
| Convert | `POST /v1/tasks` `{"tool":"pdf-to-word","options":{"ocr_type":"ocr","languages":["vie"]}}` (thêm `"careful":true` để "Chuyển thành công thức sửa được", cả với `ocr_type` `""`) |
| Poll | `GET /v1/tasks/<task_id>` → `status`, `progress`, `result.output_files`, `result.error_fe` |
| Huỷ | `POST /v1/tasks/<task_id>/cancel` (trang gửi khi đóng/tải lại): việc đang chờ bỏ ngay, đang chạy dừng ở trang sau |
| Tải về | `GET /v1/files/<token>?name=...` |

Kết quả `analyse-document` có thêm `careful`: `{"available": true, "scan_offered": true, "max_scan_pages":
5, "max_formulas": 200, "seconds_per_formula": 0.7, "seconds_per_picture_page": 3, "seconds_per_scan_page":
70}` (máy chủ dùng được mô hình
không, ô cho trang scan có nên hiện không, tối đa mấy trang scan, mấy công thức dạng ảnh mỗi file, số giây
giao diện tính cho "khoảng X phút"). Kết quả `pdf-to-word` có `careful`, `formulas` (cách đọc công thức trang
scan: `model` / `scripts` / `off`), `formulas_skipped` (muốn dùng mô hình mà file quá nhiều trang scan),
`vietocr_skipped`, `formula_pictures` / `formulas_read` (vùng công thức trang scan giữ thành ảnh / viết từ kết
quả đọc), `equations` / `native_formula_pictures` (công thức trang có chữ dựng lại thành Equation / giữ thành
ảnh), `native_picture_pages` (số trang có công thức dạng ảnh), và khi
`careful`: `native_formulas_read` (ảnh công thức đã thành Equation hay chữ có chỉ số), `native_formulas_unsure`
(đọc chưa chắc, giữ ảnh), `native_formulas_over` (quá `max_formulas`, giữ ảnh), `native_formulas_failed`
(không dùng được mô hình – không có, tắt hay hỏng –, giữ ảnh).

Giao diện hỏi trạng thái mỗi 0,7 giây và chỉ bỏ cuộc ("Xử lý quá lâu…") khi 10 phút liền không có
tin mới: trang nào xong thêm, hay `progress.tick` tăng (máy chủ báo sau mỗi nhóm 20 ảnh công thức của một
trang, nên trang nhiều công thức không bị coi là treo); thời gian chờ trong hàng đợi (cả chờ lượt chuyển
công thức) không tính, nên file scan dài hay chuyển công thức vẫn nhận được kết quả.

`task_id` do client sinh (32 ký tự hex). Việc đang chờ mà `P2W_ABANDON_SECONDS` (10 phút) không ai
hỏi trạng thái thì bị bỏ khi tới lượt. `POST /v1/tasks` trả 503 `busy` (hàng đợi đầy), 503 `careful_busy`
(đã có 3 lần chuyển công thức chờ), 429 `too_many_tasks`/`rate_limited`. Mã lỗi `error_fe` cho giao diện:
`password`, `corrupt_pdf`, `no_pages_in_pdf`, `too_many_pages`, `no_text_found`, `input_tokens` (file
không còn), `cancelled`, `abandoned`, `internal_server_error`. Mở trang với `?debug` để thấy mục "Nhật ký
API" (toàn bộ request/response).

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
  - Equation của Word (font Cambria Math, cả STIX/Latin Modern Math: chữ có Unicode thật) được
    **dựng lại thành Equation** sửa được trong Word (`app/nativemath.py`) từ vị trí glyph: chữ toán
    nghiêng 𝑚 → m, 𝒃 → chữ đậm nghiêng, chữ Hy Lạp (ε/ϵ, φ/ϕ phân biệt); chỉ số trên/dưới theo cỡ chữ
    và độ lệch đường chân chữ (cả chỉ số của số thập phân "4,1²", v′₁); gạch phân số (lồng nhau; tử/mẫu
    cân giữa gạch); mũi tên vectơ và dấu mũ trên một chữ (vị trí vẽ thật lấy từ `get_texttrace`, vì lớp
    chữ dời dấu ghép về cuối glyph trước); "⁰C" sau ngoặc hay cách số trước là chỉ số không cơ số; đơn vị
    chữ đứng ("kg", "J/kgK"), chữ thường trong Equation ("Hiệu suất", font Cambria Math như bản gốc);
    vạch "|x|" (chữ, hay nét vẽ thành cặp); ℝ ℕ ℤ ℚ ℂ ∅ ∖ ∀ ∃ ⋮ ⊆, n!; khoảng trắng gõ tay (≥0,16em; sau
    dấu chấm ≥0,2em vì Word tự giãn 0,15–0,17em). Gạch đầu dòng "- ", nhãn khác màu ("b)") và dấu câu
    cuối font chữ thường thành chữ ngoài Equation; màu chữ giữ nguyên. Equation có phân số cỡ bằng dòng
    là Equation hiển thị của Word: đứng riêng đoạn thì ghi `m:oMathPara` căn theo đoạn, cạnh chữ thì giữ
    ảnh; công thức chỉ có phân số cạnh chữ là Equation trong dòng ở cỡ chữ dòng. Equation trong dòng
    đứng một mình trên một dòng (cả sau ngắt dòng trong ô bảng) có thêm dấu cách sau nó, không thì Word
    coi là Equation hiển thị. **Không chắc thì giữ ảnh** (một Equation sai tệ hơn một ảnh): nét vẽ khác
    gạch phân số và vạch đứng thành cặp (căn, ngoặc nhọn, gạch chân, nền tô), glyph cỡ dòng lệch đường
    chân chữ (hai dòng), cận dưới "lim/min", chữ trên mũi tên, chỉ số của chỉ số, chỉ số đứng dưới chữ,
    chỉ số không cơ số trước toán tử hay dấu đóng ngoặc, phân số trong số mũ, dấu mũ giãn trên nhiều chữ
    (góc xOy), ngoặc không cân hay ngoặc giãn cao hơn mọi thứ công thức chứa (ma trận, hệ – đo mực khi vẽ
    trang), nhiều màu, nhãn có dấu, ký tự lạ, font chữ thường lẫn giữa, glyph dưới 4 pt, công thức quá
    lớn (300 glyph/nét, 32 dấu ghép), LaTeX `omml.py` không nhận; và glyph mà font toán không có bảng mã
    chữ (ToUnicode rỗng/thiếu mã: PDF chỉ còn mã "!\"#$"; font CID thì đọc lại trang không thay mã lạ,
    gặp U+FFFD là giữ ảnh), font không có trong danh sách font của trang (chỉ đọc từ điển /Font của chính
    trang, có giới hạn; font toán riêng của Form XObject thì cả trang giữ ảnh). Bảng mã ToUnicode chỉ đọc
    trong giới hạn (64 KB, giải nén Flate có chặn, 1 MB mỗi file), đối tượng font đọc một lần mỗi file.
    Word for Mac xuất chỉ số bằng glyph không có bảng mã ("x_A" đọc thành "𝑥!"), nên công thức có chỉ số
    của file Word for Mac vẫn là ảnh; file Word for Windows có bảng mã đủ.
  - Công thức **MathType (cả Equation 3.0) mà Word for Windows xuất ra PDF bằng chữ** (Times New Roman,
    Symbol, gạch phân số là nét vẽ) cũng được **dựng lại thành Equation**. Mỗi công thức là một form
    XObject riêng của Word ("/Meta18 Do", ảnh EMF của nó): luồng lệnh của trang cho khung từng form, và một
    bản sao trang chỉ vẽ các form (trước mỗi form một dấu nhỏ mang số của nó) cho biết chính xác chữ và nét
    nào của form nào, nên công thức được lấy trọn mà không kéo chữ dòng trên/dưới vào. Glyph Symbol đọc
    theo bảng mã Symbol (mã riêng U+F0xx, mã Latin, hay Unicode thật khi font có ToUnicode), chữ Hy Lạp
    thường và chữ nghiêng thành chữ toán nghiêng (biến), chữ đứng giữ đứng (đơn vị, sin/cos); chữ tiếng
    Việt TCVN3/VNI trong công thức đọc qua bảng mã cũ. Ngoặc MathType đặt cỡ lớn hơn quanh nội dung được
    nhận; phân số đặt cạnh chữ là Equation trong dòng (Word đặt tử, mẫu nhỏ hơn); dòng chỉ có công thức
    sau dấu ":" hay dưới dòng công thức khác là đoạn riêng. Chỉ là công thức khi form có dấu hiệu toán
    (dấu toán Symbol, gạch phân số có tử ngay trên và mẫu ngay dưới, chỉ số; không thì chỉ chữ nghiêng
    từng cụm ngắn có phép toán); chữ thường vẽ bằng form (câu có từ tiếng Việt, tiêu đề, bảng giá trị có
    cột, dấu đầu dòng, hình có nét cong/hình tô, bảng Excel dán dạng ảnh) giữ như trước. Không chắc thì cả
    công thức là **một ảnh** vẽ từ chính form (không vỡ ra từng mảnh như trước); công thức chỉ một dòng
    không dựng được thì là chữ. Đọc luồng lệnh và sao trang đều có giới hạn (4 MB nội dung mỗi trang,
    16 MB mỗi file; 300 form mỗi trang; mỗi form 64 KB; object sao theo 1000 mỗi trang, 20 000 mỗi file).
  - Công thức không có chữ dùng được – MathType/Equation mà Word for Mac xuất ra PDF đánh số glyph
    "!", "\"", "#"…, công thức MathType không phải form của Word và công thức trên không đọc chắc – được cắt
    thành **ảnh nội dòng** 300 DPI đúng chỗ trong
    câu, hạ theo baseline (alt "Công thức (ảnh)"). Vùng công thức là nhóm glyph công thức cùng
    glyph Symbol/Cambria Math kề bên (cùng dòng: cách ≤0,35em; chồng nhau như cận tích phân: chỉ
    glyph nhỏ hơn nằm giữa glyph lớn) và nét vẽ dính vào (gạch phân số, căn, mũi tên, vạch trị
    tuyệt đối "|x|"). Gạch phân số giữ tử và mẫu đặt gọn trên/dưới nó thành một khối, kể cả chữ
    thường trong phương trình Word ("Năng lượng nhận được", "1/110" Cambria Math); gạch chân thì
    cắt qua hộp chữ nên không bị coi là gạch phân số. Hai dòng công thức xếp sát nhau vẫn là hai
    ảnh (hay hai Equation), kể cả khi chỉ số của dòng trên chạm glyph dòng dưới. Khối chồng lên nhau gộp làm một, chữ nằm gần trọn trong khối thuộc về khối (không viết lại
    lần nữa cạnh ảnh). Ảnh chỉ có nét của chính công thức trên nền trong suốt: watermark, nền ô,
    chữ xoay chéo phía sau và mép dòng bên cạnh đều không lọt vào. Dấu câu ngay sau công thức
    (")?", ".") nằm ngoài khối chữ của công thức vẫn là chữ; số và nhãn "(1)", "20%", "(1) 6273." là chữ. Dòng
    bị PDF cắt quanh công thức được nối lại; hai dòng của PDF chen nhau trên cùng đường chân chữ (số
    mũ "⁰" cùng ") được" là một dòng riêng bắt đầu giữa "J/kg" và "C") được ghép theo thứ tự ngang,
    không thành tab; mảnh sát nhau ghép thẳng, không chèn tab. Dòng công
    thức rộng cạnh hình không bị coi là nhãn của hình.
  - **"Chuyển thành công thức sửa được"** (`careful`, `app/nativeread.py`): các ảnh công thức nội dòng trên
    được mô hình công thức đọc như vùng công thức trang scan (`mathocr.read_regions`: hai lần đọc phải
    khớp, chữ số khớp chữ Tesseract đọc cùng chỗ, kiểm chữ tiếng Việt và nhãn như trang scan), tối đa
    `P2W_MFR_MAX_FORMULAS` (200) mỗi file theo thứ tự trang. Công thức hai tầng (phân số, căn, vectơ) thành
    Equation trong dòng (ảnh giữ kèm: Equation Word không nhận thì ghi ảnh); công thức một dòng ("x_A",
    "4,5.10³", "3 km") thành chữ có chỉ số trên/dưới, font của chữ quanh nó; chữ thường mà mô hình từ chối
    (chỉ có chữ cái – trục "Ox", đơn vị "m/s" –, chữ Việt "20 phút": mô hình không có chữ Việt) thành chữ
    Tesseract đọc cùng chỗ khi cả hai lần đọc của mô hình đều tin được và ra đúng chữ đó: hai cách đọc ít khi
    sai giống nhau (Tesseract đọc chữ nghiêng MathType A thành "4", B thành "8": giữ ảnh). Dấu tiếng Việt của
    Tesseract chỉ được nhận trên từ chữ thường là âm tiết tiếng Việt ("phút", "vòng"): Tesseract cũng tự thêm
    dấu cho tên trục, tên điểm, chữ hoa ("Ox" thành "Ôx", "U" thành "Ữ"), những chỗ đó giữ ảnh. Phép so của
    trang scan chỉ xét chữ số ngoài chỉ số và ngoài phân số, nên ở đây giữ ảnh thêm khi: chỉ số dưới có d/D
    (mô hình không có đ/Đ: W_đ, R_Đ, F_đh), chỉ là chữ o (l₀ đọc thành l_o) hay có từ chữ thường ngoài vài chữ
    viết tắt (max, min, tb, ms…: "v_ngược" đọc thành v_nguoc), chỉ số có chữ mà Tesseract thấy chữ Việt ở đó;
    số mũ không phải số/chữ có dấu, độ, phẩy ("10⁻⁴ rad/s" đọc thành 10^{- rad/s}); tử hay mẫu là một chữ chỉ
    chiều cao mới phân biệt hoa/thường (s/S, v/V…); kết quả mở đầu bằng "+". Khoảng trắng trong chỉ số bị bỏ
    ("10⁻²³", không "10⁻ ²³"). Không chắc thì giữ ảnh. Ảnh đem đọc là chính ảnh công thức (chỉ nét của nó, 300
    DPI) trên nền trắng, xếp chồng trên một tấm (tối đa 20 ảnh, 8 triệu điểm ảnh, giải mã từng tấm) nên không
    lẫn chữ bên cạnh; ảnh lớn hơn mô hình đọc (2 triệu điểm ảnh) không được giải mã, giữ ảnh. Tesseract chỉ đọc
    các dải ngang của trang quanh các ảnh (dòng của chúng, thêm 30 % chiều cao ảnh trên và dưới, gộp chỗ chồng
    nhau, xếp chồng làm một ảnh 300 DPI, tối đa 60 giây), lấy các từ nằm ít nhất một nửa trong khung công
    thức: trang dày chữ chỉ tốn phần dòng có công thức. Trang rộng hơn 4500 điểm ảnh ở 300 DPI hay dải quá
    lớn, trang xoay (khung công thức theo trang chưa xoay) thì giữ ảnh. Dấu câu cuối mà dòng đã có sau ảnh
    (kể cả sau các mẩu khoảng trắng) không lặp lại. Equation đứng một mình trong đoạn hay ô lấy font chữ phổ
    biến của trang, nên vẫn là Equation trong dòng (không thành Equation giữa dòng).
  - Ảnh raster và hình vector thành ảnh nổi đúng vị trí trên trang; nhãn chữ trong hình hoặc sát
    mép (tên trục "s (km)", vạch chia "O", "2 4 5"; ở trên/dưới hình thì phải nằm trong bề ngang
    hình – "PHẦN I." trên bảng là tiêu đề) được render cùng hình; hình lớn đè lên chữ
    (watermark) đặt sau chữ; dòng chỉ chạm mép trên/dưới hình (phương án "A. … D." ngay dưới đồ
    thị) không làm hình thành nền, trừ khi trong hình có cả câu dài; đường kẻ mỏng (kẻ trên chân
    trang) chỉ nhận chữ sát nó (số trang), không nhận dòng cuối bài ngay trên. Hình (có chữ bao
    quanh) neo vào đoạn đầu tiên nằm ngang mép trên của nó (chung ít nhất 8pt, hoặc nửa đoạn, và
    bắt đầu không thấp hơn mép trên quá 8pt – dòng phương án dưới đồ thị không làm neo), không có
    thì đoạn ngay trên nó, và đặt theo **đoạn đó** (không theo trang): hình đi theo câu hỏi cả khi chữ
    dời sang trang khác. Word đo vị trí đó từ đầu khoảng trống phía trên đoạn (không từ dòng chữ đầu), và
    đoạn đầu trang bắt đầu ở lề trên: băng tiêu đề, khung "MỤC TIÊU" trên câu đầu trang nằm đúng chỗ, không
    bị đẩy lên mép trang chồng nhau. Khoảng trống chứa hình giữ đủ chiều cao (khoảng khác tối đa 72pt). Ảnh nền
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
     Chỉ từ Tesseract tin <92 hay có â ă ê ô không dấu/dấu huyền được đọc lại (`vietocr.unsure`;
     `P2W_VIETOCR_WORDS=all`: mọi từ); từ khác giữ chữ Tesseract.
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
     nó và nằm trong cột của dòng kia (dòng ngay trên nó thuộc khối dòng kia, hay cả hai đều là khối
     một dòng và đủ A–D), và hai dòng ghép lại là một hàng phương án: dòng lẻ mở đầu bằng một nhãn
     (kể cả đọc sai "B,", "c.", "Đ."), không có nhãn khác, nhãn hai dòng đọc liền nhau theo chiều
     ngang ("A" + "BCD"). Dòng hai cột, kể cả hàng phương án của hai câu cạnh nhau ("A. 10 m." bên
     "C. 25 N."), giữ riêng; đơn vị sau số ("1 A.", "2.10⁻⁶ C.") không phải nhãn; dải đầu/chân
     trang giữ nguyên.
     Tìm vùng công thức (`app/mathregions.py`, cách `rules`, theo thứ tự ưu tiên):
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
     Mặc định (`app/scriptocr.py`) chỉ vùng có nét nâng lên/hạ xuống được đọc: các đoạn trên dòng, đoạn
     nâng, đoạn hạ đặt cạnh nhau trên một dòng (chỉ số phóng lên cỡ chữ, đặt trên đường chân chữ),
     Tesseract đọc dòng đó hai lần (như cũ và phóng 1,5 lần, chỉ chữ số, chữ Latinh và vài dấu), hai lần
     phải khớp; mỗi ký tự đối chiếu với nét của nó (số ký tự bằng số nét, dấu phẩy/chấm theo nét, dấu trừ
     chỉ ở chỗ hình học thấy, chữ số đứng trên dòng cao bằng chữ số, 0/O theo bề rộng, o/O, v/V theo chiều
     cao, chữ sau số mũ phải là đơn vị); chữ cái không kiểm được phải có trong chữ OCR của trang; rồi qua
     cùng luật an toàn như kết quả mô hình. Vùng cao (phân số, căn) thành ảnh công thức.
     Chuyển công thức (`careful`): mô hình đọc mỗi vùng 2–3 lần với lề cắt khác nhau (lần đầu là chữ thường trùng chữ
     Tesseract, "6 kPa.", thì giữ luôn, không đọc lần hai), chỉ lấy kết quả
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
.venv/bin/python -m unittest discover -s tests  # kiểm thử đơn vị (VietOCR, công thức, tải mô hình, hàng đợi)
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
- Bộ Bai_1–6 (Word 2019, Equation Cambria Math): 40/40 công thức thành Equation, 0 sai so với bản
  chuẩn gõ tay (thêm 2 Equation đúng, 2 công thức font không bảng mã giữ ảnh ở một đề mẫu khác);
  Word render khớp bản gốc, mở trang khớp 24/24; tốc độ như cũ (~11 giây cả 6 file trên Mac).
- Đề Vật lí 10 gõ MathType, Word xuất PDF 6 trang: ~1,2 giây, ~145 ảnh công thức; khớp 98,7% từ
  với file Word gốc (công thức tính là một từ); Word render đúng 6 trang; trang đọc đề đọc đủ 28/28
  đáp án.
- "Chuyển thành công thức sửa được" trên đề đó (158 ảnh công thức): 120 đọc được – 11 Equation, 109 chữ có
  chỉ số –, so với 157 công thức chuẩn gõ tay: 120 đúng, 0 sai; 38 giữ ảnh (phần lớn nhãn A, B chữ nghiêng);
  so với file Word gốc mọi Equation đều khớp, chữ 99,3 %. Không chọn: 22 PDF có chữ, Bai_1–6 và VL1108 ra
  .docx giống hệt từng byte bản trước; chọn: chỉ đề Vật lí và một đề mẫu (1/2 ảnh đọc được) đổi, và chỉ ở
  đoạn có ảnh công thức; file không còn ảnh công thức giống hệt.

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

Cách chuyển cho máy chủ nhỏ (2026-10-28; trong ngoặc: bản trước, mọi lần chuyển dùng cả hai mô hình):
- Mặc định (VietOCR `unsure`, công thức không mô hình): đề Vật lí scan 3 trang chữ 99,6 % (99,6 %), số
  95,4 % (93,4 %), Equation 0/24 (13/24: phân số, căn giữ thành ảnh), 27/51 công thức và số mũ/chỉ số của
  bộ thử (47/51); 29 kết quả đặt vào đề, đều đúng; hai bản scan giả của đề Vật lí kia 2 và 1/157 (19/157),
  không kết quả nào sai; 104 kết quả đặt trên 30 trang scan giả, so với chữ của PDF gốc: không cái nào sai. 30 trang scan giả 97,53 % / 96,67 % (97,73 % / 97,09 %; chỉ Tesseract 92,22 % /
  91,48 %): mất 35 từ đúng trên 9 trang – 32 là số, đơn vị trong vùng mô hình đọc được mà cách không mô
  hình bỏ qua (số mũ trên trang scan kém, "cm³", số thập phân "1,33" của phương án không có chỉ số), 3 do
  VietOCR chỉ đọc lại từ chưa chắc ("đẳng", "HỒ" Tesseract tin mà sai dấu; một số "0" đổi theo các từ đọc
  cùng lượt) –, thêm 10. VietOCR int8: 97,58 % / 96,69 %, so với bản float mất 2 từ ("thẳng", "A"), thêm 5;
  các số khác như bản float. PDF có text không đổi (22 file).
- Chuyển công thức (trước gọi "Đọc kỹ"): .docx của đề Vật lí và hai bản scan giả, 30 trang chữ đọc được giống hệt từng byte cách đọc cũ
  của mô hình (bộ giải mã nguyên khối, vùng nào cũng đọc hai lần) cùng VietOCR `unsure`: chữ 99,7 %,
  số 93,4 %, Equation 13/24 (15/24), 47/51, 0 kết quả sai; 30 trang 97,75 % / 97,14 %, mất 3 từ đúng (3 từ
  VietOCR ở trên), thêm 6.
- Mô hình từ bản phát hành `models-2026-10-28`: hai nửa bộ giải mã tạo lại từ bản ghim giống hệt từng byte
  bản đã đo; với `models/` do `fetch_models.py` dựng, đề Vật lí và bản scan giả 6 trang ra .docx giống hệt
  bản đã đo (bản int8, mặc định: thời gian như dòng đầu của bảng ở trên).

## Giới hạn

- Bảng phải có kẻ khung đầy đủ; chưa hỗ trợ bảng không viền; màu viền chưa giữ (luôn đen); bảng
  scan (OCR) chưa có ô gộp.
- Mọi thông tin đầu/chân trang trừ số trang đều bị bỏ (cố ý): tên trường, tên đề in lặp mỗi
  trang cũng mất; dòng chỉ có ở trang đầu thì vẫn là nội dung. Dòng ở dải đầu/chân trang được so theo
  nội dung thật của Equation và ảnh công thức (không theo ký tự giữ chỗ chung); trên trang PDF có chữ, số
  của nó phải như nhau ở mọi trang hay đếm trang; dòng mở bằng "Câu 12.", "A." không bao giờ bị coi là
  đầu/chân trang (hàng đáp án ở cuối trang không mất).
- Bảng scan: màu nền ô lệch vài đơn vị so với bản gốc (ảnh chụp điện thoại lệch nhiều hơn); ô tô
  màu chỉ đậm hơn giấy dưới 10/255 (hay dưới độ chênh sáng của ảnh chụp) thì thành ô trắng.
- Công thức thành ảnh (MathType, Equation có căn/ma trận/ngoặc lớn, font toán thiếu bảng mã): xem
  và in đúng nhưng không sửa được trong Word, công cụ đọc đề cũng không đọc được nội dung – trừ khi chọn
  "Chuyển thành công thức sửa được" (đề Vật lí MathType: 120/158 ảnh; nhãn chữ nghiêng A, B mà Tesseract
  đọc thành "4", "8" vẫn là ảnh). Công thức
  có chữ thật (PDF từ LaTeX, chỉ dùng font Symbol) vẫn là chữ; công thức Cambria Math không có chữ
  toán nghiêng hay gạch phân số ("4,5 × 10³ J", "80,0 ∘C") vẫn là chữ thường font Cambria Math.
- Đọc công thức mặc định (không mô hình): chỉ số có chữ Đ/đ (R_Đ, F_đh), chỉ số chữ y/p hay chữ x, v, s…
  đứng một mình mà chiều cao không rõ hoa/thường (nét cao 0,70–0,84 x-height của dòng), chữ trong chỉ số
  có thể là chữ số (O, t, B…) mà Tesseract đọc dòng không thấy thì giữ chữ OCR; phân số, căn, gạch trên
  luôn thành ảnh; vùng có chữ Tesseract
  đọc thêm ký tự ("9v_O"), "/" không đọc được ("g/m³"), Δ dính chữ sau nó ("Δx = x_A + x_B") hay nét
  số mũ vỡ (scan 200 dpi) thì giữ chữ OCR; dấu phẩy thập phân mất trong phương án không có số mũ
  ("1,33") không lấy lại được.
- Đọc công thức (trang scan): chữ cái đứng riêng (p, ρ, k, T) và số thường trong câu dẫn vẫn
  là chữ OCR; công thức trong bảng cũng vậy. Không đọc phương án có chữ tiếng Việt Tesseract tin
  ≥80, vùng chạm dòng trên/dưới hay phương án bên cạnh, phương án mà một từ OCR chạy sang phương
  án sau ("C. 273 K. D. 373"); kết quả mô hình không chắc thì giữ chữ OCR (thường sai:
  "642.107'J."), phân số/căn/gạch trên không đọc được thành ảnh. Tesseract để nhãn đứng riêng
  một dòng mà đường chân chữ lệch hàng của nó quá 0,15em thì phương án đó vẫn thành đoạn riêng
  ngay sau hàng. Số mũ chỉ
  nhận theo hình khi tách khỏi nét chữ trước nó (scan nhoè dính liền thì không thấy).
- Trang scan: luôn dùng Times New Roman, chưa nhận in nghiêng/gạch chân; số mũ/chỉ số dưới chỉ có
  ở chỗ đọc được (mặc định: số, đơn vị, chữ có số mũ/chỉ số; chuyển công thức thêm phân số, căn);
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
- Chuyển công thức chạy lâu (máy chủ 2 lõi ARM tới ~105 giây/trang scan nhiều công thức, ~0,65 giây mỗi ảnh
  công thức của trang có chữ): trang scan chỉ cho file tối đa
  `P2W_MFR_MAX_PAGES` (5) trang scan, mỗi lúc một lần – lần sau có thể chờ tới ~9 phút sau một file 5 trang
  dày; quá `P2W_VIETOCR_MAX_PAGES` (30) trang scan thì bỏ cả VietOCR; trang bị đóng mà không kịp báo (máy
  tắt, mất mạng) thì việc đang chạy vẫn chạy hết, việc đang chờ chỉ bị bỏ sau 10 phút; chưa giới hạn thời
  gian mỗi việc. Lượt chuyển công thức chỉ một mỗi lúc cho mọi người, chưa chia theo IP. Ảnh công thức của
  trang có chữ: tối đa `P2W_MFR_MAX_FORMULAS` (200) mỗi file, phần sau giữ ảnh.
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
bố, cùng hai nửa của `decoder_model.onnx` tách bằng `scripts/split_mfr_decoder.py`. Mô hình trong
`models/vietocr/` là `vgg_transformer` của [VietOCR](https://github.com/pbcquoc/vietocr) (Apache-2.0),
chuyển sang ONNX bằng `scripts/export_vietocr.py` (giữ nguyên trọng số, lưu float16); `models/vietocr-int8/`
là bản đó với bộ mã hoá lượng tử hoá int8 (`scripts/quantize_vietocr.py`). Các file tự làm ra được đăng ở
bản phát hành `models-2026-10-28` kèm giấy phép của mô hình gốc. File PDF mẫu
không nằm trong repo: `scripts/make_test_pdfs.py` tạo lại vào `samples/`.
