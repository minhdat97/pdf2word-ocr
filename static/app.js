const $ = (id) => document.getElementById(id);
const LANG_LABEL = { vie: "Tiếng Việt", eng: "English", "vie+eng": "Tiếng Việt + English" };
const ERRORS = {
  password: "File PDF có mật khẩu. Hãy gỡ mật khẩu rồi thử lại.",
  corrupt_pdf: "File PDF bị hỏng hoặc không đọc được.",
  no_pages_in_pdf: "File PDF không có trang nào.",
  too_many_pages: "File quá 50 trang. Hãy tách thành nhiều file nhỏ hơn.",
  too_large: "File quá 50 MB.",
  no_text_found: "Không đọc được chữ nào trong file.",
  input_tokens: "File không còn trên máy chủ (đã quá 1 giờ). Hãy chọn lại file.",
  timeout: "Xử lý quá lâu nên đã dừng. Hãy thử lại sau.",
  internal_server_error: "Máy chủ gặp lỗi khi xử lý file. Hãy thử lại.",
  network: "Mất kết nối mạng. Kiểm tra mạng rồi thử lại.",
  page_too_large: "Trang PDF quá khổ (lớn hơn A0), không chuyển được.",
  rate_limited: "Gửi quá nhiều file trong ít phút. Hãy thử lại sau 10 phút.",
  too_many_tasks: "File này đang được xử lý hoặc đã chuyển nhiều lần. Hãy chọn lại file.",
  busy: "Máy chủ đang bận. Hãy thử lại sau ít phút.",
  server_full: "Máy chủ đang bận. Hãy thử lại sau ít phút.",
  careful_busy: "Đang có nhiều người chuyển công thức. Hãy thử lại sau ít phút.",
  careful_busy_tick: "Đang có nhiều người chuyển công thức. Hãy thử lại sau ít phút hoặc bỏ chọn “Chuyển thành công thức sửa được”.",
  abandoned: "Trang tạm dừng quá lâu nên lượt chuyển đã bị huỷ. Hãy chuyển lại.",
  cancelled: "Lượt chuyển đã bị huỷ. Hãy chuyển lại.",
};
const fmt = (n, d = 1) => n.toLocaleString("vi-VN", { maximumFractionDigits: d });
// Dưới 100 giây ghi giây (một chữ số lẻ), từ đó ghi phút: làm tròn một lần rồi mới chia (không ra "1 phút 60 giây").
function duration(s) {
  if (Math.round(s * 10) < 1000) return `${fmt(s)} giây`;
  const t = Math.round(s);
  return `${Math.floor(t / 60)} phút` + (t % 60 ? ` ${t % 60} giây` : "");
}
// Thời gian ước cho "Chuyển thành công thức sửa được": phút tròn, dưới một phút thì nói vậy.
const about = (s) => (s < 60 ? "dưới 1 phút" : `khoảng ${Math.round(s / 60)} phút`);
let state = {};
// Lần chọn file của trang: kết quả về sau khi đã "Chuyển file khác" (lần chuyển công thức còn chạy) thì bỏ.
let generation = 0;

// ?debug hiện nhật ký các request (khi cần dò lỗi).
const DEBUG = new URLSearchParams(location.search).has("debug");
$("debug").hidden = !DEBUG;

function log(method, url, body, res) {
  if (!DEBUG) return;
  const line = document.createElement("div");
  line.innerHTML = `<span class="req"></span>\n`;
  line.firstChild.textContent = `${method} ${url}`;
  line.append((body ? "→ " + JSON.stringify(body) + "\n" : "") + "← " + JSON.stringify(res) + "\n\n");
  $("log").append(line);
}

async function api(method, url, body) {
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body ? { "content-type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw Object.assign(new Error("network"), { code: "network" });
  }
  const json = await res.json().catch(() => ({ success: false, error: `HTTP ${res.status}` }));
  log(method, url, body, json);
  if (!res.ok || json.success === false) throw Object.assign(new Error(json.error || `HTTP ${res.status}`), { code: json.error });
  return json;
}

function putFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("content-type", "application/pdf");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let json = {};
      try { json = JSON.parse(xhr.responseText); } catch {}
      log("PUT", url.split("?")[0] + "?expires=…&sig=…", null, json);
      xhr.status < 300 ? resolve(json) : reject(Object.assign(new Error(json.error || `HTTP ${xhr.status}`), { code: json.error }));
    };
    xhr.onerror = () => reject(Object.assign(new Error("network"), { code: "network" }));
    xhr.send(file);
  });
}

const newTaskId = () => crypto.randomUUID().replaceAll("-", "");

// Việc đang chờ/chạy của trang: đóng hay tải lại trang thì báo máy chủ bỏ (lượt chuyển công thức không chạy cho ai cả).
let currentTask = null;
window.addEventListener("pagehide", (e) => {
  if (currentTask && !e.persisted) navigator.sendBeacon(`/v1/tasks/${currentTask}/cancel`);
});

// Bỏ cuộc khi 10 phút không có tin mới: thời gian chờ trong hàng đợi (cả chờ lượt chuyển công thức) không tính, file
// scan dài hay chuyển công thức (trang dày 1–2 phút trên máy chủ) không bị dừng giữa chừng khi các trang vẫn đang xong dần.
const PATIENCE_MS = 10 * 60 * 1000;
// Mạng chập chờn (Wi-Fi trường, một lần hỏi bị ngắt hay Cloudflare trả trang lỗi): hỏi lại, chỉ báo lỗi mạng khi
// đã hỏng liền 30 giây – chuyển công thức vài phút là vài trăm lần hỏi, một lần hỏng không làm mất lượt chuyển.
const NETWORK_RETRY_MS = 30 * 1000;

// label: chữ của dòng trạng thái theo trang ("Đang chuyển"); hint: thêm sau số trang; waiting: lúc còn chờ lượt.
// Lỗi "stale": trang đã sang file khác ("Chuyển file khác") – việc của file cũ thôi, người gọi bỏ qua lặng lẽ.
const stale = () => Object.assign(new Error("stale"), { code: "stale" });

async function runTask(tool, options, label, hint = "", waiting = "Đang chờ đến lượt…") {
  const task_id = newTaskId();
  const gen = generation;
  await api("POST", "/v1/tasks", { tool, task_id, input_tokens: [state.token], options });
  if (gen !== generation) {  // "Chuyển file khác" lúc việc đang được gửi: reset() chưa biết việc này để huỷ
    navigator.sendBeacon(`/v1/tasks/${task_id}/cancel`);
    throw stale();
  }
  currentTask = task_id;
  try {
    return await pollTask(task_id, label, hint, waiting, gen);
  } catch (err) {
    // trang thôi chờ: máy chủ cũng thôi
    if (["timeout", "network", "stale"].includes(err.code)) navigator.sendBeacon(`/v1/tasks/${task_id}/cancel`);
    throw err;
  } finally {
    if (currentTask === task_id) currentTask = null;
  }
}

async function pollTask(task_id, label, hint, waiting, gen) {
  let deadline = Date.now() + PATIENCE_MS;
  let seen = "";
  let logged = false;
  let failingSince = 0;
  while (true) {
    if (Date.now() > deadline) throw Object.assign(new Error("timeout"), { code: "timeout" });
    await new Promise((r) => setTimeout(r, failingSince ? 2000 : 700));
    if (gen !== generation) throw stale();  // trạng thái của file cũ không ghi lên dòng trạng thái của file mới
    let res;
    try {
      res = await fetch(`/v1/tasks/${task_id}`).then((r) => r.json());
    } catch {
      failingSince ||= Date.now();
      if (Date.now() - failingSince > NETWORK_RETRY_MS) throw Object.assign(new Error("network"), { code: "network" });
      continue;
    }
    failingSince = 0;
    const data = res.data;
    if (!data) throw new Error(res.error || "Không lấy được trạng thái");
    const { current, total } = data.progress;
    // tick: máy chủ báo tiến độ trong một trang (từng nhóm công thức) – trang nhiều công thức không bị coi là treo
    const step = `${data.status} ${current} ${data.progress.tick || 0}`;
    if (data.status === "queued" || step !== seen) deadline = Date.now() + PATIENCE_MS;
    seen = step;
    if (label && data.status === "queued") setProgress(null, waiting);
    else if (total && label) setProgress(current / total, `${label} trang ${Math.min(current + 1, total)}/${total}${hint}…`);
    if (data.status === "done" || data.status === "failed") {
      log("GET", `/v1/tasks/${task_id}`, null, res);
      if (data.status === "failed") throw Object.assign(new Error(data.result.error_fe), { code: data.result.error_fe });
      return data.result;
    } else if (!logged) {
      log("GET", `/v1/tasks/${task_id}`, null, res);
      logged = true;
    }
  }
}

function setProgress(frac, text) {
  $("barfill").parentElement.classList.toggle("wait", frac === null);
  $("barfill").style.width = frac === null ? "" : `${Math.round(frac * 100)}%`;
  $("status").textContent = text;
}

// Lỗi lúc tải lên (máy chủ trả lời bằng tiếng Anh)
const UPLOAD_ERRORS = [
  [/larger than/, "File quá 50 MB."],
  [/not a PDF|only \.pdf/, "File này không phải PDF."],
  [/expired upload URL|already uploaded/, "Phiên tải lên đã hết hạn. Hãy chọn lại file."],
];

function showError(err) {
  const upload = UPLOAD_ERRORS.find(([re]) => re.test(err.message || ""));
  $("error").textContent = ERRORS[err.code] || ERRORS[err.message] || (upload && upload[1]) || err.message;
  $("error").hidden = false;
  $("busy").hidden = true;
  // lỗi trước khi có tuỳ chọn (tải lên, đọc file): chỉ còn cách chọn file khác
  $("failed").hidden = !$("options").hidden || !$("done").hidden;
}

function reset() {
  // việc còn chạy của file trang rời đi (lần chuyển công thức, file Word đầu đang hiện) thôi chạy, kết quả của nó bị bỏ
  if (currentTask) navigator.sendBeacon(`/v1/tasks/${currentTask}/cancel`);
  currentTask = null;
  generation++;
  state = {};
  for (const id of ["error", "work", "options", "done", "kind", "failed"]) $(id).hidden = true;
  $("pick").hidden = false;
  $("input").value = "";
  $("strip").checked = true;
  $("full").checked = false;
  $("careful").checked = false;
  $("carefulrow").hidden = true;
  for (const d of document.querySelectorAll("#work details")) d.open = false;
}

// Thời gian ước khi tích ô của trang scan: mỗi trang scan; trang có chữ của file còn có công thức dạng ảnh chưa biết
// trước số lượng (cũng được chuyển) nên khi đó ghi "trở lên".
function scanEstimate() {
  const info = (state.analysis && state.analysis.careful) || {};
  const pages = ocrPages();
  const more = state.analysis.page_count > pages;
  return about(Math.max(pages * (info.seconds_per_scan_page ?? 70), more ? 60 : 0)) + (more ? " trở lên" : "");
}

// Số trang sẽ đọc chữ từ ảnh với lựa chọn hiện tại (như máy chủ đếm cho "Chuyển thành công thức sửa được").
function ocrPages() {
  const a = state.analysis;
  if (!a || new FormData($("options")).get("mode") !== "ocr") return 0;
  return $("full").checked ? a.page_count : (a.pages || []).filter((p) => p.kind === "scan").length;
}

// Ô "Chuyển thành công thức sửa được" cho trang scan: chỉ khi có trang scan sẽ được đọc và máy chủ dùng được mô hình
// công thức (không khi mọi lần OCR đã dùng: scan_offered); file quá dài thì tắt kèm lý do. Trang có chữ thì sau lần
// chuyển nhanh mới biết còn công thức dạng ảnh không: dòng ghi chú có nút (convert).
function showCareful() {
  const info = (state.analysis && state.analysis.careful) || {};
  const pages = ocrPages();
  const max = info.max_scan_pages || 0;
  const offered = Boolean(info.available && info.scan_offered && pages);
  $("carefulrow").hidden = !offered;
  const tooMany = pages > max;
  $("careful").disabled = tooMany;
  if (tooMany) $("careful").checked = false;
  $("carefulrow").classList.toggle("off", tooMany);
  $("carefulmeta").textContent = tooMany
    ? `– chỉ cho file tối đa ${max} trang scan (file này ${pages} trang)`
    : `– chậm hơn: ${scanEstimate()}`;
  state.carefulOk = offered && !tooMany;
}

function showMode() {
  $("ocrrow").hidden = new FormData($("options")).get("mode") !== "ocr";
  showCareful();
}

async function handleFile(file) {
  reset();
  const gen = generation;
  if (!file.name.toLowerCase().endsWith(".pdf")) return showError(new Error("Chỉ nhận file .pdf."));
  if (file.size > 50 * 1024 * 1024) return showError(new Error(ERRORS.too_large));
  state.name = file.name;
  $("pick").hidden = true;
  $("work").hidden = false;
  $("busy").hidden = false;
  $("fname").textContent = file.name;
  $("fmeta").textContent = `${fmt(file.size / 1024 / 1024)} MB`;
  try {
    setProgress(0, "Đang tải file lên…");
    const up = await api("POST", "/v1/upload-url", { filename: file.name });
    await putFile(up.url, file, (f) => setProgress(f, `Đang tải file lên… ${Math.round(f * 100)}%`));
    state.token = up.token;

    setProgress(null, "Đang đọc file…");
    const analysis = (await runTask("analyse-document", {})).options;
    if (gen !== generation) return;
    if (analysis.password_protected) throw Object.assign(new Error("password"), { code: "password" });
    state.analysis = analysis;
    const total = analysis.page_count;
    $("fmeta").textContent += ` · ${total} trang`;

    // Đọc chữ từ ảnh khi có trang scan; đề thường có hình vẫn giữ chữ sẵn có.
    const scans = (analysis.pages || []).filter((p) => p.kind === "scan").length;
    const noText = !analysis.contains_text;
    const mode = scans || noText ? "ocr" : "basic";
    $("kind").textContent = noText || scans === total ? "Bản scan" : scans ? `${scans} trang scan` : "PDF có chữ";
    $("kind").hidden = false;
    $("autonote").textContent =
      noText || scans === total ? "Bản scan: sẽ đọc chữ từ ảnh (OCR) để sửa được trong Word."
      : scans ? `Có ${scans} trang scan: các trang này sẽ được đọc chữ từ ảnh (OCR).`
      : "PDF có chữ: giữ nguyên chữ, công thức, bảng và hình.";
    document.querySelector(`input[name=mode][value=${mode}]`).checked = true;
    showMode();
    $("busy").hidden = true;
    $("options").hidden = false;
  } catch (err) {
    if (gen === generation) showError(err);
  }
}

// again: nút "Chuyển thành công thức sửa được" sau lần chuyển nhanh (state.estimate: số giây ước tính) – file Word
// vừa chuyển vẫn tải được trong lúc chờ, và cả khi lần này không xong.
async function convert(e, again = false) {
  if (e) e.preventDefault();
  const mode = new FormData($("options")).get("mode");
  const ocrType = mode === "ocr" ? ($("full").checked ? "ocr-full" : "ocr") : "";
  const scanCareful = Boolean(ocrType && state.carefulOk && $("careful").checked);
  const careful = again || scanCareful;
  const ci = (state.analysis && state.analysis.careful) || {};
  const gen = generation;
  const options = {
    ocr_type: ocrType,
    languages: ocrType ? $("lang").value.split("+") : [],
    strip_watermarks: $("strip").checked,
    careful,
  };
  $("options").hidden = true;
  if (!again) $("done").hidden = true;
  $("error").hidden = true;
  $("busy").hidden = false;
  const hint = careful ? ` – ${again ? about(state.estimate) : scanEstimate()}` : "";
  setProgress(null, careful ? `Đang chuyển công thức${hint}…` : "Đang chuyển…");
  const started = performance.now();
  try {
    const result = careful
      ? await runTask("pdf-to-word", options, "Đang chuyển công thức", hint, "Đang chờ đến lượt chuyển công thức…")
      : await runTask("pdf-to-word", options, "Đang chuyển");
    if (gen !== generation) return;
    const info = result.options;
    const stem = state.name.replace(/\.pdf$/i, "");
    $("download").href = `/v1/files/${result.output_files[0]}?name=${encodeURIComponent(stem + ".docx")}`;
    const seconds = (performance.now() - started) / 1000;
    $("summary").textContent = `${info.page_count} trang · ${duration(seconds)}`;

    const notes = [];
    if (info.used_ocr) notes.push(`Đã đọc chữ từ ảnh ở ${info.pages_ocr || "một số"} trang – nên soát lại chính tả.`);
    if (info.pages_image) notes.push(`${info.pages_image} trang scan giữ dạng ảnh. Chọn "Đọc chữ từ ảnh" để sửa được chữ.`);
    if (info.legacy_font_spans) notes.push("Đã đổi font VNI/TCVN3 sang Unicode.");
    if (info.removed_watermarks || info.removed_lines) notes.push("Đã bỏ watermark, chữ chìm và link ở đầu/chân trang.");
    if (info.vietocr_skipped) notes.push("File scan rất dài nên chữ chỉ được đọc nhanh – nên soát kỹ chính tả.");
    if (info.careful) {
      // trang scan đọc bằng mô hình: vùng công thức viết từ kết quả đọc / còn là ảnh
      const scans = scanCareful && info.formulas === "model";
      const read = (info.native_formulas_read || 0) + (scans ? info.formulas_read || 0 : 0);
      const kept = (info.native_formulas_unsure || 0) + (scans ? info.formula_pictures || 0 : 0);
      if (read) notes.push(`Đã chuyển ${read} công thức – nên soát lại.`);
      if (kept) notes.push(`${kept} công thức đọc chưa chắc nên giữ dạng ảnh.`);
      if (info.native_formulas_over) {
        notes.push(`${info.native_formulas_over} công thức quá giới hạn ${info.max_formulas} công thức/file nên giữ dạng ảnh.`);
      }
      // không thử đọc: trang xoay, ảnh quá lớn, hết thời gian đọc của file
      if (info.native_formulas_skipped) notes.push(`${info.native_formulas_skipped} công thức chưa chuyển được nên giữ dạng ảnh.`);
      if (scanCareful && info.formulas_skipped) {
        notes.push(`File hơn ${ci.max_scan_pages || 5} trang scan nên công thức ở trang scan chưa chuyển.`);
      }
      const failed = info.native_formulas_failed || (scanCareful && info.formulas && info.formulas !== "model");
      if (failed || (again && !read && !kept && !info.native_formulas_over && !info.native_formulas_skipped)) {
        notes.push("Máy chủ chưa chuyển được công thức lúc này.");
      }
    }
    const items = notes.map((t) => Object.assign(document.createElement("li"), { textContent: t }));
    // Công thức còn dạng ảnh mà mô hình đọc được: của trang có chữ (tối đa max_formulas mỗi file), của trang scan khi
    // ô của trang scan dùng được. Một nút chuyển lại với mô hình (cả trang scan khi ô dùng được); thời gian ước: lần
    // vừa rồi + mỗi công thức, mỗi trang có công thức dạng ảnh (Tesseract đọc lại trang), mỗi trang scan.
    const readable = info.native_readable_pictures ?? info.native_formula_pictures;  // ảnh lần chuyển lại sẽ thử đọc
    const nativeLeft = !info.careful && ci.available && ci.max_formulas > 0 ? readable || 0 : 0;
    const scanLeft = !info.careful && state.carefulOk ? info.formula_pictures || 0 : 0;
    if (nativeLeft + scanLeft) {
      state.estimate = seconds
        + Math.min(nativeLeft, ci.max_formulas) * (ci.seconds_per_formula ?? 1)
        + (nativeLeft ? (info.native_picture_pages || 0) * (ci.seconds_per_picture_page ?? 5) : 0)
        + (state.carefulOk ? ocrPages() * (ci.seconds_per_scan_page ?? 70) : 0);
      const btn = Object.assign(document.createElement("button"), {
        type: "button", className: "linkbtn", textContent: "Chuyển thành công thức sửa được",
      });
      btn.onclick = () => {
        btn.disabled = true;
        $("careful").checked = state.carefulOk;  // careful đọc cả trang scan bằng mô hình: ghi chú tính cả chúng
        convert(null, true);
      };
      const li = document.createElement("li");
      li.append(`${nativeLeft + scanLeft} công thức đang là ảnh – `, btn, ` (${about(state.estimate)})`);
      items.push(li);
    }
    $("notes").replaceChildren(...items);
    $("notes").hidden = !items.length;

    const pageKinds = [
      info.pages_native && `${info.pages_native} chữ gốc`,
      info.pages_ocr && `${info.pages_ocr} OCR`,
      info.pages_image && `${info.pages_image} ảnh`,
    ].filter(Boolean).join(" · ");
    const stats = [
      ["Trang", `${info.page_count} (${pageKinds})`],
      ["Hình", info.pictures],
      ["Đầu/chân trang", info.header_footer ? "Có" : "Không"],
    ];
    if (info.used_ocr) {
      stats.push(["Ngôn ngữ OCR", info.ocr_languages.map((l) => LANG_LABEL[l] || l).join(", ")]);
      stats.push(["Số từ đọc được", info.word_count], ["Độ tin cậy TB", `${info.avg_confidence}%`]);
    }
    if (info.legacy_font_spans) stats.push(["Đoạn font cũ đã đổi", info.legacy_font_spans]);
    if (info.removed_watermarks || info.removed_lines) {
      stats.push(["Đã bỏ", `${info.removed_watermarks} watermark · ${info.removed_lines} dòng`]);
    }
    $("stats").replaceChildren(...stats.map(([k, v]) => {
      const d = document.createElement("div");
      d.append(Object.assign(document.createElement("small"), { textContent: k }), String(v));
      return d;
    }));
    $("busy").hidden = true;
    $("done").hidden = false;
  } catch (err) {
    if (gen !== generation) return;
    if (again) {
      for (const b of document.querySelectorAll("#notes .linkbtn")) b.disabled = false;
    } else {
      $("options").hidden = false;
      if (err.code === "careful_busy" && scanCareful) {
        err = Object.assign(new Error("careful_busy_tick"), { code: "careful_busy_tick" });  // ô đang tích: bỏ chọn là chuyển được ngay
      }
    }
    showError(err);
  }
}

async function loadLanguages() {
  const { languages, default: def } = await fetch("/v1/languages").then((r) => r.json());
  const choices = [];
  if (languages.includes("vie")) choices.push("vie");
  if (languages.includes("vie") && languages.includes("eng")) choices.push("vie+eng");
  for (const l of languages) if (l !== "vie") choices.push(l);
  for (const c of choices) {
    const o = new Option(LANG_LABEL[c] || c, c);
    o.selected = c === def.join("+");
    $("lang").append(o);
  }
}

$("drop").onclick = () => $("input").click();
$("input").onchange = () => $("input").files[0] && handleFile($("input").files[0]);
for (const ev of ["dragenter", "dragover"]) {
  $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.add("over"); });
}
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]);
};
// Thả file ra ngoài ô thì trình duyệt mở file thay cho trang: chặn lại.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());
$("options").onsubmit = convert;
$("options").onchange = showMode;
for (const b of document.querySelectorAll("[data-again]")) b.onclick = reset;
loadLanguages();
