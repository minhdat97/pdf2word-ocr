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
};
const fmt = (n, d = 1) => n.toLocaleString("vi-VN", { maximumFractionDigits: d });
let state = {};

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

async function runTask(tool, options, label) {
  const task_id = newTaskId();
  await api("POST", "/v1/tasks", { tool, task_id, input_tokens: [state.token], options });
  const deadline = Date.now() + 10 * 60 * 1000;
  let logged = false;
  while (true) {
    if (Date.now() > deadline) throw Object.assign(new Error("timeout"), { code: "timeout" });
    await new Promise((r) => setTimeout(r, 700));
    let res;
    try {
      res = await fetch(`/v1/tasks/${task_id}`).then((r) => r.json());
    } catch {
      throw Object.assign(new Error("network"), { code: "network" });
    }
    const data = res.data;
    if (!data) throw new Error(res.error || "Không lấy được trạng thái");
    const { current, total } = data.progress;
    if (total && label) setProgress(current / total, `${label} trang ${Math.min(current + 1, total)}/${total}…`);
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
  state = {};
  for (const id of ["error", "work", "options", "done", "kind", "failed"]) $(id).hidden = true;
  $("pick").hidden = false;
  $("input").value = "";
  $("strip").checked = true;
  $("full").checked = false;
  for (const d of document.querySelectorAll("#work details")) d.open = false;
}

function showMode() {
  $("ocrrow").hidden = new FormData($("options")).get("mode") !== "ocr";
}

async function handleFile(file) {
  reset();
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
    if (analysis.password_protected) throw Object.assign(new Error("password"), { code: "password" });
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
    showError(err);
  }
}

async function convert(e) {
  e.preventDefault();
  const mode = new FormData($("options")).get("mode");
  const ocrType = mode === "ocr" ? ($("full").checked ? "ocr-full" : "ocr") : "";
  const options = {
    ocr_type: ocrType,
    languages: ocrType ? $("lang").value.split("+") : [],
    strip_watermarks: $("strip").checked,
  };
  $("options").hidden = true;
  $("error").hidden = true;
  $("busy").hidden = false;
  setProgress(null, "Đang chuyển…");
  const started = performance.now();
  try {
    const result = await runTask("pdf-to-word", options, "Đang chuyển");
    const info = result.options;
    const stem = state.name.replace(/\.pdf$/i, "");
    $("download").href = `/v1/files/${result.output_files[0]}?name=${encodeURIComponent(stem + ".docx")}`;
    const seconds = (performance.now() - started) / 1000;
    $("summary").textContent = `${info.page_count} trang · ${fmt(seconds)} giây`;

    const notes = [];
    if (info.used_ocr) notes.push(`Đã đọc chữ từ ảnh ở ${info.pages_ocr || "một số"} trang – nên soát lại chính tả.`);
    if (info.pages_image) notes.push(`${info.pages_image} trang scan giữ dạng ảnh. Chọn "Đọc chữ từ ảnh" để sửa được chữ.`);
    if (info.legacy_font_spans) notes.push("Đã đổi font VNI/TCVN3 sang Unicode.");
    if (info.removed_watermarks || info.removed_lines) notes.push("Đã bỏ watermark, chữ chìm và link ở đầu/chân trang.");
    $("notes").replaceChildren(...notes.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
    $("notes").hidden = !notes.length;

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
    $("options").hidden = false;
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
