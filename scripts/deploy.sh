#!/usr/bin/env bash
# Cập nhật bản đang chạy trên máy chủ: lấy mã mới từ GitHub rồi so với bản đã triển khai lần trước (ghi trong
# .git/pdf2word-deployed sau mỗi lần xong, nên "git pull" tay trước đó không làm sót gì; chưa có file này thì coi
# như mọi thứ đổi): cài lại thư viện khi requirements.txt đổi, tải mô hình khi scripts/models.json đổi hoặc
# mô hình còn thiếu/hỏng (fetch_models.py --check; lần tải hỏng được tải lại ở lần chạy sau), cài lại dịch vụ
# systemd khi deploy/pdf2word.service đổi, restart chỉ khi mã Python, mô hình hoặc dịch vụ đổi (giao diện static/
# được đọc thẳng từ ổ đĩa, không cần restart). Cuối cùng nhắc nếu một drop-in systemd đang tắt mô hình.
#   ./scripts/deploy.sh            cập nhật
#   ./scripts/deploy.sh --restart  cập nhật và restart dù không có gì đổi
# Biến môi trường: P2W_SERVICE (tên dịch vụ systemd, mặc định pdf2word), P2W_PORT (mặc định 8000).
set -euo pipefail

cd "$(dirname "$0")/.."
SERVICE="${P2W_SERVICE:-pdf2word}"
PORT="${P2W_PORT:-8000}"
UNIT="${P2W_UNIT:-/etc/systemd/system/${SERVICE}.service}"  # P2W_UNIT: chỉ để chạy thử script
STATE="$(git rev-parse --git-dir)/pdf2word-deployed"  # commit đã triển khai xong lần trước
force=0
[ "${1:-}" = "--restart" ] && force=1

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Có thay đổi chưa commit trên máy này – không cập nhật được:"
  git status --short --untracked-files=no
  exit 1
fi

if [ -z "${P2W_FROM:-}" ]; then  # P2W_FROM: đã pull, đang chạy bằng deploy.sh mới vừa kéo về (xem dưới)
  pre=$(git rev-parse HEAD)
  git pull --ff-only --quiet
  # Bản pull về có deploy.sh mới: chạy tiếp bằng bản đó (bash đang đọc bản cũ).
  if ! git diff --quiet "$pre" HEAD -- scripts/deploy.sh; then
    P2W_FROM=$pre exec "$0" "$@"
  fi
fi
new=$(git rev-parse HEAD)

old=""
if [ -f "$STATE" ]; then
  old=$(tr -d '[:space:]' < "$STATE")
  git cat-file -e "${old}^{commit}" 2>/dev/null || old=""
fi
if [ -z "$old" ]; then
  echo "Chưa biết bản đã triển khai lần trước – cài lại, tải mô hình và restart như mọi thứ đều đổi."
  changed=$(git ls-files)
elif [ "$old" = "$new" ]; then
  echo "Đã là bản mới nhất ($(git log -1 --format='%h %s'))."
  changed=""
else
  echo "Bản mới:"
  git log --format='  %h %s' "$old..$new"
  changed=$(git diff --name-only "$old" "$new")
fi

# Ghi bản vừa triển khai xong; nhắc drop-in systemd đang tắt mô hình hay đổi cách đọc công thức.
finish() {
  git rev-parse HEAD > "$STATE"
  local f vars
  for f in "${UNIT}.d"/*.conf; do
    [ -r "$f" ] || continue
    vars=$(grep -E '^[[:space:]]*Environment=' "$f" | grep -oE 'P2W_(VIETOCR|MFR)=0|P2W_FORMULAS=[A-Za-z]*' | sort -u | tr '\n' ' ' || true)
    if [ -n "$vars" ]; then
      echo "Chú ý: $f đặt ${vars% } – bỏ: sudo rm $f && sudo systemctl daemon-reload && sudo systemctl restart ${SERVICE}"
    fi
  done
}

restart=$force
if printf '%s\n' "$changed" | grep -qx 'requirements.txt'; then
  echo "requirements.txt đổi – cài lại thư viện…"
  .venv/bin/pip install --quiet -r requirements.txt
  restart=1
fi
models=0
if printf '%s\n' "$changed" | grep -qx 'scripts/models.json'; then
  models=1
elif ! .venv/bin/python scripts/fetch_models.py --check > /dev/null 2>&1; then
  models=1
fi
if [ "$models" = 1 ]; then
  echo "Tải mô hình vào models/…"
  fetched_ok=1
  out=$(.venv/bin/python scripts/fetch_models.py 2>&1) || fetched_ok=0
  printf '%s\n' "$out" | grep -v '^ok ' || true
  if printf '%s\n' "$out" | grep -q '^fetched'; then
    restart=1
  fi
  if [ "$fetched_ok" = 0 ]; then
    echo "Chưa tải đủ mô hình (lần chạy sau tải tiếp): công cụ chạy không có mô hình còn thiếu (VietOCR/công thức, xem README)."
  fi
fi
if printf '%s\n' "$changed" | grep -q '^app/'; then
  restart=1
fi
unit_changed=0
if ! sudo cmp -s deploy/pdf2word.service "${UNIT}"; then  # the installed unit is not the repo's
  unit_changed=1
  restart=1
fi

if [ "$restart" = 0 ]; then
  if [ -n "$changed" ] && [ "$models" = 0 ]; then
    echo "Chỉ đổi giao diện/tài liệu – không cần restart."
  fi
  finish
  exit 0
fi

healthy() {
  for _ in $(seq 1 30); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)" = 200 ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

mkdir -p data  # the service may write only there (ReadWritePaths)
if [ "$unit_changed" = 1 ]; then
  echo "Cài lại dịch vụ ${SERVICE} (deploy/pdf2word.service)…"
  sudo cp -p "${UNIT}" "${UNIT}.bak" 2>/dev/null || true
  sudo install -m 644 deploy/pdf2word.service "${UNIT}"
  sudo systemctl daemon-reload
fi
echo "Restart ${SERVICE}…"
sudo systemctl restart "${SERVICE}"
if healthy; then
  echo "Xong: ${SERVICE} đang chạy bản $(git log -1 --format='%h')."
  finish
  exit 0
fi
if [ "$unit_changed" = 1 ] && sudo test -f "${UNIT}.bak"; then
  echo "Dịch vụ mới không lên – trả lại file dịch vụ cũ…"
  sudo journalctl -u "${SERVICE}" -n 20 --no-pager || true
  sudo mv "${UNIT}.bak" "${UNIT}"
  sudo systemctl daemon-reload
  sudo systemctl restart "${SERVICE}"
  healthy && echo "Đã chạy lại với file dịch vụ cũ. Gửi đoạn log ở trên để sửa."
  exit 1
fi
echo "Sau 30 giây ${SERVICE} vẫn chưa trả lời. Xem log: journalctl -u ${SERVICE} -n 50 --no-pager"
exit 1
