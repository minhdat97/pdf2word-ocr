#!/usr/bin/env bash
# Cập nhật bản đang chạy trên máy chủ: lấy mã mới từ GitHub, cài lại thư viện khi requirements.txt đổi,
# cài lại dịch vụ systemd khi deploy/pdf2word.service đổi, restart chỉ khi mã Python hoặc dịch vụ đổi
# (giao diện static/ được đọc thẳng từ ổ đĩa, không cần restart).
#   ./scripts/deploy.sh            cập nhật
#   ./scripts/deploy.sh --restart  cập nhật và restart dù không có gì đổi
# Biến môi trường: P2W_SERVICE (tên dịch vụ systemd, mặc định pdf2word), P2W_PORT (mặc định 8000).
set -euo pipefail

cd "$(dirname "$0")/.."
SERVICE="${P2W_SERVICE:-pdf2word}"
PORT="${P2W_PORT:-8000}"
UNIT="${P2W_UNIT:-/etc/systemd/system/${SERVICE}.service}"  # P2W_UNIT: chỉ để chạy thử script
force=0
[ "${1:-}" = "--restart" ] && force=1

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Có thay đổi chưa commit trên máy này – không cập nhật được:"
  git status --short --untracked-files=no
  exit 1
fi

if [ -n "${P2W_FROM:-}" ]; then  # chạy lại bằng deploy.sh mới vừa kéo về (xem dưới)
  old=$P2W_FROM
else
  old=$(git rev-parse HEAD)
  git pull --ff-only --quiet
fi
new=$(git rev-parse HEAD)
# Bản pull về có deploy.sh mới: chạy tiếp bằng bản đó (bash đang đọc bản cũ).
if [ -z "${P2W_FROM:-}" ] && ! git diff --quiet "$old" "$new" -- scripts/deploy.sh; then
  P2W_FROM=$old exec "$0" "$@"
fi

if [ "$old" = "$new" ]; then
  echo "Đã là bản mới nhất ($(git log -1 --format='%h %s'))."
  changed=""
else
  echo "Bản mới:"
  git log --format='  %h %s' "$old..$new"
  changed=$(git diff --name-only "$old" "$new")
fi

restart=$force
if printf '%s\n' "$changed" | grep -qx 'requirements.txt'; then
  echo "requirements.txt đổi – cài lại thư viện…"
  .venv/bin/pip install --quiet -r requirements.txt
  restart=1
fi
if printf '%s\n' "$changed" | grep -q '^app/'; then
  restart=1
fi
unit_changed=0
if printf '%s\n' "$changed" | grep -qx 'deploy/pdf2word.service' || ! sudo cmp -s deploy/pdf2word.service "${UNIT}"; then
  unit_changed=1
  restart=1
fi

if [ "$restart" = 0 ]; then
  [ -n "$changed" ] && echo "Chỉ đổi giao diện/tài liệu – không cần restart."
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
