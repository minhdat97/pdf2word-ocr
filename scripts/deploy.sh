#!/usr/bin/env bash
# Cập nhật bản đang chạy trên máy chủ: lấy mã mới từ GitHub, cài lại thư viện khi requirements.txt đổi,
# restart dịch vụ chỉ khi mã Python đổi (giao diện static/ được đọc thẳng từ ổ đĩa, không cần restart).
#   ./scripts/deploy.sh            cập nhật
#   ./scripts/deploy.sh --restart  cập nhật và restart dù không có gì đổi
# Biến môi trường: P2W_SERVICE (tên dịch vụ systemd, mặc định pdf2word), P2W_PORT (mặc định 8000).
set -euo pipefail

cd "$(dirname "$0")/.."
SERVICE="${P2W_SERVICE:-pdf2word}"
PORT="${P2W_PORT:-8000}"
force=0
[ "${1:-}" = "--restart" ] && force=1

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Có thay đổi chưa commit trên máy này – không cập nhật được:"
  git status --short --untracked-files=no
  exit 1
fi

old=$(git rev-parse HEAD)
git pull --ff-only --quiet
new=$(git rev-parse HEAD)

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

if [ "$restart" = 0 ]; then
  [ -n "$changed" ] && echo "Chỉ đổi giao diện/tài liệu – không cần restart."
  exit 0
fi

echo "Restart ${SERVICE}…"
sudo systemctl restart "${SERVICE}"
for _ in $(seq 1 30); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)" = 200 ]; then
    echo "Xong: ${SERVICE} đang chạy bản $(git log -1 --format='%h')."
    exit 0
  fi
  sleep 1
done
echo "Sau 30 giây ${SERVICE} vẫn chưa trả lời. Xem log: journalctl -u ${SERVICE} -n 50 --no-pager"
exit 1
