#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法: bash scripts/train_smallobj.sh <config> [work_dir] [extra args...]"
  exit 1
fi

CONFIG="$1"
WORK_DIR="${2:-work_dirs/$(basename "${CONFIG%.py}")}"

if [ $# -ge 2 ]; then
  shift 2
else
  shift 1
fi

if command -v mim >/dev/null 2>&1; then
  mim train mmyolo "$CONFIG" --work-dir "$WORK_DIR" "$@"
else
  python -m mim train mmyolo "$CONFIG" --work-dir "$WORK_DIR" "$@"
fi
