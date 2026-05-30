#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "用法: bash scripts/test_smallobj.sh <config> <checkpoint> [extra args...]"
  exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
shift 2

if command -v mim >/dev/null 2>&1; then
  mim test mmyolo "$CONFIG" "$CHECKPOINT" "$@"
else
  python -m mim test mmyolo "$CONFIG" "$CHECKPOINT" "$@"
fi
