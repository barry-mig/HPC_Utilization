#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=auto_push.log
MAX_SIZE=$((1024*50)) # 50KB for demo
if [ -f "$LOG" ]; then
  size=$(wc -c < "$LOG")
  if [ "$size" -gt "$MAX_SIZE" ]; then
    ts=$(date -u +%Y%m%d-%H%M%S)
    mv "$LOG" "${LOG%.log}-$ts.log"
    echo "Rotated $LOG"
  fi
fi
