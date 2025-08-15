#!/usr/bin/env bash
# Auto commit & push watcher.
# Starts a loop that checks for changes every INTERVAL seconds (default 60) and auto commits & pushes.
# Override interval: INTERVAL=30 ./auto_push.sh
set -euo pipefail

required_name="barry-mig"
required_email="70722391+barry-mig@users.noreply.github.com"
current_name="$(git config user.name 2>/dev/null || true)"
current_email="$(git config user.email 2>/dev/null || true)"
if [ "$current_name" != "$required_name" ]; then
  git config user.name "$required_name"
  echo "Set git user.name to $required_name"
fi
if [ "$current_email" != "$required_email" ]; then
  git config user.email "$required_email"
  echo "Set git user.email to $required_email"
fi

INTERVAL="${INTERVAL:-60}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[auto-push] Watching repo on branch $BRANCH every ${INTERVAL}s. Stop with: pkill -f auto_push.sh"

auto_commit() {
  # Stage changes
  git add -A
  if git diff --cached --quiet; then
    return 0
  fi
  ts="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  msg="auto: ${ts}"
  if git commit -m "$msg" >/dev/null 2>&1; then
    if git push -u origin "$BRANCH" >/dev/null 2>&1; then
      echo "[auto-push] Committed & pushed: $msg"
    else
      echo "[auto-push] Push failed (will retry next cycle)"
    fi
  else
    echo "[auto-push] Commit failed (maybe race); will retry."
  fi
}

# Main loop
while true; do
  if ! git diff --quiet || ! git diff --cached --quiet; then
    auto_commit
  fi
  sleep "$INTERVAL"
  # Optional stop file mechanism
  if [ -f .auto_stop ]; then
    echo "[auto-push] .auto_stop detected. Exiting."; exit 0
  fi
done
