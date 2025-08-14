#!/usr/bin/env bash
# Quick push helper.
# Usage: ./push.sh "commit message"
set -euo pipefail
msg="${1:-chore: update}"
branch="$(git rev-parse --abbrev-ref HEAD)"
 git add -A
if git diff --cached --quiet; then
  echo "No changes to commit"
else
  git commit -m "$msg"
fi
git push -u origin "$branch"
