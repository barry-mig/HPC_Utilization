#!/usr/bin/env bash
# Quick push helper.
# Usage: ./push.sh "commit message"
set -euo pipefail

# Ensure correct identity (uses local repo config; change to --global if desired)
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

msg="${1:-chore: update}"
branch="$(git rev-parse --abbrev-ref HEAD)"

git add -A
if git diff --cached --quiet; then
  echo "No changes to commit"
else
  git commit -m "$msg"
fi

git push -u origin "$branch"
