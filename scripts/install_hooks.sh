#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
mkdir -p .git/hooks
for h in pre-commit commit-msg pre-push; do
  cp .github/hooks/$h .git/hooks/$h
  chmod +x .git/hooks/$h
done
echo "Hooks installed."
