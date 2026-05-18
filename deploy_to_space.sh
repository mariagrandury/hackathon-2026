#!/usr/bin/env bash
# Sync runtime files into a local clone of the HF Space repo.
#
# Usage:
#   ./deploy_to_space.sh [path/to/space-clone]
#
# Default path: ../2026-space-cultural-preferences (sibling of this repo).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_DIR="${1:-${REPO_DIR}/../2026-space-cultural-preferences}"

if [[ ! -d "$SPACE_DIR/.git" ]]; then
  echo "error: '$SPACE_DIR' is not a git repo." >&2
  echo "" >&2
  echo "Clone the Space first:" >&2
  echo "  git clone https://huggingface.co/spaces/somosnlp-hackathon-2026/cultural-preferences \\" >&2
  echo "    \"$SPACE_DIR\"" >&2
  exit 1
fi

# Soft check: warn if the remote doesn't look like a HF Space.
remote_url="$(git -C "$SPACE_DIR" remote get-url origin 2>/dev/null || true)"
if [[ "$remote_url" != *"huggingface.co/spaces/"* ]]; then
  echo "warning: '$SPACE_DIR' origin is '$remote_url'" >&2
  echo "         expected huggingface.co/spaces/... — continuing anyway." >&2
fi

# Whitelist of paths to mirror into the Space. Anything not in this list
# stays out of the Space repo (CLAUDE.md, seed/migrate scripts, tests,
# data/, .env, etc.).
#
# NOTE: the entry-test bank (``data/test-2026.json``) is deliberately
# NOT here — it would expose the answer key to anyone admitted to the
# private Space. ``test_data.py`` fetches it at runtime from the private
# ``mariagrandury/hackathon_test_bank`` dataset using ``HF_TOKEN``.
PATHS=(
  app.py
  data.py
  test_data.py
  requirements.txt
  README.md
  guidelines
  images
)

echo "Syncing runtime files into $SPACE_DIR"
for p in "${PATHS[@]}"; do
  src="$REPO_DIR/$p"
  if [[ ! -e "$src" ]]; then
    echo "  [skip] $p (missing in source)"
    continue
  fi
  if [[ -d "$src" ]]; then
    # Mirror directory contents; --delete removes files in dest that no
    # longer exist in src (so renames/removals propagate).
    # ``test.md`` under guidelines/ is dev notes — keep it out of the
    # Space mirror.
    mkdir -p "$SPACE_DIR/$p"
    rsync -a --delete --exclude="test.md" "$src/" "$SPACE_DIR/$p/"
  else
    # Ensure the destination's parent directory exists. Today every
    # file in PATHS is at the repo root, so this is a no-op; kept so
    # adding a nested file later (e.g. ``some/dir/file.py``) doesn't
    # silently fail when ``some/dir`` doesn't exist on the Space side.
    mkdir -p "$(dirname "$SPACE_DIR/$p")"
    rsync -a "$src" "$SPACE_DIR/$p"
  fi
  echo "  [ok]   $p"
done

echo
echo "Sync complete. Review and publish:"
echo "  cd \"$SPACE_DIR\""
echo "  git status"
echo "  git add -A"
echo "  git commit -m \"Sync from main repo\""
echo "  git push"
