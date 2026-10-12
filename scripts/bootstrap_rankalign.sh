#!/usr/bin/env bash
# Clone the upstream TriviaQA source without vendoring third-party files.
#
# Optional: RANKALIGN_REF=<verified upstream commit SHA> bash scripts/bootstrap_rankalign.sh
# An explicit ref is necessary for exact upstream reproduction. With no ref,
# the script retains its historical behavior of cloning the current upstream
# default branch, which is NOT necessarily the revision originally used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/rankalign"
URL="https://github.com/juand-r/rankalign.git"
REF="${RANKALIGN_REF:-}"

if [[ -d "$DEST/.git" ]]; then
  CURRENT="$(git -C "$DEST" rev-parse HEAD)"
  if [[ -n "$REF" ]]; then
    EXPECTED="$(git -C "$DEST" rev-parse --verify "${REF}^{commit}" 2>/dev/null || true)"
    if [[ -z "$EXPECTED" || "$CURRENT" != "$EXPECTED" ]]; then
      echo "Existing RankAlign checkout ($CURRENT) is not the requested ref ($REF)." >&2
      echo "Inspect/move the existing checkout manually; refusing to change it." >&2
      exit 1
    fi
  fi
  echo "RankAlign already present at $DEST"
  git -C "$DEST" status --short
  echo "commit: $CURRENT"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
if [[ -n "$REF" ]]; then
  git clone "$URL" "$DEST"
  git -C "$DEST" checkout --detach "$REF"
else
  git clone --depth 1 "$URL" "$DEST"
fi
echo "RankAlign cloned at commit: $(git -C "$DEST" rev-parse HEAD)"
