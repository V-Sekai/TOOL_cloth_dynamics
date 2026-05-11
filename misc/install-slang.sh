#!/usr/bin/env bash
# Fetch the upstream Slang shader compiler into bin/.slang/ for use by
# the bin/slangc wrapper. Run from the repo root.
#
# Why this isn't a brew install: brew has no Slang shader formula
# (the formula named `s-lang` is John E. Davis's S-Lang interpreter,
# unrelated). Upstream ships pre-built tarballs on GitHub Releases.
#
# The fetched ~200MB toolchain is gitignored under bin/.slang/.
# bin/slangc is a checked-in shim that exec's bin/.slang/bin/slangc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLANG_DIR="$REPO_ROOT/bin/.slang"
SLANG_VERSION="${SLANG_VERSION:-2026.8.1}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)   ASSET="slang-${SLANG_VERSION}-macos-aarch64.zip" ;;
  Darwin-x86_64)  ASSET="slang-${SLANG_VERSION}-macos-x86_64.zip"  ;;
  Linux-x86_64)   ASSET="slang-${SLANG_VERSION}-linux-x86_64.tar.gz" ;;
  Linux-aarch64)  ASSET="slang-${SLANG_VERSION}-linux-aarch64.tar.gz" ;;
  *)
    echo "no prebuilt slang asset for $(uname -s)-$(uname -m)" >&2
    echo "see https://github.com/shader-slang/slang/releases" >&2
    exit 1
    ;;
esac

URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${ASSET}"

echo "Fetching $URL"
mkdir -p "$SLANG_DIR"
TMP="$(mktemp -d)"
trap "rm -rf '$TMP'" EXIT

cd "$TMP"
curl -fL -o slang.archive "$URL"
case "$ASSET" in
  *.zip)     unzip -q slang.archive ;;
  *.tar.gz)  tar -xzf slang.archive ;;
esac

# Strip the toolchain into bin/.slang/, replacing whatever was there.
rm -rf "$SLANG_DIR"/{bin,lib,include,share,LICENSE,README.md} 2>/dev/null || true
mv bin lib include share LICENSE README.md "$SLANG_DIR"/

# Smoke-test the shim.
"$REPO_ROOT/bin/slangc" -v
echo "Installed Slang ${SLANG_VERSION} into bin/.slang/"
