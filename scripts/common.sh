#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
command -v uv >/dev/null 2>&1 || { printf 'Install uv: https://docs.astral.sh/uv/getting-started/installation/\n' >&2; exit 1; }
