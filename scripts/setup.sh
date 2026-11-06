#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
uv sync --extra dev
uv run reachy-rl fetch-assets
uv run reachy-rl check
