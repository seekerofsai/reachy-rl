#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
uv run reachy-rl check
exec uv run --extra dev pytest -q
