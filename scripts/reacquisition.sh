#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec uv run python -m reachy_rl.reacquisition "$@"
