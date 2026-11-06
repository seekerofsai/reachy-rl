#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec uv run reachy-rl evaluate "$@"
