#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ "${1:-}" == record && "$(uname -s)" == Linux ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
exec uv run python -m reachy_rl.advanced photo "$@"
