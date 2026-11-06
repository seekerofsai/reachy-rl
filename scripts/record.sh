#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
# EGL enables offscreen NVIDIA rendering over SSH. Set MUJOCO_GL explicitly to override.
if [[ "$(uname -s)" == Linux ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
exec uv run reachy-rl record "$@"
