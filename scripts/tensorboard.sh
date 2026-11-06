#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec uv run python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6007 "$@"
