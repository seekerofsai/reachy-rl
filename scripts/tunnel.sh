#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
printf 'Reachy viewer: http://localhost:8081\nTensorBoard: http://localhost:6007\n'
exec ssh -N -o ExitOnForwardFailure=yes -L 8081:localhost:8081 -L 6007:localhost:6007 "${SPARK_HOST:-saiseeker@spark-c64c}"
