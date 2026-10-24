#!/bin/bash
set -eu
exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 /tmp/codex-rngd-gpu-201/bin/python /tmp/stream265-controlled-server.py "$@"
