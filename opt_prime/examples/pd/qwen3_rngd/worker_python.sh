#!/bin/bash
# Adapt the existing worker launch without modifying the installed serving package.
set -eu
if [ "${1:-}" != -m ] || [ "${2:-}" != opt_prime.accel.pd.prefill_server ]; then
    echo 'expected SubprocessPrefillWorker module launch' >&2
    exit 2
fi
shift 2
exec "$(dirname "$0")/furiosa_prefill_python.sh" -m worker "$@"
