#!/bin/bash
# Validation-only Python shim for SubprocessPrefillWorker's `-m module` launch.
set -eu
: "${FURIOSA_PYTHON:?set the validated Furiosa Python interpreter}"
exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$FURIOSA_PYTHON" -c '
import sys,runpy
from furiosa.torch._register import _register
_register()
import rngd_pd.engines.fsdk as sdk
# The allocated single-card Pod aborts on fusion=8/Mesh(2); fusion=4 passed a round trip.
sdk.open_device=lambda *a,**k: (sdk.set_fusion(4),sdk.set_mesh_kind(sdk.MeshKind.Mesh(2)))
if len(sys.argv)<3 or sys.argv[1]!="-m":raise SystemExit("expected -m module [args]")
module=sys.argv[2];sys.argv=sys.argv[2:]
runpy.run_module(module,run_name="__main__",alter_sys=True)
' "$@"
