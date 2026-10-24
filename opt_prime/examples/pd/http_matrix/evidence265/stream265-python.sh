#!/bin/bash
set -eu
exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 /tmp/codex-rngd-gpu-201/bin/python -c '
import runpy,sys
from furiosa.torch._register import _register
_register()
import rngd_pd.engines.fsdk as sdk
sdk.open_device=lambda *a,**k:(sdk.set_fusion(4),sdk.set_mesh_kind(sdk.MeshKind.Mesh(2)))
sys.argv=["prefill_server"]+sys.argv[3:]
runpy.run_module("opt_prime.accel.pd.prefill_server",run_name="__main__")
' "$@"
