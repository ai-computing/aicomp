set -eu
export PYTHONUSERBASE=/home/gspark/.local
export OPT_PRIME_CACHE=/tmp/kv194-cache OPT_PRIME_DEVICE_SOURCE=local
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/tmp/kv194-xdg HF_HOME=/home/gspark/.cache/huggingface
export RBLN_DEVICES=0 RBLN_NPUS_PER_DEVICE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/home/gspark/workspace/codex-serving-matrix-201/stream265-code:/home/gspark/workspace/codex-serving-matrix-201:/home/gspark/workspace/claude/aicomp-134/opt_prime:/home/gspark/workspace/01_pd_disag/rngd_llmd/src:/home/gspark/workspace/01_pd_disag/hetero-npu-p2p/src
cd /home/gspark/workspace/codex-serving-matrix-201
export FURIOSA_PYTHON=/home/gspark/venvs/furiosa-2026.4/bin/python
exec /home/gspark/workspace/01_pd_disag/experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python -u container_bootstrap.py stream265_atom.py "$@"
