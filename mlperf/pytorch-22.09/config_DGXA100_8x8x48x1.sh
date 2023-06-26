## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=0.0020992
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=1059
export OPT_LAMB_BETA_1=0.60466
export OPT_LAMB_BETA_2=0.85437
export START_WARMUP_STEP=0
export WARMUP_STEPS=0
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=4096.0

export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=15
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME_MINUTES} + 5 ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
