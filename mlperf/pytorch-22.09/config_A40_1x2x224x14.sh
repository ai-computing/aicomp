## DL params
export BATCHSIZE=224
export GRADIENT_STEPS=14
export LR=3.7e-4
export MAX_SAMPLES_TERMINATION=20000000
export MAX_STEPS=7100
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0

#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=2 --fused_bias_fc --fused_bias_mha --fused_dropout_add "
export EXTRA_PARAMS="--dense_seq_output --unpad --exchange_padding --dwu-group-size=2 --fused_bias_fc --fused_dropout_add "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=1
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXSYSTEM="A40_1x2x224x14"
export WALLTIME=04:00:00

## System config params
export DGXNGPU=2
export DGXSOCKETCORES=8
export DGXNSOCKET=1
export DGXHT=2         # HT is on is 2, HT off is 1

export CONT=swsok/mlperf-nvidia:language_model
export DATADIR="/home/swosok/mlperf/bert/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR_PHASE2="/home/swsok/mlperf/bert/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/home/swsok/mlperf/bert/hdf5/eval_varlength"
export CHECKPOINTDIR_PHASE1="/home/swsok/mlperf/bert/phase1"
export CHECKPOINTDIR="/home/swsok/mlperf/bert/checkpoints"
export CUDA_VISIBLE_DEVICES="0,1"
export NEXP=1
