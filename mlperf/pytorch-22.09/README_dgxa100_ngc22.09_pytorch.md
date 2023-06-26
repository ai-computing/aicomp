## Steps to launch training on a single node

### NVIDIA DGX A100 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
multi node submission are in the following scripts:
* for the 1-node NVIDIA DGX A100 submission: `config_DGXA100_1x8x56x1.sh`

Steps required to launch multi node training on NVIDIA DGX A100:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

2. Launch the training:

1-node NVIDIA DGX A100 training:

```
source config_DGXA100_1x8x56x1.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1 sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
