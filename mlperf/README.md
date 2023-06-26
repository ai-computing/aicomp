# MLPerf HOWTO
This document describes how to prepare dataset for MLPerf bert benchmark and to run the tests.  
ETRI AI Computing platform is consists as followings.  
- 8 nodes   
Each node has   
- AMD EPYC 7313 16-Core Processor x 2   
- 512GB Memory
- 1 Gigabit Ethernet x 1  
- 100 Gb/s Infiniband x 1  
- Nvidia A40 GPU x 8

## Building Docker image
Change to pytorch-22.09/ directory.  
```
docker build --pull -t swsok/mlperf-nvidia:language_model .
docker push swsok/mlperf-nvidia:language_model
```
## Prepareing dataset
Make bert data directory.  
```
cd
mkdir -p mlperf/bert
```
Run a docker container.  
```
docker run -it --runtime=nvidia --ipc=host -v /home/etri/mlperf/bert:/workspace/bert_data swsok/mlperf-nvidia:language_model
```
Within the container, run following command to download original datasets from Google drive and process it.  
```
cd /workspace/bert
./input_preprocessing/prepare_data.sh --outputdir /workspace/bert_data
```
This script will download the required data and model files from [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) creating the following foldes structure  
```
/workspace/bert_data/
                     |_ download
                        |_results4                               # 500 chunks with text data
                        |_bert_reference_results_text_md5.txt    # md5 checksums for text chunks
                     |_ phase1                                   # checkpoint to start from (both tf1 and pytorch converted)
                     |_hdf5
                           |_ eval                               # evaluation chunks in binary hdf5 format fixed length (not used in training, can delete after data   preparation)
                           |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length *used for training*
                           |_ training                           # 500 chunks in binary hdf5 format
                           |_ training_4320                      #
                              |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length (not used in training, can delete after data   preparation)
                              |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length *used for training*
```
After all processing done, close the container.  
```
exit
```
   
## Run 2 GPU benchmark for A40 GPU
Execute following commands.  
```
export CONT=swsok/mlperf-nvidia:language_model
export DATADIR="/home/etri/mlperf/bert/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR_PHASE2="/home/etri/mlperf/bert/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/home/etri/mlperf/bert/hdf5/eval_varlength"
export CHECKPOINTDIR_PHASE1="/home/etri/mlperf/bert/phase1/"
export CHECKPOINTDIR="/home/etri/mlperf/bert/checkpoints"
export CUDA_VISIBLE_DEVICES="0,1"
source config_A40_1x2x224x14.sh
./run_with_docker.sh
```
Because A40 GPUs don't support some features of A100 and A30, we removed following parameters.  
It causes some performance degradation comparing to A100 and A30.  
```
--unpad_fmha		# Support unpadded computing with Fused Multi-Head Attention kernel.
--fused_bias_mha	# Support fused bias addition in Multi-Head Attention mechanism.
```
## Run Multi node benchmark
To be updated...  
