## Test environment

SW:
* Python 3.8.10
* ubuntu 20.04.4 LTS

HW:
* AMD EPYC 7313
* NVIDIA A40

## RUN

torchrun --nproc_per_node=3 example_train_gpu.py

## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
