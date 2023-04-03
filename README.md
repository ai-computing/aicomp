## High-efficiency AI computing SW core technology project (AIcomp)

This project is for the development of low-cost and high-efficiency AI computing platform technology to overcome the inefficiency of using excessive computing resources required for learning a large model and the dependency on a specific high-cost hypercluster when training a large learning model.

For now, we present the result of developing PoC that applies out of order technology (https://dl.acm.org/doi/pdf/10.1145/3492321.3519563) to pipeline parallelization in torchgpipe_OOO_PP folder.

In addition, we want to apply 3D to the model based on the compiler. In this regard, related PoCs are being developed preemptively, and the ETRI framework SW will be developed in earnest in the second half of the year.

As part of the related project, tunib, a joint research and development organization, is separately carrying out the OSLO project(https://github.com/EleutherAI/oslo).

## Usage

This SW requires:
1. torchgpipe_OOO_PP
* Python3.7
* Pytorch 1.12+ 
* torchgpipe 0.0.7

2. compiler_FX
* Python 3.8
* Pytorch 1.13.1+ (compiler_fx)

## RUN

python3 source.py

## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
