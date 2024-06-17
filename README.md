# High-efficiency AI computing SW core technology project (AIcomp)

This project is for the development of low-cost and high-efficiency AI computing platform technology to overcome the inefficiency of excessive computing resource usage required for training large models and the dependency on specific high-cost hyperclusters for training such models.

We are developing a parallelization framework software called OptimusPrime, which now provides two-dimensional parallelization of pipeline parallel/data parallel and memory-efficient optimization features ( [opt_prime folder](./opt_prime) ).


Additionally, we have developed multiple PoCs related to this: In the early stages of this project, we presented the results of developing training PoCs that integrated Out Of Order technology (https://dl.acm.org/doi/pdf/10.1145/3492321.3519563) on top of torchgpipe ( [torchgpipe_OOO_PP folder](./torchgpipe_OOO_PP) ). In the next stage, we developed multiple PoCs that extract IR from the model and perform distributed training by partitioning it across multiple GPUs ( [compiler_fx folder](./compiler_fx) ).


In addition, we want to apply 3D to the model based on the compiler. In this regard, related PoCs are being developed preemptively, and the ETRI framework SW will be developed in earnest in the second half of the year.


## Features

An open-source AI training framework that provides automatic parallelization without model modifications ( [opt_prime](./opt_prime) )

* Enabling general model application for parallelization by removing constraints on model representation (compatible with Hugging Face models and PyTorch nn.Module)
* Automatic parallelization (model split) without user intervention
* Distributed parallel runtime supporting both Intra/Inter hosts concurrently (currently supports PP + DP)
* An IR-based system aiming for flexible optimization at a global level
* Memory optimization technology for CPU/GPU memory OOM avoidance


## License

The results of the AIcomp project are distributed under the 3-clause BSD license.
