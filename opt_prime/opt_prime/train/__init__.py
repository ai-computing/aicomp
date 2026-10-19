#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Training backends: one job description (`TrainJob`), several ways to run it, chosen per model and device.

  optimus   Optimus_p pipeline training on GPUs (any model the IR exports; LoRA or full fine-tuning)
  npu-lora  the npu-lora library (pip install npu-lora): LoRA or full fine-tuning on FuriosaAI RNGD (furiosa:N) with
            CUDA / CPU reference backends; model families it implements (Llama today)

`train(job)` builds the model's block spec (no weights read), picks the first backend that supports the model on the
job's device kind - reporting why the others did not -, leases the device and runs the job. Adding a device or a
library is one TrainBackend registration; NPU-specific training mechanics stay in their library.
"""
from .backends import (NpuLoraTrainBackend, OptimusTrainBackend, TrainBackend, TrainJob, TrainResult,  # noqa: F401
                       register_train_backend, registered_train_backends, select_train_backend, train)
