<div align="center">

<h1>AIcomp</h1>

<h3>Automatic parallelization of large models across GPU clusters, NPUs and host memory</h3>

<p>
  <strong>English</strong> · <a href="README.ko.md">한국어</a>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L3%20unified%20engine-34d399" alt="ai-rm layer: L3 unified engine"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1%20%7C%202.13-ee4c2c?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.5.1 and 2.13">
  <img src="https://img.shields.io/badge/transformers-4.46.2%20%7C%205.x-ffcc4d" alt="transformers 4.46.2 and 5.x">
  <a href="https://dl.acm.org/doi/10.1145/3625549.3658687"><img src="https://img.shields.io/badge/FASOP-HPDC%20'24-8b5cf6" alt="FASOP: HPDC 2024"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-64748b" alt="License: BSD-3-Clause"></a>
</p>

<p>
  <a href="#-highlights"><b>Highlights</b></a> ·
  <a href="#-repository-layout"><b>Layout</b></a> ·
  <a href="#-quick-start"><b>Quick start</b></a> ·
  <a href="#-compatibility"><b>Compatibility</b></a> ·
  <a href="#-ecosystem"><b>Ecosystem</b></a> ·
  <a href="#-earlier-pocs"><b>Earlier PoCs</b></a>
</p>

</div>

AIcomp is ETRI's high-efficiency AI computing software project. Its aim is to train and serve large models
without depending on one specific high-cost hypercluster. The repository holds two projects:

- **OptimusPrime** (`opt_prime`): a parallelization framework.
- **FASOP** (`fasop`): a search engine for the parallelization strategy.

## ✨ Highlights

| | **OptimusPrime** · [`opt_prime`](opt_prime) | **FASOP** · [`fasop`](fasop) |
| --- | --- | --- |
| **What** | Training and inference framework that parallelizes Hugging Face and `nn.Module` models without model changes | Cost-based optimizer that finds the TP / PP / DP strategy for heterogeneous GPU clusters |
| **How** | Captures the model as a PyTorch FX graph (HFTracer, or TorchDynamo with `--dynamo-capture`), splits it into pipeline stages, and runs PP + DP + TP inside and across hosts | Profiles FX-node costs, models stage and communication time, and searches device placements and partitions; outputs a Pareto front |
| **Training** | GPipe / 1F1B, LoRA, CPU optimizer offload, HF-compatible checkpoints | |
| **Inference** | Prefill/decode API, three KV-cache modes, PP / TP, MPS sharing | Inference planning is future work |
| **Status** | Maintained | Maintained · [paper](https://dl.acm.org/doi/10.1145/3625549.3658687) |

**Heterogeneous accelerators: `opt_prime.accel` (experimental, transformers 5.x).** These features are optional and
off unless requested:
- **MoE expert offloading.** Hot experts run on a GPU, RNGD or ATOM tier and the cold experts on the host CPU, so MoE
  models larger than device memory can run. The same split is used for MoE training.
- **Engine selection.** OptimusPrime chooses between its own GPU engine and native NPU engines: the RNGD layer engine
  and ATOM `optimum-rbln`.
- **Disaggregated prefill/decode.** RNGD prefill feeds ATOM decode over device-to-device KV transport, with an llm-d
  sidecar for serving.

Details: [opt_prime/README.md](opt_prime/README.md) and the [PR series overview](opt_prime/docs/PR_SERIES_KO.md).

## 📦 Repository layout

```
aicomp/
├── opt_prime/            OptimusPrime
│   ├── opt_prime/        framework: IR capture, partitioning, schedules, runtime, inference
│   │   └── accel/        heterogeneous accelerators (MoE offload, engines, prefill/decode, serving)
│   ├── examples/         training and inference scripts (pp_train_gpt2.py, pp_train_llama5.py, ...)
│   ├── tests/
│   └── docs/
└── fasop/                FASOP: search, cost model, profiler (make_profiledb), demo viewer
```

## 🚀 Quick start

```bash
git clone https://github.com/ai-computing/aicomp.git
pip3 install "transformers==4.46.2" datasets    # with an installed PyTorch + CUDA; see Compatibility
cd aicomp/opt_prime/examples

# GPT-2 with pipeline parallelism on 2 GPUs (HFTracer capture)
torchrun --nproc_per_node=2 --master_port=29500 pp_train_gpt2.py

# the same through TorchDynamo capture (required with transformers 5.x)
torchrun --nproc_per_node=2 --master_port=29500 pp_train_gpt2.py --dynamo-capture

# Llama 3.2 1B with PP=2, TP=2, DP=2 on 8 GPUs (gated model: pass a Hugging Face token)
torchrun --nproc_per_node=8 --master_port=29500 pp_train_llama5.py <hf token>
```

- **Multi-node runs, parallelism options, partitioners, inference, LoRA and checkpoints:**
  [opt_prime/README.md](opt_prime/README.md).
- **Strategy search with FASOP:** [fasop/README.md](fasop/README.md).

## 🧪 Compatibility

The examples must run in both of these configurations before a change is merged:

| Capture | PyTorch | transformers | Notes |
| --- | --- | --- | --- |
| HFTracer (default) | 2.5.1 | 4.46.2 | Reference setup of the examples |
| `--dynamo-capture` | 2.13.0 | 5.15.0 | Required for `opt_prime.accel` |

Other combinations covered by the regression set: transformers 5.1.0 and 5.8.1.

`opt_prime.accel` requires transformers 5.x. Its hardware paths are validated only on the devices and SDK versions
named in [opt_prime/docs](opt_prime/docs).

## 🧩 Ecosystem

AIcomp is the engine layer of [**ai-rm**](https://github.com/ai-computing/ai-rm), the organization's resource manager
for heterogeneous AI infrastructure. OptimusPrime builds on these optional libraries, each installed separately:

| Area | Repository | Used by |
| --- | --- | --- |
| Device discovery and leases | [rm-accel](https://github.com/ai-computing/rm-accel) | `accel.platforms` |
| NPU training | [npu-train](https://github.com/ai-computing/npu-train) | `accel.train`: LoRA / full fine-tuning on RNGD and ATOM |
| Device-to-device transport | [hetero-npu-p2p](https://github.com/ai-computing/hetero-npu-p2p) | `accel.pd.npu`: RNGD → ATOM KV transfer |
| NPU kernels and offload | [npu-offload](https://github.com/ai-computing/npu-offload) | RNGD expert kernels (`npu-kernels`) |
| RNGD prefill/decode | [rngd-pd](https://github.com/ai-computing/rngd-pd) | KONI RNGD endpoints |
| Cluster serving | [hetero-llm-d](https://github.com/ai-computing/hetero-llm-d) | llm-d integration |

## 🗄 Earlier PoCs

Earlier proof-of-concept work was removed from the tree after release **v4.0.0**. Each folder below is preserved there
unchanged:

| Folder | Content |
| --- | --- |
| [`compiler_fx`](https://github.com/ai-computing/aicomp/tree/v4.0.0/compiler_fx) | FX-IR pipeline and 2D/3D parallelism PoCs, the predecessor of OptimusPrime; also the layer profiler used by `fasop/make_profiledb_legacy` |
| [`torchgpipe_OOO_PP`](https://github.com/ai-computing/aicomp/tree/v4.0.0/torchgpipe_OOO_PP) | Out-of-order backprop ([EuroSys '22](https://dl.acm.org/doi/pdf/10.1145/3492321.3519563)) applied to torchgpipe pipeline training |
| [`gpt-neox`](https://github.com/ai-computing/aicomp/tree/v4.0.0/gpt-neox) | GPT-NeoX training setup |
| [`mlperf`](https://github.com/ai-computing/aicomp/tree/v4.0.0/mlperf) | MLPerf BERT benchmark how-to for the ETRI platform |
| [`k8s_kubeflow_install`](https://github.com/ai-computing/aicomp/tree/v4.0.0/k8s_kubeflow_install) | Kubernetes / Kubeflow installation scripts |
| [`llama3_inference`](https://github.com/ai-computing/aicomp/tree/v4.0.0/llama3_inference) | Llama 3 8B inference examples |

## 📄 License

The results of the AIcomp project are distributed under the [3-clause BSD license](LICENSE).
