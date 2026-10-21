<div align="center">

<h1>AIcomp</h1>

<h3>GPU 클러스터·NPU·호스트 메모리에 걸친 대규모 모델 자동 병렬화</h3>

<p>
  <a href="README.md">English</a> · <strong>한국어</strong>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L3%20unified%20engine-34d399" alt="ai-rm 계층: L3 통합 엔진"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10 이상">
  <img src="https://img.shields.io/badge/PyTorch-2.5.1%20%7C%202.13-ee4c2c?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.5.1, 2.13">
  <img src="https://img.shields.io/badge/transformers-4.46.2%20%7C%205.x-ffcc4d" alt="transformers 4.46.2, 5.x">
  <a href="https://dl.acm.org/doi/10.1145/3625549.3658687"><img src="https://img.shields.io/badge/FASOP-HPDC%20'24-8b5cf6" alt="FASOP: HPDC 2024"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-64748b" alt="라이선스: BSD-3-Clause"></a>
</p>

<p>
  <a href="#-주요-기능"><b>주요 기능</b></a> ·
  <a href="#-저장소-구조"><b>구조</b></a> ·
  <a href="#-빠른-시작"><b>빠른 시작</b></a> ·
  <a href="#-호환성"><b>호환성</b></a> ·
  <a href="#-관련-저장소"><b>관련 저장소</b></a> ·
  <a href="#-이전-poc"><b>이전 PoC</b></a>
</p>

</div>

AIcomp는 ETRI의 고효율 AI 컴퓨팅 SW 핵심기술 과제입니다. 특정 고가 하이퍼클러스터에 의존하지 않고 대규모 모델을 학습하고
서빙하는 것이 목표입니다. 이 저장소에는 두 프로젝트가 있습니다.

- **OptimusPrime** (`opt_prime`): 병렬화 프레임워크
- **FASOP** (`fasop`): 병렬화 전략 탐색기

## ✨ 주요 기능

| | **OptimusPrime** · [`opt_prime`](opt_prime) | **FASOP** · [`fasop`](fasop) |
| --- | --- | --- |
| **무엇** | Hugging Face 모델과 `nn.Module`을 수정 없이 병렬화하는 학습·추론 프레임워크 | 이기종 GPU 클러스터에서 TP / PP / DP 전략을 찾는 비용 기반 최적화기 |
| **방법** | 모델을 PyTorch FX 그래프로 추출(HFTracer, 또는 `--dynamo-capture`로 TorchDynamo)해 파이프라인 단계로 나누고, 호스트 안과 호스트 사이에서 PP + DP + TP로 실행 | FX 노드 단위 비용을 프로파일링해 단계·통신 시간을 모델링하고, 장치 배치와 분할을 탐색해 Pareto front를 출력 |
| **학습** | GPipe / 1F1B, LoRA, CPU optimizer offload, HF 호환 체크포인트 | |
| **추론** | prefill/decode API, KV 캐시 모드 3가지, PP / TP, MPS 공유 | 추론 계획은 향후 과제 |
| **상태** | 유지 관리 중 | 유지 관리 중 · [논문](https://dl.acm.org/doi/10.1145/3625549.3658687) |

**이기종 가속기 기능 `opt_prime.accel`** (실험적, transformers 5.x 전용). 모두 선택 기능이며 요청하지 않으면 꺼져 있습니다.
- **MoE expert offload**: hot expert는 GPU·RNGD·ATOM 계층에서, cold expert는 호스트 CPU에서 계산합니다. 그래서 장치
  메모리보다 큰 MoE 모델도 실행할 수 있습니다. MoE 학습에도 같은 분할 방식을 씁니다.
- **엔진 선택**: OptimusPrime 자체 GPU 엔진과 NPU 네이티브 엔진(RNGD layer 엔진, ATOM `optimum-rbln`) 중에서 고릅니다.
- **Prefill/decode 분리**: RNGD에서 prefill하고, 장치 간 KV 전송을 거쳐 ATOM에서 decode합니다. 서빙용 llm-d sidecar를
  포함합니다.

자세한 내용: [opt_prime/README.md](opt_prime/README.md), [PR 시리즈 개요](opt_prime/docs/PR_SERIES_KO.md)

## 📦 저장소 구조

```
aicomp/
├── opt_prime/            OptimusPrime
│   ├── opt_prime/        프레임워크: IR 추출, 분할, 스케줄, 런타임, 추론
│   │   └── accel/        이기종 가속기 (MoE offload, 엔진, prefill/decode, 서빙)
│   ├── examples/         학습·추론 스크립트 (pp_train_gpt2.py, pp_train_llama5.py, ...)
│   ├── tests/
│   └── docs/
└── fasop/                FASOP: 탐색, 비용 모델, 프로파일러(make_profiledb), 데모 뷰어
```

## 🚀 빠른 시작

```bash
git clone https://github.com/ai-computing/aicomp.git
pip3 install "transformers==4.46.2" datasets    # PyTorch + CUDA 설치 후; 호환성 참고
cd aicomp/opt_prime/examples

# GPT-2, GPU 2장 파이프라인 병렬 (HFTracer 추출)
torchrun --nproc_per_node=2 --master_port=29500 pp_train_gpt2.py

# 같은 예제를 TorchDynamo 추출로 (transformers 5.x에서는 필수)
torchrun --nproc_per_node=2 --master_port=29500 pp_train_gpt2.py --dynamo-capture

# Llama 3.2 1B, PP=2 · TP=2 · DP=2, GPU 8장 (gated 모델: Hugging Face 토큰 필요)
torchrun --nproc_per_node=8 --master_port=29500 pp_train_llama5.py <hf token>
```

- **멀티 노드 실행, 병렬화 옵션, 분할기, 추론, LoRA, 체크포인트**: [opt_prime/README.md](opt_prime/README.md)
- **FASOP으로 전략 탐색**: [fasop/README.md](fasop/README.md)

## 🧪 호환성

변경을 머지하기 전에 예제가 아래 두 구성에서 모두 동작해야 합니다.

| 추출 방식 | PyTorch | transformers | 비고 |
| --- | --- | --- | --- |
| HFTracer (기본) | 2.5.1 | 4.46.2 | 예제의 기준 환경 |
| `--dynamo-capture` | 2.13.0 | 5.15.0 | `opt_prime.accel`에 필요 |

회귀 검사에서는 이 밖에 transformers 5.1.0과 5.8.1도 확인합니다.

`opt_prime.accel`은 transformers 5.x가 필요합니다. 하드웨어 경로는 [opt_prime/docs](opt_prime/docs)에 적힌 장치와 SDK 버전에서만
검증되었습니다.

## 🧩 관련 저장소

AIcomp는 조직의 이기종 AI 인프라 자원 관리 체계인 [**ai-rm**](https://github.com/ai-computing/ai-rm)에서 엔진 계층을 맡습니다.
OptimusPrime은 따로 설치하는 아래 라이브러리들을 선택적으로 사용합니다.

| 영역 | 저장소 | 쓰는 곳 |
| --- | --- | --- |
| 장치 탐색·lease | [rm-accel](https://github.com/ai-computing/rm-accel) | `accel.platforms` |
| NPU 학습 | [npu-train](https://github.com/ai-computing/npu-train) | `accel.train`: RNGD·ATOM LoRA / 전체 파인튜닝 |
| 장치 간 전송 | [hetero-npu-p2p](https://github.com/ai-computing/hetero-npu-p2p) | `accel.pd.npu`: RNGD → ATOM KV 전송 |
| NPU 커널·offload | [npu-offload](https://github.com/ai-computing/npu-offload) | RNGD expert 커널 (`npu-kernels`) |
| RNGD prefill/decode | [rngd-pd](https://github.com/ai-computing/rngd-pd) | KONI RNGD 엔드포인트 |
| 클러스터 서빙 | [hetero-llm-d](https://github.com/ai-computing/hetero-llm-d) | llm-d 연동 |

## 🗄 이전 PoC

이전 PoC들은 릴리스 **v4.0.0** 이후 트리에서 제거했습니다. 아래 폴더는 모두 v4.0.0에 그대로 남아 있습니다.

| 폴더 | 내용 |
| --- | --- |
| [`compiler_fx`](https://github.com/ai-computing/aicomp/tree/v4.0.0/compiler_fx) | FX IR 기반 파이프라인과 2D/3D 병렬화 PoC. OptimusPrime의 전신이며, `fasop/make_profiledb_legacy`가 쓰는 레이어 프로파일러도 있음 |
| [`torchgpipe_OOO_PP`](https://github.com/ai-computing/aicomp/tree/v4.0.0/torchgpipe_OOO_PP) | Out-of-order backprop([EuroSys '22](https://dl.acm.org/doi/pdf/10.1145/3492321.3519563))을 torchgpipe 파이프라인 학습에 적용 |
| [`gpt-neox`](https://github.com/ai-computing/aicomp/tree/v4.0.0/gpt-neox) | GPT-NeoX 학습 환경 |
| [`mlperf`](https://github.com/ai-computing/aicomp/tree/v4.0.0/mlperf) | ETRI 플랫폼용 MLPerf BERT 벤치마크 실행 방법 |
| [`k8s_kubeflow_install`](https://github.com/ai-computing/aicomp/tree/v4.0.0/k8s_kubeflow_install) | Kubernetes / Kubeflow 설치 스크립트 |
| [`llama3_inference`](https://github.com/ai-computing/aicomp/tree/v4.0.0/llama3_inference) | Llama 3 8B 추론 예제 |

## 📄 라이선스

AIcomp 과제의 결과물은 [3-clause BSD 라이선스](LICENSE)로 배포합니다.
