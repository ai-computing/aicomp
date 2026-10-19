# Solar / Qwen / Llama / Gemma × offload · PD · LoRA · fine-tuning 계획

## 0. 현황 체크리스트 (2026-10-19 기준, 브랜치 feat/optimus-moe-offload)

범례: [x] 완료·검증, [~] 부분 완료(조건 표기), [ ] 남음. "tiny" = 실제 config 축소 무작위 모델, "실모델" = 체크포인트.

### 모델 × 기능

| 계열 (대표 체크포인트) | MoE offload | PD 분리 (torch 엔진) | LoRA 학습 | fine-tuning |
|---|---|---|---|---|
| Solar (Solar-Open-100B) | [x] 실모델 exact, RNGD×2+CPU 8.6 tok/s · PoC 엔진 29.4 tok/s | [x] tiny parity | [x] GPU tiny fp32/bf16 · [ ] RNGD | [x] GPU tiny · [ ] RNGD |
| Qwen (Qwen3-30B-A3B, Qwen2.5) | [x] 실모델 exact, 최고 18.4 tok/s | [x] dense·MoE tiny parity | [x] GPU tiny · [ ] RNGD | [x] GPU tiny · [ ] RNGD |
| Llama (Llama-3.2-1B / 3.1-8B) | 해당 없음(dense) · [ ] Llama 4 MoE 어댑터 | [x] tiny parity | [x] GPU 실모델 1B · [~] RNGD 1B 실행 중 | [x] GPU tiny · [ ] RNGD |
| Gemma (gemma-4-26B-A4B) | [x] 실모델 fp32 exact(9e-5) · bf16은 모델 자체가 장치별로 갈림 | [x] Gemma3·4 tiny parity | [x] GPU tiny fp32/bf16 · [ ] RNGD | [x] GPU tiny · [ ] RNGD |

### 장치별 지원

- [x] GPU: expert tier, 전체 실행(Optimus + torch.compile), 학습(Optimus_p)
- [x] RNGD: expert tier(rngd, rngd-opt 실칩), 전체 실행(PoC 엔진, Solar 형상), 학습(npu-lora, Llama)
- [x] ATOM: expert tier(atom, 실칩) · [ ] dense/전체 실행 엔진 · [ ] 커스텀 커널(SDK 0.11 미지원) · 학습 범위 밖
- [x] CPU: expert tier(native 커널), 학습 기준 백엔드(npu-lora cpu)

### 구조 (Optimus = 컨트롤 플레인)

- [x] devices: 탐색, capability, 칩 lease(RNGD pe_occupancy, ATOM rbln-smi 확인)
- [x] model: 블록 스펙(MoE 20개 패밀리 + gemma4), KV 의미 스펙
- [x] kernels: (블록, 장치) provider registry
- [x] exec: 엔진 선택(optimus / rngd-layer)
- [x] train: TrainJob/TrainBackend(optimus / npu-lora)
- [x] moe_offload: N-tier planner, 네이티브 dispatcher, 적응형 재배치, offload backward 없이 학습 경로(opt_batched_mm)
- [x] PD: 칩 lease 연동, MoE·hybrid 계열 torch 엔진

### 라이브러리 (데이터 플레인)

- [x] hetero-npu-p2p (기존, PD 팀)
- [~] npu-kernels: 로컬 저장소, RNGD expert/GEMV 생성기 이전 완료 · [ ] GitHub 위치 결정·push
- [~] npu-lora → npu-train: Optimus 어댑터 완료(Llama) · [ ] Qwen/Gemma/Solar 함수형 모델 · [ ] 이름·저장소 정리

### 성능

- [x] Qwen3-30B: 10.6 → 15.4(dispatcher) → 18.4 tok/s(compile)
- [x] Solar: 공통 진입점 → PoC 엔진 29.4 tok/s decode
- [x] CUDA graph 시도(효과 없음, 기록)
- [ ] RNGD layer 엔진 형상 템플릿화(3단계: GEMV 생성기 완료·실칩 724 GB/s, attention·norm·router 남음)

### PR

- [x] 계획(아래 3절)
- [ ] main 기준 PR 브랜치 분할·push
- [ ] PR 생성(공개 저장소: 생성 전 확인)

### 남은 문제

- [ ] Llama 4 MoE: 입력 쪽 라우팅 가중(sigmoid top-1) 어댑터
- [ ] RNGD 학습 계열 확장(Qwen q/k norm, MoE, Gemma sliding)
- [ ] ATOM dense 엔진(RBLN 그래프 ~60 µs/호출) ↔ PD decode 엔진 통합
- [ ] 네이티브 CPU expert 커널 반올림 차이(동률 토큰 갈림)


2026-10-19. 목표: 네 모델 계열에서 MoE offload, PD 분리, LoRA 학습, 전체 fine-tuning이 동작하도록 하고, main에 기능
단위 PR로 합칠 수 있게 브랜치를 나눈다.

## 1. 대상 체크포인트와 기능 정의

| 계열 | 대표 체크포인트 (보유) | 구조 | 비고 |
|---|---|---|---|
| Solar | upstage/Solar-Open-100B | MoE 128×top8, shared expert, sigmoid+group 라우터 | 205 GB |
| Qwen | Qwen/Qwen3-30B-A3B (MoE), Qwen/Qwen2.5-0.5B (dense) | MoE softmax 라우터, q/k norm | Qwen3.6-35B-A3B는 transformers 상향 필요 |
| Llama | meta-llama/Llama-3.2-1B, Llama-3.1-8B | dense | Llama 4(MoE)는 체크포인트 없음 |
| Gemma | google/gemma-4-26B-A4B-it | MoE (gemma4) | transformers 5.1.0이 gemma4 미지원 → 상향 필요 |

| 기능 | 정의 | dense 모델에서 |
|---|---|---|
| offload | MoE expert를 GPU/RNGD/ATOM/CPU tier에 나눠 실행 (moe_offload) | 해당 없음(expert 없음) — 모델이 장치에 들어가지 않을 때만 의미; 범위 밖으로 둠 |
| PD 분리 | prefill/decode를 다른 엔진에서, KV 전달 | 동일 |
| LoRA 학습 | Optimus_p + opt_prime.lora, base 고정 | 동일 |
| fine-tuning | Optimus_p 전체 파라미터 학습 | 동일 |

MoE + offload + 학습: expert를 offload한 채 LoRA 학습을 하려면 offload op에 backward가 필요하다(expert는 고정,
입력에 대한 gradient만). host tier에서 torch로 계산한다.

## 2. 검증 기준

- 정확도: offload/PD는 HF generate 대비 토큰 일치(동률 판정 포함), 학습은 첫 step loss가 HF와 같고(fp32 tiny 1e-5)
  N step 후 loss 감소, LoRA는 base 가중치 불변.
- 각 계열 tiny 모델(실제 config 축소)로 CI 테스트, 대표 체크포인트로 실측 1회.
- PD는 GPU/CPU torch 엔진으로 네 계열 모두; RNGD→ATOM NPU 경로는 PD 팀의 모델별 엔진(현재 Qwen2/KONI-Llama).

## 3. 브랜치와 PR

main 관례: `feature/<name>` 브랜치 하나에 기능 하나, PR 제목 "Feature/<name>". 개발은 통합 브랜치
(`feat/optimus-moe-offload`)에서 하고, 기능이 검증되면 main 기준 PR 브랜치로 경로 단위로 나눠 낸다(모듈이 디렉터리로
나뉘어 있어 충돌이 적다). 각 PR 브랜치는 단독으로 테스트가 통과해야 한다.

| 순서 | PR 브랜치 | 내용 | 선행 |
|---|---|---|---|
| 1 | feature/transformers5-ir | IR export: transformers ≥ 5 (deferred asserts, DynamicCache, assert 제거), 버전 명시 | - |
| 2 | feature/devices | opt_prime.devices (탐색, capability, 칩 lease) | - |
| 3 | feature/model-spec | opt_prime.model (블록 스펙, KV 의미) | - |
| 4 | feature/moe-offload | moe_offload 코어 (gpu/cpu tier, planner, layer, dispatcher, 예제) | 1, 2, 3 |
| 5 | feature/moe-offload-npu | rngd / rngd-opt (+커널 생성기) / atom 백엔드 | 4 |
| 6 | feature/exec-engines | opt_prime.kernels, opt_prime.exec, RNGD layer 엔진 어댑터 | 3, 5 |
| 7 | feature/model-coverage | 네 계열 학습(LoRA/FT)·offload·PD 테스트와 수정, offload backward, transformers 상향 | 1, 4 |
| PD | (PD 팀 분할) | feat/optimus-llmd의 P2P/PD — 칩 lease 연동만 6 이후 | 2 |

PR마다: 변경 요약, 영향 범위(기존 API 불변), 테스트 명령과 결과, 실측 표, 알려진 한계.

## 4. 진행 순서

1. 네 계열 tiny 학습 probe(FT, LoRA) — transformers 5.1.0에서 현재 상태 파악, 실패 수정 (PR 7)
2. transformers 상향 venv에서 gemma4 / qwen3.6 지원 (PR 7)
3. offload op backward → MoE LoRA 학습 with offload (PR 4, 7)
4. PD torch 엔진 계열 확장 (llama/qwen/gemma/solar, MoE 포함) + parity 테스트 (PD 팀과 조율)
5. 대표 체크포인트 실측, 문서
6. PR 브랜치 분할과 PR 작성 (main 기준, 순서대로)
