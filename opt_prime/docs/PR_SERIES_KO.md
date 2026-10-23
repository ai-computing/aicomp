# PR 시리즈: opt_prime.accel (이종 가속기 offload · 학습 · 엔진 · PD)

2026-10-20. 절차는 조직 기본 가이드(ai-computing/.github `CONTRIBUTING.md`, GitHub flow)를 따른다. 상위 Issue #59,
구조 Issue #71, PD 병합 계획 #69. 모든 PR은 **`optimus-rm`**(main 62a7831에서 만든 통합 브랜치)으로 모으고 main은 아직 바꾸지 않는다. 새 코드는 모두 `opt_prime/accel/` 아래에 있고, 기존 엔진 파일은 선택 인자(`moe_offload=`)와
export 수정만 바뀐다. 각 PR 브랜치는 **혼자서 테스트가 통과**하고(transformers 5.1.0 / 5.8.1, CUDA 숨김), 쌓인 PR은 앞 PR이
`optimus-rm`에 병합되면 base를 `optimus-rm`으로 바꾸고 Draft를 해제한다. 리뷰와 병합은 @gspark-etri가 한다.

PR 전 최소 조건: `examples/pp_train_gpt2.py`, `examples/pp_train_llama5.py`가 HFTracer(PyTorch 2.5.1 + Transformers 4.46.2)와
`--dynamo-capture`(PyTorch 2.13.0 + Transformers 5.15.0) 양쪽에서 동작.

## 구조

```
opt_prime/accel/
  spec/       모델 기술 하나: ModelSpec, ExpertSpec, 계열 어댑터, expert 탐색
  platforms/  장치·lease(rm-accel 연동), 벤더별 구현: cuda/ cpu/(native C) rngd/ atom/
  moe/        expert offload(planner, 레이어 실행, custom op), moe/train.py
  engines/    엔진 선택 + Optimus 엔진 (벤더 엔진은 platforms/<벤더>/engine.py)
  pd/         PD 계약·런타임·torch 엔진, pd/npu: RNGD→ATOM 파이프라인
  train/      TrainJob, 학습 백엔드 (Optimus_p / npu-train)
  serve/      llm-d sidecar
```
의존 방향: `spec ← platforms ← moe / engines / pd / train ← serve`. 기능 패키지는 벤더 SDK를 import하지 않는다.

## 순서

```
#57 ─ A1 ─┬─ A2 ─ A3 ─ A4
          └─ P1 ─┬─ P2 ─ P4
                 └─ P3
P0, #82 (optimus-rm 기준, 독립)
```

| PR | 브랜치 | 내용 | 바뀌는 기존 파일 | 테스트 (5.1 / 5.8 / 5.15, 누적) |
|---|---|---|---|---|
| #57 | `feature/transformers5-ir` | transformers 5 export 수정 | `IR.py`, README(설명) | 5 / 5 / 5 |
| A1 | `feat/71-accel-spec-platforms` | spec, platforms 핵심(탐색·lease·capability, 벤더별 device) | 없음 | 43 / 44 / 43 |
| A2 | `feat/60-accel-moe-offload` | moe offload, 벤더별 expert tier, native CPU 커널 | `inference.py`, `opti_pri.py`(`moe_offload=None`), README | 142 / 143 / 141 |
| A3 | `feat/61-accel-moe-train` | MoE 학습(expert 구현 교체, 한 expert씩 역전파) | `opti_pri.py`(MoE 모델에서만 hook) | 151 / 153 / 151 |
| A4 | `feat/62-accel-engines-train` | 엔진 선택(Optimus, RNGD layer, ATOM rbln), 학습 백엔드 | 없음 | 158 / 160 / 157 |
| P0 | `chore/69-opt-prime-packaging` | pyproject, LICENSE, .gitignore | 없음 | - |
| P1 | `feat/69-accel-pd-core` | PD 계약·런타임·torch 엔진(dense·MoE·hybrid) | `__init__.py`(지연 export), README | 69 / 72 / 71 |
| P2 | `feat/69-accel-pd-npu` | RNGD prefill → ATOM decode, P2P worker·factory | 없음 | 83 / 86 / 85 |
| P3 | `feat/69-accel-serve-llmd` | llm-d sidecar, HTTP 취소 | 없음 | 79 / 82 / 81 |
| P4 | `exp/69-atom-native-kv-resume` | ATOM native KV resume (실험적) | 없음 | 82 / 84 |

외부 패키지(`hetero_p2p`, `rngd_pd`, `npu_kernels`, `npu_lora`, rebel/furiosa SDK)가 없으면 해당 테스트는 건너뛴다.
PD 검증 원본(로그·JSON·소스 사본)은 main에 넣지 않고 요약(`docs/validation/**/RESULTS.md`)만 둔다(#69).

## 기존 동작 회귀 검증

기존 파일을 바꾸는 PR(#57, A2, A3, P1)에는 main 대비 회귀 결과를 첨부한다: 예제가 쓰는 13개 계열(gpt2, gpt2 분류, llama,
llama+LoRA, opt, bert, electra, gpt-neo, gptj, whisper, vit, qwen2-moe, 사용자 정의 모듈)과 Optimus_Inference 생성을,
transformers 4.46.2(HFTracer, dynamo) / 5.1.0 / 5.8.1 / 5.15.0(torch 2.13.0)에서 같은 seed로 돌려 loss·토큰을 비트 단위로 비교.

## 검증 요약 (실측)

- offload: Qwen3-30B-A3B exact 15.4 tok/s(compile 18.4), Solar-Open-100B exact 8.6–10.2, gemma-4-26B fp32 logits 9.2e-5
- 엔진: Solar-Open-100B → RNGD layer 엔진 29.4 tok/s decode; ATOM rbln: Llama-3.2-1B 85 tok/s(토큰 HF와 동일)
- 학습(GPU, expert offload): Qwen3-30B-A3B 3.21→0.38, gemma-4-26B 6.42→0.72, Solar-Open-100B 3.53→0.60
- 학습(RNGD, npu-train): Llama-3.2-1B LoRA·전체 FT, Qwen2.5-0.5B, Qwen3-0.6B, gemma-3-1b LoRA
- PD: dense + Qwen3-MoE·Solar·Gemma3·Gemma4 parity, RNGD→ATOM P2P(KONI, Qwen2.5 0.5B/1.5B/3B)
