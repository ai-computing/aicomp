# PR 시리즈: 이종 가속기 offload · PD · 학습 (main 병합용)

2026-10-19. 개발은 통합 브랜치 `feat/optimus-moe-offload`에서 했고, main에는 아래 순서의 기능 PR로 나눠 올린다.
각 PR 브랜치는 앞 PR 위에 쌓여 있고(stacked), **혼자서 테스트가 통과**한다. 앞 PR이 병합되면 다음 PR의 base를 main으로
바꾼다. 모든 PR은 기존 API를 바꾸지 않는다(새 인자·새 모듈만 추가).

| # | 브랜치 (원격) | base | 내용 | 테스트 (transformers 5.1.0 / 5.8.1) |
|---|---|---|---|---|
| 1 | `feature/transformers5-ir` | main | IR export 경로의 transformers 5.x 대응 | 5 / 5 passed |
| 2 | `feature/devices` | 1 | 장치 탐색, capability, 칩 lease | 6 passed, 2 skipped(다음 PR 의존) |
| 3 | `feature/moe-offload` | 2 | MoE expert offload (GPU/CPU/RNGD/ATOM tier, planner, dispatcher) | 97 / 94 passed |
| 4 | `feature/moe-training` | 3 | MoE 학습(모든 dtype), 학습 probe | 102 / 100 passed |
| 5 | `feature/model-exec-train` | 4 | 블록 스펙, provider, 엔진 선택, 학습 백엔드, 계획 문서 | 137 / 134 passed |
| PD | `feature/pd-families-leases` | `feat/optimus-llmd` | PD torch 엔진 MoE·hybrid 계열, 칩 lease(선택) | PD 52 OK + 9 passed (etri-001) |

외부 라이브러리(별도 저장소, 위치 결정 후 push): `npu-kernels`(RNGD 커널 생성기, PR 3의 `rngd-opt`가 선택적으로 사용),
`npu-lora`(RNGD 학습, PR 5의 `npu-lora` 학습 백엔드가 선택적으로 사용). 둘 다 없으면 해당 백엔드만 제공되지 않는다.

## PR별 리뷰 포인트

1. **transformers5-ir** — `IR.py`만 수정. transformers 4.x 경로(HFTracer)는 변경 없음. 확인: deferred runtime assert,
   assert 노드 제거 범위(값 계산 없는 노드만), 여러 번 호출되는 파라미터 없는 leaf 모듈 inline, 빈 출력 region.
2. **devices** — 새 패키지, 다른 코드에서 아직 사용하지 않음. 확인: sysfs만 읽고 장치를 열지 않음, flock lease의 프로세스
   간 배타성, RNGD `pe_occupancy`/ATOM `rbln-smi` busy probe.
3. **moe-offload** — `Optimus_Inference(moe_offload=None)` 기본값이면 기존 동작 그대로. 확인: custom op이 export에 불투명,
   planner 비용 모델, native C 코드(gcc로 첫 사용 시 빌드), NPU 백엔드는 SDK를 지연 import.
4. **moe-training** — `Optimus_p`가 MoE 모델에서만 expert 구현을 `opt_batched_mm`으로 바꿈(dense 모델 무관). 확인: eager
   대비 출력·gradient 일치 테스트.
5. **model-exec-train** — 새 패키지 4개(`model`, `kernels`, `exec`, `train`). 기존 실행 경로에는 끼어들지 않음.
PD. PD 팀 브랜치에 대한 PR. 기존 dense 계열 동작 불변, 새 계열은 parity 테스트로 추가.

## 검증 요약 (통합 브랜치 실측)

- offload: Qwen3-30B-A3B exact 15.4 tok/s(compile 18.4), Solar-Open-100B exact 8.6–10.2, Gemma-4-26B fp32 logits 9.2e-5
- 엔진 선택: Solar-Open-100B → RNGD layer 엔진 29.4 tok/s decode
- 학습(GPU): Llama·Qwen2/3·Gemma3·Qwen3-MoE·Solar·Gemma4 tiny FT/LoRA PASS(pp=1,2), Llama-3.2-1B LoRA loss 3.11→1.26; 실모델 expert offload LoRA — Qwen3-30B-A3B 3.21→0.38, gemma-4-26B-A4B 6.42→0.72 (GPU 20 GB + CPU)
- 학습(RNGD): Optimus train() → npu-lora, Llama-3.2-1B LoRA loss 3.75→0.59/30 step (첫 컴파일 2.2 h 후 캐시), Qwen2.5-0.5B LoRA 2.41→0.49, Llama-3.2-1B 전체 FT 3.75→0.34 (HF 로드·생성 확인)
- PD: dense + Qwen3-MoE·Solar·Gemma3·Gemma4 parity
