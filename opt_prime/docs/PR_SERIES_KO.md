# PR 시리즈: 이종 가속기 offload · PD · 학습 (main 병합용)

2026-10-19. 개발은 통합 브랜치 `feat/optimus-moe-offload`에서 했고, main에는 아래 순서의 기능 PR로 나눠 올린다.
각 PR 브랜치는 **혼자서 테스트가 통과**하고, 모든 PR은 기존 API를 바꾸지 않는다(새 인자·새 모듈만 추가).

2026-10-20 갱신. 절차는 조직 기본 가이드(ai-computing/.github `CONTRIBUTING.md`, GitHub flow)를 따른다:
Issue(문제·근거·범위·완료 조건·담당) → `<type>/<issue#>-<slug>` 브랜치 → PR 템플릿(배경·변경 내용·검증·호환성과 제약·관련
Issue) → 리뷰 + CI → Squash and merge → 브랜치 삭제. 상위 Issue는 #59. 쌓인(stacked) PR은 앞 브랜치를 base로 하고, 앞 PR이
병합되면 다음 PR의 base를 main으로 바꾸고 Draft를 해제한다. Merge는 aicomp 관리자가 한다.

| # | Issue | PR | 브랜치 | base | 내용 | 테스트 (5.1.0 / 5.8.1, 브랜치 단독) |
|---|---|---|---|---|---|---|
| 1 | #58 | #57 | `feature/transformers5-ir` | main | IR export 경로의 transformers 5.x 대응 | 5 / 5 passed |
| 2 | #53 | #64 | `feat/53-devices` | 1 | 장치 탐색(rm-accel 연동), capability, 칩 lease | 13 / 13 passed |
| 3 | #60 | #65 | `feat/60-moe-offload` | 2 | MoE expert offload (tier, planner, dispatcher, 네이티브 커널 반올림 #54) | 109 / 106 passed |
| 4 | #61 | #66 | `feat/61-moe-training` | 3 | MoE 학습(모든 dtype), offload 상태 학습, 학습 probe | 115 / 113 passed |
| 5 | #62 | #67 | `feat/62-model-exec-train` | 4 | 블록 스펙, provider, 엔진 선택, 학습 백엔드, 계획 문서 | 151 / 149 passed |
| PD | #63 | #68 | `feat/63-pd-families` | `feat/optimus-llmd` | PD torch 엔진 MoE·hybrid 계열, 칩 lease(선택) | PD 52 OK + 9 passed (RNGD+ATOM 호스트) |

PR 1은 Issue 번호 규칙 도입 전에 열려 브랜치 이름을 유지한다(열린 PR의 head 브랜치 이름을 바꾸면 PR이 닫힌다).
CUDA를 가린 상태의 단위 테스트 수이고, GPU·NPU 실측은 각 PR의 검증 절에 있다.

외부 라이브러리(선택 의존): `npu-kernels` — ai-computing/npu-offload 하위 프로젝트(Issue #6, PR #7), PR 3의 `rngd-opt` tier가
사용. `npu-lora` — ai-computing/npu-train(계열 확장 Issue #8, PR #9; MoE는 #6), PR 5의 학습 백엔드가 사용. 없으면 해당 백엔드만
빠진다.

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
