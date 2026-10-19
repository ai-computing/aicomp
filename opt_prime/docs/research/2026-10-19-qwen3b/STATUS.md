# STATUS — PD Disaggregation

<!-- research-hub:header:begin (maintained by .research/report.py; keep the "- **Key**: value" format) -->
- **State**: active
- **Updated**: 2026-10-19 05:08+00:00
- **Agent**: codex@etri-001
- **Task**: PDDISAG-6
- **Last result**: Qwen3B RNGD→ATOM P2P16요청320토큰 및 llm-d8요청160토큰 전송 exact·정상종료. FP32 teacher-forced147/160으로 일반 품질 미확정(한국어27/32). 장치 선택 일반화·30시험 통과. GPU PCI 미검출.
- **Next**: 3B 오차를 prefill/decode 단계별로 분리하고 동일 정밀도 단독 기준과 비교; GPU는 사용 가능한 장치 확보 후 검증
- **Blockers**: none
<!-- research-hub:header:end -->

## Goal

Furiosa RNGD와 Rebellions ATOM에서 LLM의 prefill/decode를 분리하고, 실제 KV cache 전달과 llm-d 연계를 통해 이종 NPU 서빙을 구현·검증한다. 현재 중심은 KONI-Llama3.1-8B의 RNGD→ATOM P2P DMA, 배치 및 요청 스케줄링이며, 단독 실행과 비교해 처리량·지연·전력효율을 개선할 수 있는 조건과 한계를 밝힌다. 전송 정합성, 모델 품질, 동일 자원 예산의 성능을 구분해 평가한다. 근거: [작업 이력](WORKSPACE_HISTORY.md), [현재 라이브러리](hetero-npu-p2p/README.md), [최신 검증 범위](hetero-npu-p2p/docs/ITERATION_REPLICAS_KO.md).

## Findings

아래 수치는 기존 기록을 정리한 것이며 이번 bootstrap에서 재측정하지 않았다. 성능은 각 문서의 표시 정밀도를 유지했다. 워크스페이스 루트는 Git 저장소가 아니며, 발견한 하위 저장소 7개의 전체 ref 대상 로그·브랜치·worktree를 조회했다. 별도 worktree는 없었다. 주요 연구 이력은 `hetero-npu-p2p/main`의 `44c3b4d`와 `rngd_llmd/master`의 `42f3e78`까지 확인했다. 실행 당시 수정본·소스 해시는 각 run manifest를 기준으로 하며, 후속 커밋과 동일하다고 가정하지 않는다. 근거: 각 하위 저장소 `.git`, [최신 실행 소스 기록](hetero-npu-p2p/results/pd-iter-two-r3/source-manifest.json).

- **2026-10-16 — ATOM 2장에서는 sealed 배치가 현재 기본 정책이다.** RNGD 1장+ATOM2/3, KONI B8, 입력140/167/227/271, 각3회 중앙값에서 sealed/refill 처리량은 출력32 **136.35/136.49 tok/s**, 출력128 **177.73/176.66 tok/s**, 혼합32요청 **147.01/117.31 tok/s**였다. 혼합 출력32 응답시간은 **3.77→5.04초**로 악화했다. 최종424요청 full-logit exact, 소유권 위반0, 진행 중 실제 refill50건이며 초기424요청을 포함해848요청을 보존했다. 초기 배치 수집 누락으로 생긴 **94.39→173.70 tok/s** 차이는 대표 개선치에서 제외한다. 3회 결과로 일반적인 우열을 단정하지 않는다. 근거: `hetero-npu-p2p` 커밋 `44c3b4d`, [결과](hetero-npu-p2p/docs/ITERATION_REPLICAS_KO.md), [검증 JSON](hetero-npu-p2p/docs/iteration-replicas-validation.json), [원시 run](hetero-npu-p2p/results/pd-iter-two-r3/analysis.json).

- **2026-10-16 — ATOM 1장의 적응형 quantum은 시간차 도착에서 유리했다.** B8의 full128/partial4 정책은 fixed128 대비 시나리오 처리량 **39.51→56.89 tok/s**, 출력32 응답시간 **10.71→4.77초**, 먼저 시작한 출력128 응답시간 **10.04→10.69초**였다. 포화 처리량은 **94.12→94.08 tok/s**였다. 304요청 전체 logits 및 lease 검증 통과, 소유권 위반0. 단일 제어 도착 패턴의 결과이며 후속 초기 배치 수집 변경 뒤 모든 부하에 그대로 적용할 수 없다. 근거: 커밋 `a8be600`, [결과](hetero-npu-p2p/docs/ADAPTIVE_QUANTUM_KO.md), [검증 JSON](hetero-npu-p2p/docs/quantum-validation.json).

- **2026-10-16 — 동일 3장 구성에서 B8이 채워진 배치의 처리량을 개선했다.** llm-d HTTP 경로의 B4→B8 중앙값은 출력32 **85.66→134.83 tok/s**, 출력128 **100.52→176.50 tok/s**, 혼합 **93.10→127.47 tok/s**였다. 부분 배치3요청은 **35.80→33.78 tok/s**로 악화했다. 최종424요청 전체 logits exact, 요청마다64/64 DMA CQE 완료. 이는 PD 배치 폭 비교이며 단독 NPU 대비 개선이나 전력·MFU·SLO 달성 결과가 아니다. 근거: 커밋 `765aee9`, [조건·실패 기록](hetero-npu-p2p/docs/PD_B8_RESULTS_KO.md), [검증 JSON](hetero-npu-p2p/docs/pd-b8-validation.json).

- **2026-10-15 — 실제 한국어 질의에서는 처리량과 에너지 효율의 승자가 달랐다.** 입력140/167/227/271의4질의, 출력32/128에서 PD는 **89.654/98.285 tok/s**, RNGD 단독 **61.565/66.284 tok/s**, 제조사 ATOM 단독 **40.154/52.255 tok/s**였다. 에너지 효율은 PD **0.5527/0.6070 tok/J**, RNGD **0.4023/0.4326 tok/J**, ATOM **0.6255/0.8713 tok/J**였다. PD는3장, 단독은1장이며 보드 전력만 포함한다. FP32 teacher-forced top1 일치는 PD **505/512**, 각 단독 **508/512**로 전송 exact가 품질 동등성을 뜻하지 않는다. 근거: [성능](experiments/native_pd_2026_4/real_workload_20261015/RESULTS_KO.md), [품질](experiments/native_pd_2026_4/real_workload_20261015/QUALITY_KO.md).

- **2026-10-15 — 실제 이종 P2P DMA와 라이브러리·로컬 llm-d 연결을 검증했다.** 최초 RNGD npu1→ATOM3 **4096/16384 bytes** 복사에서 CQE0, 데이터 일치와 정상 종료를 확인했다. 성공 경로는 **ATOM 소유 버퍼 export→Furiosa import/PDMA write**이며 host staging fallback이 없다. 이후 분리한 라이브러리로 llm-d3요청+직접1요청, 요청당 **2MiB·64 DMA 명령·32 decode step**의 전체 logits 최대 절대 오차0을 확인했다. 반대 importer 경로의 커널 패닉 기록은 성공 경로와 구분한다. 근거: [DMA 성공 기록](experiments/native_pd_2026_4/p2p_dma/P2P_SUCCESS_20261015.md), 커밋 `441db9a`, [라이브러리 검증](hetero-npu-p2p/docs/VALIDATION.md). 기존 [P2P STATUS](experiments/native_pd_2026_4/p2p_dma/STATUS.md)의 미완료 표기는 이 후속 결과로 갱신해 해석해야 한다.

- **2026-10-14 — host-staged 경로의 KPI 한계와 로컬 llm-d 동작을 확인했다.** 초기8조건×2반복에서 전체 효율 **3.94–120.29 GFLOPS/W**, 전체 MFU의 수학적 상한 **0.149–4.496%**로 해당 조건의 목표341 GFLOPS/W·20%에 미달했다. 후속 direct-admit 검증3072요청 및 온라인·route810요청은 모두 참조와 일치했고, B24 2ATOM admission은 약 **171–173→84–86ms**, 시스템 효율은 **348–349→352–362 GFLOPS/W**로 기록됐다. 서로 다른 조건의 수치이며 최신 P2P/B8 측정과 합치지 않는다. ATOM BF16 peak 미확인으로 정확한 전체 BF16 MFU는 미산출이다. 근거: [초기 KPI](experiments/npu_pd_extended_20261014/priority_results/REPORT.md), [후속 상태와 검증 범위](experiments/native_pd_2026_4/llmd_integration/STATUS.md).

- **2026-08-25 — 컴파일 설정과 공정한 대조군이 비교 결과를 좌우했다.** RNGD B4/S256/32레이어에서 group2·무제한 shape의 **26.20ms/step** 대비 group8·shape8은 **17.62ms/step**, 게이트웨이 TPOT는 **27.00→20.10ms**였다. 앞선 공정 배치 비교에서는 D100–160/N8에서 solo1장 **106.6 tok/s**, PD2장 **143.7 tok/s**로 총처리량과 카드당 효율의 결론이 달랐다. 초기 prefix-cache 에뮬레이션 **534 tok/s**는 실제 KV 전송 성능이 아니다. 근거: `rngd_llmd` 커밋 `42f3e78`, `384310c`, [이력의 3.4·5.4·6.5절](WORKSPACE_HISTORY.md).

## Open questions

- **최우선:** 최신 B8/sealed 경로를 동일 카드 예산의 단독·복제 기준과 비교하고, 다양한 도착률·실제 trace에서 goodput, token interval p99, 전력효율을 함께 검증해야 한다. 최신 iteration 시험은 전력·SLO를 측정하지 않았다. 근거: [최신 남은 과제](hetero-npu-p2p/docs/ITERATION_REPLICAS_KO.md), [B8 비교 범위](hetero-npu-p2p/docs/PD_B8_RESULTS_KO.md).
- 두 replica의 remaining-work 기반 배치가 혼합 길이 손실을 줄일 수 있는가? 한 장의 adaptive refill 이득을 두 장 시간차 도착에도 유지할 수 있는가? B8 실행 메모리는 약 **15.0/15.7GiB**로 B16은 먼저 메모리 적합성 검증이 필요하다. 근거: [독립 replica 결과](hetero-npu-p2p/docs/ITERATION_REPLICAS_KO.md).
- 다른 입력 분포·긴 문맥·응용 정확도에서 단독 대비 품질 비열등성을 만족하는가? 현재 KONI/context512·고정 생성 길이·non-streaming 및 같은 배치 host logits 검증을 일반적인 모델 품질로 확대 해석할 수 없다. 근거: [품질 평가](experiments/native_pd_2026_4/real_workload_20261015/QUALITY_KO.md), [B8 조건](hetero-npu-p2p/docs/PD_B8_RESULTS_KO.md).
- 설치된 ATOM thunk ABI와 eager allocation 의존성을 넘어 지원 가능한 전송 계약은 무엇인가? Kubernetes/EPP/Envoy 전체 경로, 다중 노드, 일반 sampling·streaming, 취소/TTL 및 다른 모델 지원은 추가 검증이 필요하다. 로컬 official sidecar 성공을 클러스터 배포 완료로 해석하지 않는다. 근거: [라이브러리 범위](hetero-npu-p2p/README.md), [서빙 계약](hetero-npu-p2p/docs/SERVING_LIBRARY.md).
- ATOM BF16 peak 근거와 동일 구간의 에너지 측정을 확보해 전체 MFU 및 연차 KPI를 어떻게 확정할 것인가? lane 비율·MFU 상한·FP16 peak를 BF16 실측 MFU로 대체하지 않는다. 근거: [KPI 정의와 한계](experiments/npu_pd_extended_20261014/priority_results/REPORT.md).

## Log

<!-- research-hub:log (report.py inserts new entries below this line, newest first) -->

### 2026-10-19 05:08+00:00 — PDDISAG-6

- Result: Qwen3B RNGD→ATOM P2P16요청320토큰 및 llm-d8요청160토큰 전송 exact·정상종료. FP32 teacher-forced147/160으로 일반 품질 미확정(한국어27/32). 장치 선택 일반화·30시험 통과. GPU PCI 미검출.
- Next: 3B 오차를 prefill/decode 단계별로 분리하고 동일 정밀도 단독 기준과 비교; GPU는 사용 가능한 장치 확보 후 검증
- Params: batch=1, context=512
- Artifacts: experiments/qwen3b_pd_20261019/quality.json, experiments/qwen3b_pd_20261019/pd-llmd-r1/result.json
- Experiments: experiments/2026-10-19-qwen3b-extension.md
- By: codex@etri-001

### 2026-10-19 04:55+00:00 — PDDISAG-6

- Result: Qwen 0.5B/1.5B OptimusPrime llm-d 실제 NPU 경로 검증 완료: 직접32요청640토큰 + sidecar16요청320토큰, 출력 일치·정상 종료. 통합27시험 및 SDK 없는 새 venv wheel import 통과. 엔진 a2740f1, llm-d 병합 커밋 준비.
- Next: 같은 카드 예산 성능 비교 및 배치·긴 문맥 확장; GPU와 새 ABI는 별도 실측 필요
- Params: batch=1, context=512
- Artifacts: experiments/qwen_pd_extension_20261019/pd05-llmd-r1/result.json, experiments/qwen_pd_extension_20261019/pd15-llmd-r1/result.json
- Experiments: experiments/2026-10-19-qwen-pd-extension.md
- By: codex@etri-001

### 2026-10-19 02:34+00:00 — PDDISAG-6

- Result: OptimusPrime Qwen RNGD→ATOM 직접 P2P: 두 모델 합계32요청640토큰1664 DMA 완료, host-admission 전체 logits exact. FP32 teacher-forced top1 0.5B160/160, 1.5B158/160. 단위시험22개 통과. 모듈 경계와 제약 문서화; 새 코드는 아직 미커밋.
- Next: Qwen 확장 코드·증빙 커밋 후 llm-d feature에 통합하고 실제 sidecar 경로 검증; GPU 및 새 SDK별 검증은 별도
- Params: batch=1, context=512
- Artifacts: experiments/qwen_pd_extension_20261019/quality-pd05-r2.json, experiments/qwen_pd_extension_20261019/quality-pd15-r2.json
- Experiments: experiments/2026-10-19-qwen-pd-extension.md
- By: codex@etri-001

### 2026-10-19 02:17+00:00 — PDDISAG-6

- Result: Qwen 이종PD 확장 진행: 모델 geometry/QKV bias/tied weight 모듈 분리, RNGD0.5B prefix 실기 gate통과, ATOM1 외부KV graph 컴파일완료. 그룹변경/사용중 ATOM3는 제외
- Next: ATOM host-admission 수치검증 후 모델별4KiB정렬 P2P 연결,1.5B 확대
- Experiments: experiments/2026-10-19-qwen-pd-extension.md
- By: codex@etri-001

### 2026-10-19 01:49+00:00 — PDDISAG-6

- Result: 실제 NPU 모델검증/PD 계측 ca9b695 커밋 및 llm-d feature 6b599ca 통합. 측정144회·KONI24요청 보존, engine17tests와 llm-d22tests 통과, 양쪽 worktree clean. 원격 push 미수행
- Next: Qwen 이종PD용 graph·KV 형상 지원을 별도 확장하고 KONI 포화 decode/검증비용을 분리 계측
- Artifacts: experiments/npu_model_expansion_20261019/llmd-merged-tests.log
- Experiments: experiments/2026-10-19-npu-model-expansion.md
- By: codex@etri-001

### 2026-10-19 01:47+00:00 — PDDISAG-6

- Result: 실제 NPU 검증완료: ATOM Qwen0.5B/1.5B 144회·FP32 top1 240/240; chunk128 긴입력 prefill25–36% 감소, total1.011–1.105x, 짧은입력 악화. KONI RNGD→ATOM24요청396토큰·DMA1536 exact, 단계계측추가,17tests 통과
- Next: Qwen RNGD native registry 미등록/전용 graph·KV 형상 확장 필요. KONI full-batch decode·검증 오버헤드 분리 계측
- Params: atom_device=2, rngd_device=3, native_batch=1, prefill_chunks=64,128
- Artifacts: experiments/npu_model_expansion_20261019/analysis/summary.json, experiments/npu_model_expansion_20261019/analysis/native-atom.png
- Experiments: experiments/2026-10-19-npu-model-expansion.md
- By: codex@etri-001

### 2026-10-19 01:40+00:00 — PDDISAG-6

- Result: 실제 ATOM2에서 Qwen2.5-0.5B 6조건·18측정 실행성공. 두 Qwen 크기 chunk64/128 컴파일완료; RNGD native FXB registry 두 모델 미등록 확인. KONI 이종 PD 단계계측 실행중
- Next: 컴파일 종료 후 ATOM ABBA 비교와 FP32 teacher-forced 품질 확인, KONI 단계별 병목 집계
- Experiments: experiments/2026-10-19-npu-model-expansion.md
- By: codex@etri-001

### 2026-10-19 01:30+00:00 — PDDISAG-6

- Result: 모델/모듈 정리143e618 완료 및 llm-d feature 회귀통과: 21tests·실제 sidecar3요청/18토큰 exact. 5계열120cases, 학습Qwen0.5B/1.5B 8cases, CPU405조건측정 원시값·그래프·기여가이드 보존
- Next: 새 NPU 모델은 별도 graph 컴파일·KV 형식 검증이 필요. CPU 개선을 NPU 개선으로 확대해석하지 않음
- Experiments: experiments/2026-10-19-optimus-model-expansion.md
- By: codex@etri-001

### 2026-10-19 01:28+00:00 — PDDISAG-6

- Result: Optimus 모델/모듈 확장: 5계열120cases·실가중치Qwen0.5B/1.5B 8cases parity통과, CPU405측정요청에서 total1.052–1.275x·prefill1.356–2.094x. 계약/모델정책/캐시전송 분리, 개발자 가이드 추가
- Next: 엔진 변경 커밋 후 llm-d feature에 반영하여 회귀 검증
- Experiments: experiments/2026-10-19-optimus-model-expansion.md
- By: codex@etri-001

### 2026-10-19 01:15+00:00 — PDDISAG-6

- Result: Optimus 엔진0c54548에서 feat/optimus-llmd 별도 분기·구현. 18tests 통과, 실제 llm-d sidecar→KONI8요청/132토큰 direct exact·512 DMA CQE0·full-logit exact; Torch3요청도 통과, 정상종료 확인
- Next: 로컬 통합은 검증 완료. Kubernetes/EPP·streaming·다중 노드 확장은 별도 작업으로 유지
- Experiments: experiments/2026-10-19-optimus-llmd-validation.md
- By: codex@etri-001

### 2026-10-19 01:07+00:00 — PDDISAG-6

- Result: Optimus 통합엔진 실제 Torch/KONI 어댑터 검증: CPU13 tests·실모델24 cases 통과, RNGD3→ATOM2 24요청/1288토큰 full-logit exact, DMA1536/1536 CQE0, 정상종료·장치반환 확인
- Next: 검증 엔진 커밋 후 llm-d 연동을 별도 feature 브랜치로 분리
- Experiments: experiments/2026-10-19-optimus-engine-validation.md
- By: codex@etri-001

### 2026-10-19 00:53+00:00 — PDDISAG-34 ✅

- Result: hetero-npu-p2p PR#1 갱신(4b5933f): rm-core v0.1.1로 올리고 이중상속 NoHardware 우회 삭제, rm_core.errors.NoHardware 직접 사용, CLI 테스트를 runpy 기반으로 수정. 검증: unittest 91 OK, rm_core.conformance p2p SKIP(rc=0)·전체 rc=0. PR에 리뷰 반영 코멘트 남김. 실HW conformance는 미실행.
- By: claude@etri-001

### 2026-10-19 00:53+00:00 — PDDISAG-35 ✅

- Result: PR ai-computing/rngd-pd#1 갱신(b2e107a): rm-core pin v0.1.0→v0.1.1만 변경. conformance shm PASS·전체 rc=0, pytest test_airm 18 passed·tests 44 passed, check_sensitive clean. 리뷰 반영 코멘트 등록
- Next: PR 머지 후 ai-rm make bump
- Params: rm_core=0.1.1
- Artifacts: ../../../../tmp/claude-1002/-home-gspark-workspace-01-pd-disag/8d06644e-47b8-4f9d-8c05-ca55ce14fd70/scratchpad/conformance.txt, ../../../../tmp/claude-1002/-home-gspark-workspace-01-pd-disag/8d06644e-47b8-4f9d-8c05-ca55ce14fd70/scratchpad/pytest.txt
- By: claude@etri-001

### 2026-10-19 00:52+00:00 — PDDISAG-6

- Result: aicomp OptimusPrime 새 브랜치에 범용 PD 실행 계약과 선택형 P2P 어댑터 구현; CPU 계약 테스트 7개 통과. 실제 KONI 엔진 연결은 미완료.
- Next: KONI prefill/decode worker를 엔진 어댑터로 추출하고 캐시 위치·첫 토큰·마스크 전달 및 출력 동등성 검증
- Experiments: experiments/2026-10-19-optimus-integration.md
- By: codex@etri-001

### 2026-10-19 00:48+00:00 — PDDISAG-24

- Result: 동일2장 비교문서·그래프·2.2MB 증빙ZIP 완성. ZIP을 독립 디렉터리에서 재분석해1680요청 감사 및 전체수치 동일 재현, 입력변조 거부 확인. 모든 측정 서버 정상종료.
- Next: PD/vendor 동일 token history logits 비교로 prompt1/3 분기 원인 확인 후 작은batch decode·고부하역순검증
- Artifacts: experiments/same_budget_evidence_20261019.zip, hetero-npu-p2p/docs/same-budget-comparison.png
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-19 00:47+00:00 — PDDISAG-24

- Result: 동일 RNGD1+ATOM1 양구성1680요청 전검증·정상종료. 고부하PD127.862 vs native124.438tok/s(+2.75%), 저부하응답7.115 vs1.927초 악화. 본측정토큰열378/756일치로 품질동등성 미확정. 그래프·상대경로증빙 완성.
- Next: 두 프롬프트의 동일이력 logits 분기 원인 확인, 고부하 역순재현, 작은배치 decode graph의 안전한 KV/가중치 공유 검토
- Params: rngd_cards=1, atom_cards=1, seeds=27,28,29
- Artifacts: hetero-npu-p2p/docs/same-budget-comparison.json, hetero-npu-p2p/docs/same-budget-comparison.png
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-19 00:33+00:00 — PDDISAG-24

- Result: 동일예산 PD 첫seed336요청 검증 통과. 저부하응답7.115초 vs native1.927초 악화; 고부하127.862 vs124.438tok/s(+2.75%,단회). 나머지시드 진행.
- Next: 나머지2seed 완료 후 전체비교·단계별병목/한계 포함 최종문서·그래프
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-19 00:28+00:00 — PDDISAG-24

- Result: PD 같은 RNGD3+ATOM2 정상 로딩, warm84 전요청 full-logit exact/64DMA완료/소유권 검증 통과. client11254 본측정 진행.
- Next: 동일840요청 완료·정상종료 후 native와 최종비교·단계별병목진단·그래프
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-19 00:24+00:00 — PDDISAG-24

- Result: Native 동일2장840요청 전검증·정상종료·상대경로bundle 재감사 통과. 3req/s 중앙처리량124.438tok/s. 다음 PD 동일조건 시작.
- Next: 동일 RNGD3+ATOM2 PD 로딩·840요청·전체DMA/소유권/출력 감사 및 비교그래프
- Artifacts: hetero-npu-p2p/results/same-budget-native-r1/audit.json
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-19 00:21+00:00 — PDDISAG-33 ✅

- Result: PR https://github.com/ai-computing/rngd-pd/pull/1 — rm_core.conformance airm.kv_transports shm PASS (전체 conformance rc=0), pytest 44 passed, helm template/lint OK, check_sensitive clean
- Next: PR 리뷰·머지 후 ai-rm make bump; hetero-npu-p2p p2p adapter가 생기면 AIRM_KV_TRANSPORT=p2p 실기 검증
- New todos: rm-core: python -m rm_core.conformance 실행 시 __main__.NoHardware 불일치로 SKIP이 FAIL 처리되는 문제 이슈 등록; FencedTransport 다중 Pod(p2p) 경로: payload·tier를 설정으로 받아 cross-process 전송 정의
- By: claude@etri-001

### 2026-10-19 00:15+00:00 — PDDISAG-32

- Result: PR https://github.com/ai-computing/hetero-npu-p2p/pull/1 (airm/v0.1, af962ee). conformance: sim/loopback/greedy PASS, p2p SKIP (etri-001 probe·helper OK, live RNGD/ATOM allocation fixture 없음). unittest 91 OK(기존75+16), check_sensitive clean
- Next: live-allocation fixture 작성 후 NPU가 비었을 때 rebellion 노드 conformance PASS 확인, 머지 후 v0.6.0 태그
- New todos: hetero-npu-p2p conformance용 live RNGD/ATOM allocation fixture(HETERO_P2P_CONFORMANCE_FIXTURE) 작성·실측 PASS; PR #1 머지 후 v0.6.0 태그; rm-core: python -m rm_core.conformance가 __main__ NoHardware만 잡아 adapter의 SKIP이 FAIL로 나오는 문제 이슈화
- Artifacts: ../../../../tmp/claude-1002/-home-gspark-workspace-01-pd-disag/4209a7dd-0ffa-40ed-9ff2-a059dbee955c/scratchpad/conformance.txt
- By: claude@etri-001

### 2026-10-19 00:14+00:00 — PDDISAG-24

- Result: Native 동일2장 첫seed27 3조건 완료·워밍업포함336요청 정합성 통과. 고부하124.438tok/s, 응답중앙12.613초(단회). 두 시드 계속 측정.
- Next: Native 전체840 완료 후 정상 종료·감사, 동일2장 PD 시험
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-19 00:06+00:00 — PDDISAG-24

- Result: 동일2장 비교 시작: RNGD3+ATOM2 native router 준비, 양사 artifact 참조 해시 일치. 고정가중치63:70, 새seed27/28/29. client58448 warmup 진행.
- Next: Native840요청 완료·정상 종료·감사 후 동일 카드 PD840요청 실행
- Params: cards=2, seeds=27,28,29
- Experiments: experiments/2026-10-19-same-budget.md
- By: codex@etri-001

### 2026-10-18 23:49+00:00 — PDDISAG-24

- Result: RNGD 단독 840요청+참조4+smoke16 원시 감사 통과·정상 종료. 3seed 중앙 처리량 40.216/62.654/62.721 tok/s (0.5/1.5/3req/s). SDK batched delivery로 ITL p99 미산출.
- Next: backend별 참조 검증을 추가해 1RNGD+1ATOM whole-request 라우팅과 동일 두 장 PD 비교
- Params: backend=rngd, cards=1
- Artifacts: hetero-npu-p2p/docs/native-rngd-arrivals-audit.json, hetero-npu-p2p/docs/native-rngd-arrivals.png
- Experiments: experiments/2026-10-18-native-rngd-audit.md
- By: codex@etri-001

### 2026-10-16 10:13+00:00 — PDDISAG-24

- Result: 후속28667 poll:첫시드1.5req/s HTTP·참조검증 통과,총3조건252요청 완료.3req/s 진행
- Next: 같은28667 남은7조건 완료 후 RNGD 전용감사·동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:12+00:00 — PDDISAG-24

- Result: RNGD 동일28667 50초 live 확인,첫시드1.5req/s 진행. 완료2조건168요청 검증 유지
- Next: 같은28667 전체840완료 후 RNGD 전용감사·동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:11+00:00 — PDDISAG-24

- Result: RNGD 동일28667 verified wait:첫시드0.5req/s 완료,2조건168요청 HTTP·참조일치
- Next: 같은28667 남은8조건 완료 후 RNGD 전용감사·동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:10+00:00 — PDDISAG-24

- Result: RNGD 동일28667 50초 live 확인:첫시드0.5req/s 진행,워밍업84요청 검증 유지
- Next: 같은28667 전체840완료 후 RNGD 전용감사·동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:08+00:00 — PDDISAG-24

- Result: RNGD 동일28667 verified wait:워밍업84요청 HTTP·참조검증 통과,본측정 진행
- Next: 같은28667 본측정9조건 완료 후 RNGD 전용감사·동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:08+00:00 — PDDISAG-24

- Result: RNGD 동일28667 50초 live 확인,워밍업 진행. 관측71완료(기준·기능검사20포함),오류0
- Next: 같은28667 전체840요청 완료 후 RNGD 전용감사·동일카드예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:06+00:00 — PDDISAG-24

- Result: RNGD28667 live 유지. 실제로그에서 토큰묶음전달 확인; 개별ITL 미산출·TPOT 조건부계산 분석기 구현,3시험 통과
- Next: 동일28667 완료 후 RNGD 요청·토큰묶음 전체감사 및 동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:05+00:00 — PDDISAG-24

- Result: RNGD4입력 기준출력 확보·혼합16HTTP 참조일치. 동일840요청 단독기준 부하시험 시작,서버94148 유지
- Next: RNGD 부하완료·chunk로그검증 후 ATOM+RNGD 동시전체추론/PD 동일예산 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:04+00:00 — PDDISAG-24

- Result: RNGD3 재시도 서버94148 정상획득·native HTTP ready. 이전EBUSY 원인해소 이유미상,reset없음. 4입력×128 기준캡처67401 시작
- Next: 67401 기준출력 완료 후 RNGD 혼합HTTP정합성 및 동일카드예산 비교 진행
- Params: rngd_device=npu3
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:03+00:00 — PDDISAG-24

- Result: 실제ATOM 분할prefill 순차10+교차재사용8개 모두128토큰 exact·정상종료. pending prefill 중 다른슬롯decode10회,종료후 장치반환 확인
- Next: B8 decode용 pending KV 보호 경로 구현·검증; RNGD 재확인 및 동일카드예산 PD 비교
- Params: chunk=64, decode_batch=1
- Artifacts: hetero-npu-p2p/results/native-atom-chunks-r1/audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:01+00:00 — PDDISAG-24

- Result: chunk probe 동일58816 live 확인,전체기준10개 완료·분할출력 순차비교 진행. 완료구간 참조일치,교차재사용 포함 최종검증 대기
- Next: 같은58816 전체18비교 종료 후 원본·실제교차횟수 검증
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 10:00+00:00 — PDDISAG-24

- Result: chunk probe 동일58816 50초 live 확인,전체prefill/B1 기준출력 생성 중. 오류기록 없음·분할비교 아직 미완료
- Next: 같은58816 참조생성·분할·교차재사용 비교 완료 후 원본감사
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:58+00:00 — PDDISAG-24

- Result: 64토큰 분할prefill 실제ATOM2 정합성 probe58816 시작. 전체prefill/B1 기준10개·분할10개·교차재사용8개 계획,성능비교 아님
- Next: 같은58816 종료후 참조·교차·슬롯재사용 결과검증 및 원본보존
- Params: device=ATOM2, chunk=64, decode_batch=1
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:56+00:00 — PDDISAG-24

- Result: budget8/1 총1680측정요청+32기능검사 감사통과. 1.5req/s 토큰간격p99 631.59→431.38ms,처리량70.218→69.376(-1.20%);응답시간악화도 기록
- Next: 64토큰 분할prefill 실제정합성 검사; RNGD 획득 재확인 및 동일카드예산 PD 비교
- Params: batch=8, max_prefills=8,1
- Artifacts: hetero-npu-p2p/docs/native-atom-budget-comparison.json, hetero-npu-p2p/docs/native-atom-budget-comparison.png
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:53+00:00 — PDDISAG-24

- Result: 후속54332 poll에서 마지막시드0.5req/s 완료 확인:대조군8조건672요청 HTTP·참조일치
- Next: 같은54332 마지막2조건 종료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:52+00:00 — PDDISAG-24

- Result: 대조군 동일54332 50초 live 확인:마지막시드0.5req/s 진행. 완료7조건588요청 검증 유지
- Next: 같은54332 마지막3조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:51+00:00 — PDDISAG-24

- Result: 대조군 동일54332 verified wait:두번째시드 완료,7조건588요청 HTTP·참조일치. 마지막시드 진행
- Next: 같은54332 마지막3조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:50+00:00 — PDDISAG-24

- Result: 대조군 동일54332 verified wait:두번째시드1.5req/s 완료,6조건504요청 HTTP·참조일치. 3req/s 진행
- Next: 같은54332 남은4조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:48+00:00 — PDDISAG-24

- Result: 대조군 동일54332 50초 live 확인,두번째시드1.5req/s 진행. 완료5조건420요청 검증 유지
- Next: 같은54332 남은조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:47+00:00 — PDDISAG-24

- Result: 대조군 동일54332 verified wait:두번째시드0.5req/s 완료,5조건420요청 HTTP·참조일치
- Next: 같은54332 남은5조건 완료·전체감사 후 budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:46+00:00 — PDDISAG-24

- Result: 대조군 동일54332 50초 live 확인,두번째시드0.5req/s 진행. 완료4조건336요청 검증 유지
- Next: 같은54332 남은6조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:44+00:00 — PDDISAG-24

- Result: 대조군 후속54332 poll:첫시드3req/s HTTP·참조검증 통과,총4조건336요청 완료. 두번째시드 진행
- Next: 같은54332 남은6조건 완료·전체감사 후 budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:44+00:00 — PDDISAG-24

- Result: 대조군 동일54332 50초 live 확인,첫시드3req/s 진행. 재시작·실행설정변경 없음
- Next: 같은54332 남은측정 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:43+00:00 — PDDISAG-24

- Result: 대조군 동일54332 verified wait:첫시드1.5req/s 완료. 3조건252요청 HTTP·참조일치
- Next: 같은54332 남은7조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:41+00:00 — PDDISAG-24

- Result: 대조군54332 첫도착률 완료·168요청 HTTP/참조일치. chunk adapter 모의런타임3시험 통과(슬롯보호·재사용·실패차단),실제NPU 미검증
- Next: 같은54332 대조시험 완료·budget비교 후 chunk 실제정합성 검사
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:40+00:00 — PDDISAG-24

- Result: 대조군54332 live 유지. 별도64토큰prefill+B1decode 정합성검사 adapter 구현·문법검사 통과(실행 미검증)
- Next: 같은54332 대조측정 완료·budget비교 후 chunk adapter CPU/실제NPU 정합성검사
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:39+00:00 — PDDISAG-24

- Result: budget8 동일54332 verified wait:워밍업84요청 HTTP·참조검증 통과,본측정 진행
- Next: 같은54332 9개측정조건 완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:37+00:00 — PDDISAG-24

- Result: budget8 대조군 동일54332 50초 live 확인,워밍업 진행. 재시작·코드·설정 변경 없음
- Next: 같은54332 전체840완료 후 정상drain·전체감사·budget8/1 비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:36+00:00 — PDDISAG-24

- Result: budget8 대조군16요청 참조exact·237연산 재생통과. 실제startup 코드/바이너리/조건 일치 확인;840요청 client54332/server62726 시작
- Next: 동일54332 완료 후 정상drain·전체감사·budget8/1 처리량 및 지연 비교
- Params: batch=8, max_prefills=8
- Artifacts: ../../../../tmp/hetero-native-atom-b8-budget8-smoke-r1/operation-audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:35+00:00 — PDDISAG-24

- Result: budget1 전체840요청 감사통과·정상종료. 기능검사포함856요청12884연산 재생통과,733refill 최대1개. 동일계측budget8 서버62726 로딩 시작
- Next: 62726 준비후 동일16기능검사·840부하측정,완료후 budget8/1 비교
- Params: batch=8, max_prefills=1
- Artifacts: hetero-npu-p2p/results/native-atom-b8-budget1-arrivals-r1/audit.json, hetero-npu-p2p/results/native-atom-b8-budget1-arrivals-r1/operation-audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:31+00:00 — PDDISAG-24

- Result: 동일41811 live:마지막시드0.5req/s 완료. 8조건672요청 HTTP·참조일치
- Next: 동일41811 마지막2조건 종료 후 정상drain·전체감사·budget8 대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:30+00:00 — PDDISAG-24

- Result: 동일41811 50초 live 확인:마지막시드0.5req/s 진행. 기존7조건588요청 검증 유지,재시작 없음
- Next: 동일41811 마지막3조건 완료 후 정상drain·전체감사·budget8 대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:29+00:00 — PDDISAG-24

- Result: 후속 동일41811 poll에서 두번째시드3req/s 완료 확인:7조건588요청 HTTP·참조일치. 마지막시드 진행
- Next: 같은41811 마지막3조건 완료 후 정상drain·전체감사·budget8 대조시험
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:29+00:00 — PDDISAG-24

- Result: 동일41811 50초 live 확인,두번째시드3req/s 실행 중. 완료6조건504요청 검증 유지; 재시작·설정변경 없음
- Next: 동일41811 완료조건 확인,전체840요청 종료 후 감사·budget8 대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:27+00:00 — PDDISAG-24

- Result: 동일41811 verified wait:두번째시드1.5req/s 완료. 6조건504요청 HTTP·참조검증 통과
- Next: 같은41811 남은4조건 완료 후 정상drain·전체감사 및 budget8 대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:26+00:00 — PDDISAG-24

- Result: 동일41811 live:두번째시드0.5req/s 완료,총5조건420요청 HTTP·참조일치
- Next: 동일41811 남은5조건 완료 후 정상drain·전체감사·budget8 대조시험
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:24+00:00 — PDDISAG-24

- Result: 동일41811 verified wait:첫시드3도착률 완료,워밍업포함4조건336요청 HTTP/참조일치. 두번째시드 진행
- Next: 같은41811 남은6조건 완료·전체감사 후 동일계측budget8 대조시험
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:22+00:00 — PDDISAG-24

- Result: 동일41811 verified wait:첫시드1.5req/s 완료. 워밍업포함3조건252요청 HTTP/참조검증 통과,3req/s 진행
- Next: 같은41811 남은7조건 완료 후 정상drain·전체감사 및 동일계측budget8 대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:20+00:00 — PDDISAG-24

- Result: budget1 동일41811:워밍업+첫도착률168요청 HTTP/참조일치. 기능검사포함184요청3717연산 재생통과,87refill 모두1개제한 준수
- Next: 41811 남은8조건 완료 후 전체감사; 같은계측 budget8 대조시험
- Artifacts: hetero-npu-p2p/results/native-budget1-first-rate-operation-audit-r1/audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:19+00:00 — PDDISAG-24

- Result: 동일41811 live 유지. SDK 명시 block_tables·64토큰 prefix 경로 확인; pending 슬롯 보호 필요와 B1정합성→별도scratch B8 검증 순서 문서화
- Next: 현재budget1 본측정 완료 및 budget8 대조 우선; 이후 분할prefill 실제정합성 검사
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:17+00:00 — PDDISAG-24

- Result: 동일41811 live 확인:워밍업84요청 HTTP·참조검증 통과. 코드·설정 유지, 본측정 진행
- Next: 41811의 3도착률×3시드 완료 확인 후 정상drain·전체감사·budget8 대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:16+00:00 — PDDISAG-24

- Result: prefill 제한 후보 시험41811 실행 유지. 관측72요청 완료(기능검사16 포함), 오류0. budget8/1 전용 비교 검증 구현 및 조건 불일치 거부 시험2개 통과
- Next: 동일41811 완료·840요청 전체검증 후 동일 계측 budget8 대조 시험
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:16+00:00 — PDDISAG-24

- Result: budget1 client41811継続確認、観測72完了(16smoke含む)失敗0。budget8/1専用比較監査実装、条件不一致拒否テスト2件通過
- Next: 同じ41811完了待ち・全840監査後、同一計測budget8対照試験
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:14+00:00 — PDDISAG-24

- Result: B8 prefill budget1 실제 ATOM16요청 참조exact;238연산 재생검증·8회refill 모두1개제한 준수. 동일840요청 client41811/server90651 시작
- Next: 41811 완료 후 정상drain·전체감사; 같은계측 budget8 대조군 실행
- Params: batch=8, max_prefills=1, device=ATOM2
- Artifacts: ../../../../tmp/hetero-native-atom-b8-budget1-smoke-r1/operation-audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:11+00:00 — PDDISAG-24

- Result: B4/B8 총1680요청 최종감사 통과. 3req/s 처리량44.945→70.504tok/s(+56.87%), 응답중앙63.623→36.629초. 토큰간격p99 악화도 그래프에 보존
- Next: prefill budget8/1 하드웨어 정합성 및 동조건 대조; RNGD 획득 재확인 후 동일 카드 예산 비교
- Params: model=KONI, device=ATOM2, batches=4,8, seeds=17,18,19
- Artifacts: hetero-npu-p2p/docs/native-atom-b4-b8-comparison.png, hetero-npu-p2p/docs/native-atom-b4-b8-comparison.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:03+00:00 — PDDISAG-24

- Result: B4동일53949 verified wait:두번째seed완료·7/10조건588요청HTTP/참조일치. 마지막seed19진행·실행설정유지
- Next: 같은53949/46959 마지막3조건완료후정상drain·전체감사·B4/B8비교그래프
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 09:00+00:00 — PDDISAG-24

- Result: B4 5/10조건420요청exact·53949계속. 동일예산native프런트엔드준비·실CPU HTTP16요청두모의backend8/8분산·중복0·출력일치검증. 실제NPU라우팅미실행
- Next: 같은B4 53949/46959완료·B4/B8비교;RNGD확보후native동시분산검증과스케줄러ablation
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:58+00:00 — PDDISAG-24

- Result: B4 53949계속·두번째seed0.5에서74완료. 동시native대조용HTTPadapter추가:정상응답검증·HTTP500불확실완료예약유지 실제CPU HTTP2시험통과;NPU통합미실행
- Next: 같은B4 53949/46959완료·최종비교;RNGD가용성확인후두backend동시라우터연결
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:55+00:00 — PDDISAG-24

- Result: B4동일53949 verified wait:첫seed3rate완료·4/10조건336요청HTTP/참조일치,실패없음. 다음seed측정계속·코드설정유지
- Next: 같은53949/46959 seed18/19완료후정상drain·전체감사·B4/B8최종비교
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:53+00:00 — PDDISAG-24

- Result: B4 3/10조건252요청exact·53949계속. 동일seed17 rate1.5 B4/B8 45.23/69.50tok/s·응답42.62/20.20s,조건/출력검증통과(단회중간값)
- Next: 같은53949/46959나머지7조건완료후전체감사·3seed최종비교;이후스케줄러ablation
- Artifacts: hetero-npu-p2p/results/native-b4-first-two-rates-r1/paired-rate1.5.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:51+00:00 — PDDISAG-24

- Result: B4첫도착률완료·168요청exact,53949계속. 동일첫seed0.5rate 단회B4/B8 37.68/39.09tok/s,응답8.05/7.23s;조건/출력일치검증,최종결론아님
- Next: 같은53949/46959남은8조건완료후전체감사·3seed비교;후속스케줄러ablation
- Artifacts: hetero-npu-p2p/results/native-b4-first-rate-diagnostic-r1/paired-first-rate.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:49+00:00 — PDDISAG-24

- Result: B4동일53949진행확인. 독립라우터의네트워크실패≠SDK종료계약보강:불확실완료예약유지·새배정차단·재전송없음 CPU3시험통과;NPU연결아직미구현
- Next: 같은B4 53949/46959완료·감사후비교. 동시native라우터는백엔드완료확인계약과함께연결
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:47+00:00 — PDDISAG-24

- Result: B4워밍업84요청HTTP·참조일치,53949측정계속. 동일예산독립추론용라우터코어구현·중복방지/반복취소예약유지/실패기록 CPU2시험통과;NPU연결미검증
- Next: 같은53949/46959전체완료·B4/B8감사비교;라우터HTTP연결과RNGD가용성확인후동일예산대조
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:45+00:00 — PDDISAG-24

- Result: B4 client53949/server46959동일실행진행중. 실제B4/B8startup24파일해시·설정동일확인. RNGD drain없음읽기전용확인(EBUSY해소증거아님)
- Next: 같은53949/46959완료·전체감사후B4/B8비교;RNGD실제획득재확인은측정간섭없는시점에
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:43+00:00 — PDDISAG-24

- Result: B8전체840요청·10조건exact/최종감사통과·정상종료. 도착률0.5/1.5/3 처리량중앙값39.09/69.66/70.50tok/s·최종그래프생성. B4동조건server46959/client53949시작
- Next: 같은B4 53949/46959완료후전체감사·B4/B8비교;이후budget1동일계측ablation
- Artifacts: hetero-npu-p2p/docs/native-atom-b8-arrivals.json, hetero-npu-p2p/docs/native-atom-b8-arrivals.png
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:36+00:00 — PDDISAG-24

- Result: 동일B8 client76183 verified wait:2개seed완료·7/10조건588요청exact·서버오류0. 마지막seed19진행,코드/설정변경없음
- Next: 같은76183/94183 마지막3조건완료확인→정상drain→전체감사→B4동조건실측
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:35+00:00 — PDDISAG-24

- Result: B8기준6/10조건504요청exact·76183계속. 동일계측control8/budget1 독립실행경로·설정검증준비;현재서버코드유지·새하드웨어실험미시작
- Next: 같은76183/94183남은4조건완료·감사,B4동조건측정;그후budget하드웨어출력검증
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:34+00:00 — PDDISAG-24

- Result: B8기준5/10시나리오420요청exact·76183계속. 원본/fullbudget/budget1의24혼합요청 CPU작업보존검증추가·3시험통과;하드웨어개선미검증
- Next: 같은76183/94183남은5시나리오완료·감사후B4측정;후속budget하드웨어정합성검증
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:33+00:00 — PDDISAG-24

- Result: 현재B8기준측정유지. 별도bounded-prefill스케줄러구현:진행중decode사이1요청admit·SDK구간계측·단일스레드/오류전파 CPU2시험통과. 하드웨어미적용
- Next: 같은76183/94183완료와B4동조건대조; 이후계측동일control vs budget1하드웨어parity·ITL검증
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:30+00:00 — PDDISAG-24

- Result: B8첫seed3rate완료·워밍업포함336요청exact. hostTPOTp99 106/123/126ms vs ITLp99 409/432/509ms 중간그래프생성·시각검수. 나머지seed계속
- Next: 같은76183/94183 seed18/19완료후전체감사와B4대조;prefill간섭후보는동조건별도검증
- Artifacts: hetero-npu-p2p/results/native-b8-interval-diagnostic-r2/host-token-latency.png, hetero-npu-p2p/results/native-b8-interval-diagnostic-r2/diagnostic.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:28+00:00 — PDDISAG-24

- Result: B8첫Poisson84요청exact. 단회host TPOTp99 106ms vs ITLp99 409ms;150ms초과248간격 모두 다른요청첫토큰과겹침(인과미확정). 1.5rate측정계속
- Next: 전체같은76183/94183완료후B4동조건대조; prefill시각계측·admission제한ablation으로긴토큰간격원인검증
- Artifacts: ../../../../tmp/hetero-native-atom-b8-arrivals-r2/interim-poisson-0.5-r0.json, ../../../../tmp/hetero-native-atom-b8-arrivals-r2/interim-prefill-overlap-r0.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:26+00:00 — PDDISAG-24

- Result: B8 r2계속:워밍업84+첫Poisson57완료·서버오류0. B4/B8비교검증기추가(동일timeout/plan/tokens/runtime강제),불일치거부시험통과
- Next: 같은76183/94183 전체완료까지관찰; 완료감사후 B4동조건600초실측. RNGD관리자진단은인증필요
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:24+00:00 — PDDISAG-24

- Result: B8 r2 워밍업84/84 HTTP·참조일치:기존15연결실패재발없음. client76183 Poisson측정계속. Sarathi적용시SDK블록초기화·dummy lane충돌제약코드확인
- Next: 동일76183/94183 전체10시나리오완료·감사 후 B4동조건실측;prefill간섭제한후보검증
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:21+00:00 — PDDISAG-24

- Result: B8 r1:69/84출력일치·15연결reset실패보존,정상drain. backlog5→128수정·84연결CPU통과/기존설정실패. r2 server94183/client76183 timeout600실측시작
- Next: 동일76183/94183 완료까지관찰·840요청감사; B4동조건600초대조실행
- Params: http_backlog=128, http_timeout_seconds=600
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:18+00:00 — PDDISAG-24

- Result: B8 HTTP16요청 exact·호스트시간검증통과. 840요청실측 client54923/server33531진행중. 완료결과감사도구840합성fixture/실제prompt변조검출통과;180초timeout비교조건점검필요
- Next: 같은54923/33531 완료확인·보존; 최종B4/B8비교는 충분한 동일client timeout 적용
- Artifacts: ../../../../tmp/hetero-native-atom-b8-http-r1/smoke-audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:15+00:00 — PDDISAG-24

- Result: B8컴파일293.089초·ATOM2로딩14.6GiB성공. 8x128/혼합32요청/refill24회참조동일, B4/B8참조도동일. 호스트토큰시각기록 추가; B8 HTTP33531시작
- Next: 동일HTTP33531 ready확인 후16요청검증·84요청3rate3seed단독측정; B4대조필요
- Params: batch=8, context=512
- Artifacts: hetero-npu-p2p/results/native-atom-b8-probe-r1/audit.json
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:12+00:00 — PDDISAG-24

- Result: B8컴파일3144 세번째graph진행중 확인. PD와동일84요청·3도착률·3seed native측정도구 준비, 참조불일치시84응답보존·중단 CPU검증통과
- Next: 동일compile3144 완료후 ATOM2 B8로딩·출력검증, native B4/B8동조건부하측정
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:09+00:00 — PDDISAG-24

- Result: 제조사B8/context512/chunk64 독립컴파일 시작(session3144); 배치폭 일반화 및 8슬롯 사용 CPU검증통과. 실제B8메모리/출력미검증
- Next: 동일compile3144 완료 확인 후 ATOM2 B8참조·refill검증, B4/B8동조건비교
- Params: batch=8, context=512
- Experiments: experiments/2026-10-16-native-atom-b8.md
- By: codex@etri-001

### 2026-10-16 08:07+00:00 — PDDISAG-24

- Result: ATOM32/64/96/128 전조합검증: native16요청·refill12회 및 HTTP16요청1280토큰 참조동일·해시동일·정상종료. RNGD원인미확정
- Next: 제조사 단독B8 메모리/성능가능성 및 라우팅준비; RNGD획득복구후 동일카드예산대조
- Artifacts: hetero-npu-p2p/results/native-atom-http-long-r1/http-long-audit.json
- Experiments: experiments/2026-10-16-native-http-baseline.md
- By: codex@etri-001

### 2026-10-16 08:02+00:00 — PDDISAG-24

- Result: ATOM native HTTP8요청 vendor참조일치·서버/클라이언트출력일치·artifact해시동일·정상종료. 작업스레드 소유권검사수정. RNGD r2 EBUSY실패보존
- Next: 32~128토큰 native 참조/HTTP검증 및 라우팅준비; RNGD획득문제 진단 후 동일예산실측
- Artifacts: hetero-npu-p2p/results/native-atom-http-r2/http-smoke-audit.json
- Experiments: experiments/2026-10-16-native-http-baseline.md
- By: codex@etri-001

### 2026-10-16 07:58+00:00 — PDDISAG-24

- Result: ATOM native B4 슬롯재사용 검증 통과: 혼합8요청·진행중교체4회 vendor출력동일. 단일SDK스레드 async큐 구현, 취소·실패·drain 포함 CPU4시험통과
- Next: native ATOM HTTP 연결 및 실제 요청 검증 후 동일카드예산 비교; RNGD3 EBUSY 재확인
- Artifacts: hetero-npu-p2p/results/native-atom-probe-refill-r1/result.json
- Experiments: experiments/2026-10-16-native-http-baseline.md
- By: codex@etri-001

### 2026-10-16 07:48+00:00 — PDDISAG-24

- Result: 단독SDK HTTP연결구현·CPU실HTTP시험통과. RNGD3엔진초기화EBUSY로실패보존;정확한fusion8최소할당도npu3EBUSY재현,성능미측정
- Next: ATOM단독배치API/동일예산라우팅준비와RNGD가용성재확인;reset/타작업종료없음
- Artifacts: experiments/native_pd_2026_4/rngd_busy_20261016/native-http-fused-probe-r1/result.json
- Experiments: experiments/2026-10-16-native-http-baseline.md
- By: codex@etri-001

### 2026-10-16 07:41+00:00 — PDDISAG-25

- Result: partial4최종완료:양3192요청exact·소유권0·출력동일·runtime해시동일. ref3 115.41→131.87tok/s(+14.26%),응답23.27→13.56s(-41.71%).sealed회귀보존·refill한정권장
- Next: 동일카드예산단독대조준비와공개trace큰context검증;전력/순서교차추가검증도남음
- Artifacts: hetero-npu-p2p/docs/b14-arrival-quantum-comparison.json, hetero-npu-p2p/docs/b14-arrival-quantum-comparison.png
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:34+00:00 — PDDISAG-25

- Result: partial4 14/19완료·30775계속.공개Azure대화trace19366행확보/해시검증:현재448/128길이조건1690행(8.7%)뿐,긴문맥검증필요
- Next: 현재partial4완료후비교;공개trace는길이제약·제외비율명시하고큰context별도검증
- Artifacts: experiments/native_pd_2026_4/azure_trace_20261016/manifest.json
- Experiments: experiments/2026-10-16-azure-trace-coverage.md
- By: codex@etri-001

### 2026-10-16 07:31+00:00 — PDDISAG-25

- Result: partial4 13/19완료1092요청exact/위반0·대조출력동일.두번째ref3중앙17.43→13.56s개선, sealed3 111.75→97.07tok/s회귀보존
- Next: 31364/30775 마지막seed6구간완료후전3192요청쌍검증;partial4는refill한정효과검토
- Artifacts: ../../../../tmp/hetero-b14-arrival-partial4-r1/second-repeat-comparison.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:24+00:00 — PDDISAG-25

- Result: partial4 8/19완료672요청exact/위반0.두번째0.5refill중앙9.91→7.11s,p9917.62→11.51s,처리량거의동일.30775계속
- Next: 같은31364/30775 잔여11구간완료 후전반복비교,완료전기본정책변경없음
- Artifacts: ../../../../tmp/hetero-b14-arrival-partial4-r1/second-refill05-comparison.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:20+00:00 — PDDISAG-25

- Result: partial4 첫seed완료588요청exact/위반0·대조군출력일치. ref1.5 중앙13.26→7.65s,ref3 중앙25.22→12.29s·97.24→131.87tok/s;후보7/19,추가seed진행
- Next: 같은30775 seed18·19완료후모든반복·순서변동포함비교,단회개선과일반효과구분
- Artifacts: ../../../../tmp/hetero-b14-arrival-partial4-r1/first-repeat-comparison.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:17+00:00 — PDDISAG-25

- Result: partial4 첫refill0.5 개선관측:중앙9.08→7.29s,p9916.75→11.93s,prefill-ready후dispatch대기1.90→0.17s.양prefix각252요청exact/위반0;후보3/19진행
- Next: 30775의1.5/3req/s와추가seed계속;전19구간후최종감사·단일요인비교,기본채택보류
- Artifacts: ../../../../tmp/hetero-b14-arrival-partial4-r1/first-refill-comparison.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:13+00:00 — PDDISAG-25

- Result: partial4 첫sealed0.5완료168요청exact/위반0.응답중앙11.52→11.28s,p99 21.18→22.53s악화,처리량거의동일.30775 refill0.5진행
- Next: 동일31364/30775 후보잔여17구간 완료후모든반복비교;부정적결과보존
- Artifacts: ../../../../tmp/hetero-b14-arrival-partial4-r1/first-sealed-comparison.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:09+00:00 — PDDISAG-25

- Result: partial4 워밍업84요청완료,첫0.5req/s sealed본측정진입.실제iteration policy partial_quantum4확인;30775 계속
- Next: 19구간완료후 전1596요청검증 및 fixed128대조;같은live핸들유지
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:09+00:00 — PDDISAG-25

- Result: partial4 31364 HTTP_READY,30775 워밍업진행.양실행startup runtime80파일동일,actualconfig 차이out·partialquantum뿐확인
- Next: 같은30775 워밍업후19구간 진행;실제policy4확인·최종검증 및 대조군비교
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:06+00:00 — PDDISAG-25

- Result: 완료대조군보존후 partial4 후보server31364/readiness-client30775 시작.모델·장치·도착계획동일,warm128유지;후보성능미측정
- Next: 31364/30775 준비·19구간측정완료확인 후 전체감사 및 대조군단일요인비교
- Params: partial_quantum=4, batch=14
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 07:06+00:00 — PDDISAG-25

- Result: 대조군전체완료1596요청exact/위반0,클라이언트·서버exit0.3req/s중앙sealed/refill102.34/115.41tok/s·응답29.60/23.27s.원본보존·그래프검수완료
- Next: 준비된partial4 동일19구간시험 시작후단일요인비교
- Artifacts: hetero-npu-p2p/docs/b14-arrival-length32-validation.json, hetero-npu-p2p/docs/b14-arrival-length32.png
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:57+00:00 — PDDISAG-25

- Result: 13/19완료1092요청exact/위반0.두번째3req/s sealed/refill111.75/122.78tok/s,응답중앙21.69/17.43s.최종seed19진행;전반복그래프도구준비
- Next: 같은54604 잔여6구간완료후전체감사·그래프·종료,partial4시험
- Artifacts: ../../../../tmp/hetero-b14-arrival-length32-r1/two-repeats-diagnostic.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:55+00:00 — PDDISAG-25

- Result: 11/19구간완료,924요청prefix replay exact/소유권0.두번째0.5req/s sealed/refill중앙11.46/9.91s.54604는3req/s seed18계속
- Next: 잔여8구간 유지실행후 최종1596요청감사·종료·보존 및partial4 대조
- Artifacts: ../../../../tmp/hetero-b14-arrival-length32-r1/second-repeat-partial-diagnostic.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:51+00:00 — PDDISAG-25

- Result: 대조군54604계속.시작시점대비runtime/부하생성80Python파일변경0.후속partial4 설정준비 및 소스변경거부검증추가,2시험통과
- Next: 현재대조군전체완료/종료/감사후 후보partial4시작;동시에장치실험시작하지않음
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:49+00:00 — PDDISAG-25

- Result: verifiedwait:54604 live50s×2 확인,refill0.5 seed18완료;총8/19시나리오완료.실측worker시간에full-logit검증비용포함확인,현재조건유지
- Next: 같은33403/54604 유지해잔여11시나리오완료;최종감사후partial4 동일조건비교
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:46+00:00 — PDDISAG-25

- Result: 첫seed6구간완료588요청exact/위반0.3req/s sealed/refill78.60/97.24tok/s·응답중앙34.47/25.22s.1.5req/s refill처리량소폭하락도보존.두번째seed진행
- Next: 54604/33403 유지,seed18·19 완료후 전1596요청검증 및 partial4 대조
- Artifacts: ../../../../tmp/hetero-b14-arrival-length32-r1/first-repeat-diagnostic.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:43+00:00 — PDDISAG-25

- Result: 1.5req/s sealed r0완료,54604계속. 실제dispatch프롬프트/출력길이와도착계획 연결검증추가;완료4구간583dispatch일치,손상프롬프트거부시험통과
- Next: 동일54604 전체완료 뒤1596요청 전검증;partial4 동일조건비교
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:41+00:00 — PDDISAG-25

- Result: 0.5req/s 첫쌍완료:sealed/refill 응답중앙11.521/9.085s,응답p99 21.181/16.745s.252요청exact/소유권0.다음1.5req/s실측지속
- Next: 54604 전체18구간완료검증; 같은plan partial4 대조. 기존단독16요청결과는 동등예산대조에서 제외
- Artifacts: ../../../../tmp/hetero-b14-arrival-length32-r1/first-pair-diagnostic.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:38+00:00 — PDDISAG-25

- Result: 첫0.5req/s sealed 완료:중앙응답11.521s,lane22.73%,최대전송지연6.46ms. warm포함168요청 replay exact/소유권0.54604 refill측정지속
- Next: 기존33403/54604 유지,18구간완료 후 전체감사 및 같은조건 partial4 대조
- Artifacts: ../../../../tmp/hetero-b14-arrival-length32-r1/first-completed-diagnostic.json
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:36+00:00 — PDDISAG-25

- Result: 54604 실측지속. 첫0.5req/s sealed 중간13iteration의 유효lane14.73%; 부분구간진단으로만보존. 집계에 길이별응답·점유분포·예정도착지연 추가,테스트통과
- Next: 전체18구간 완료검증; sparse batch의 partial4 효과를 같은부하에서 비교
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:35+00:00 — PDDISAG-25

- Result: client54604 warm passed True 확인, 완료응답90개 관측:84워밍업 뒤 첫0.5req/s 본측정 진행. 서버33403 유지
- Next: 18본측정 구간 전체 완료 및 등록계획·DMA·출력 감사; 이후 partial4 대조
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:34+00:00 — PDDISAG-25

- Result: 현재33403 HTTP_READY,54604 워밍업 실측중. 후속 partial4 설정 opt-in 연결, warm128 유지 및 실제정책검증; 집중시험5개통과
- Next: 기존54604의 18개 Poisson 구간 완료까지 유지, 출력/lease감사 후 같은조건 partial4 시험
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:33+00:00 — PDDISAG-25

- Result: server33403 WORKER_READY/WEIGHTS_LOADED 확인, client54604 대기 유지. 도착계획·quantum 고정조건 검증 및 손상거부 테스트 추가통과
- Next: 동일 live세션 측정 지속; 완료 후 전요청 감사. ORCA 원리의 partial4 대조 후보 평가
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:31+00:00 — PDDISAG-25

- Result: 독립길이 Poisson 도착 시험 실행기·DMA/출력 검증기 연결. 관련CPU10시험통과. ATOM2 idle 확인 후 server33403시작
- Next: 서버33403 및 readiness/client 동일세션 유지, 19시나리오1596요청 완료 후 사건검증
- Params: batch=14, lookahead=32
- Experiments: experiments/2026-10-16-b14-arrivals.md
- By: codex@etri-001

### 2026-10-16 06:28+00:00 — PDDISAG-25

- Result: 전력 비교 완료: 두 구성1240요청exact/위반0. padding84 sealed132.109tok/s·합산1.20547tok/J; refill131.812tok/s·1.20072tok/J. 기본정책변경보류
- Next: 독립 길이·도착 분포의 open-loop 실행 및 이벤트 검증 연결; 동일 카드 예산 단독 대조군
- Metrics: tok_per_j=1.205473929
- Params: batch=14, context=448, queue_policy=padding, lookahead=84
- Artifacts: hetero-npu-p2p/docs/b14-power-padding84-validation.json
- Experiments: experiments/2026-10-16-b14-power-mixed84.md
- By: codex@etri-001

### 2026-10-16 06:20+00:00 — PDDISAG-25

- Result: paddingpower warm144.944/첫sealed132.109tok/s,client44868계속. 분포독립단일replica검증분리,기존620/368로그재생통과·손상CQE/세대거부·검증기해시추가
- Next: 현재44868완료검증및에너지비교; 다음arrival전체실행/검증연결
- Experiments: experiments/2026-10-16-arrival-plan.md
- By: codex@etri-001

### 2026-10-16 06:17+00:00 — PDDISAG-25

- Result: padding전력server42049/client44868유지. 도착계획HTTPadapter추가, 로컬테스트2개통과:route/ID/요청형식·개수불일치실패보존. 신규NPU도착실측은아직없음
- Next: 현재전력비교완료검증; 이후도착시험전체이벤트검증·실행연결
- Experiments: experiments/2026-10-16-arrival-plan.md
- By: codex@etri-001

### 2026-10-16 06:16+00:00 — PDDISAG-25

- Result: padding84 power서버42049/client44868시작, actualbinding·ATOMpreflight통과. 에너지요약스크립트로length기존수치정확재현, 토큰수로그직접계산
- Next: 42049/44868관측→620요청/6전력구간검증→length32와설정조합비교
- Params: batch=14, lookahead=84, queue_policy=padding, power_sampling=true
- Experiments: experiments/2026-10-16-b14-power-mixed84.md
- By: codex@etri-001

### 2026-10-16 06:14+00:00 — PDDISAG-25

- Result: 전력시험620요청exact/ownership0/정상종료,6구간전력검증통과. refill133.868tok/s·중앙1.203758tok/J, 평균RNGD56.27W/ATOM55.72W(보드한정)
- Next: padding/lookahead84도동일power계측으로측정해처리량·응답·에너지비교
- Metrics: tok_per_j=1.203758, pd_sat_tok_s=133.867664
- Params: batch=14, queue_policy=length, lookahead=32, power_sampling=true
- Artifacts: hetero-npu-p2p/docs/b14-power-length32-validation.json
- Experiments: experiments/2026-10-16-b14-power-mixed84.md
- By: codex@etri-001

### 2026-10-16 06:09+00:00 — PDDISAG-25

- Result: 전력client45430 warm151.07완료/센서양보드오류0(중간검사). 다음도착시험open-loop실행기추가,느린응답독립dispatch·실패보존3검사통과
- Next: 현재전력본측정관측·완료검증; open-loop HTTP연결과실제도착지연분석추가
- Experiments: experiments/2026-10-16-arrival-plan.md
- By: codex@etri-001

### 2026-10-16 06:07+00:00 — PDDISAG-25

- Result: 전력server60362 REFERENCES_READY/client45430대기진행. 다음도착시험용84요청16조합균형계획+seededburst/periodic/Poisson생성,3검사통과·7계획저장(실측아님)
- Next: 현재전력시험계속관측; 이후open-loop client에실제submit지연·실패기록연결
- Experiments: experiments/2026-10-16-arrival-plan.md
- By: codex@etri-001

### 2026-10-16 06:05+00:00 — PDDISAG-25

- Result: B14 length/lookahead32 전력포함시험 server60362/client45430시작. RNGDactualbinding·ATOMpreflight통과; 동일구간보드J/tokJ평가준비, 새전력수치아직없음
- Next: 현재60362/45430관측, 620요청검증후보드전력coverage검사; 다음padding84도동일계측
- Params: batch=14, lookahead=32, power_sampling=true
- Experiments: experiments/2026-10-16-b14-power-mixed84.md
- By: codex@etri-001

### 2026-10-16 06:03+00:00 — PDDISAG-25

- Result: lookahead84완료620요청exact/ownership0/정상종료. padding대비sealed+4.16%,refill+4.40%;32응답16.62→7.99s개선/64응답악화. length최고처리량139.04에는미달
- Next: length/refill 처리량기준과padding/lookahead84 지연절충구성의실제도착·전력비교준비
- Metrics: pd_sat_tok_s=131.352
- Params: batch=14, lookahead=84, queue_policy=padding
- Artifacts: hetero-npu-p2p/docs/b14-lookahead84.png
- Experiments: experiments/2026-10-16-b14-lookahead84.md
- By: codex@etri-001

### 2026-10-16 05:59+00:00 — PDDISAG-30 ✅

- Result: ai-rm 정리 완료. rngd_llmd(ffdcf70): bench/kpi_slo.py·load_gen.py·sweep_llmd.py의 게이트웨이·제어노드 IP 4건 제거→bench/cluster_env.py가 LLMD_GATEWAY_URL/PD_GATEWAY/KUBECTL_CMD를 env 또는 untracked cluster.local.env에서 읽고 미설정 시 오류 종료, cluster.example.env·ai-rm.yaml·README ai-rm 절 추가. hetero-npu-p2p(36c454a): ai-rm.yaml·README ai-rm 절 추가(env 불필요). checker 양쪽 clean, 테스트 hetero unittest 75 OK, rngd native_v4 26 passed+transport/connector_tcp PASS
- Next: rngd_llmd tests/run_selftests.sh가 옛 디렉터리 구조를 가리켜 11/12 실패: 정리 또는 삭제 필요
- New todos: rngd_llmd tests/run_selftests.sh를 src/rngd_pd 구조에 맞게 갱신
- By: claude@etri-001

### 2026-10-16 05:58+00:00 — PDDISAG-25

- Result: lookahead84 warm142.129/첫sealedmixed130.604tok/s, client50566실행중. 기존padding32사건분석:576step동일, 출력32디코드전대기sealed약26s/refill약12s
- Next: client50566나머지반복확인후전체검증, lookahead확대의초기대기·처리량tradeoff비교
- Experiments: experiments/2026-10-16-b14-lookahead84.md
- By: codex@etri-001

### 2026-10-16 05:56+00:00 — PDDISAG-25

- Result: lookahead84 server30209 HTTP_READY; client50566진행. 비교도구lookahead전용검증추가, 기존pair회귀·confound거부검사통과
- Next: 동일client50566 반복완료후620요청재생검증/정상종료, paddinglookahead32대비처리량·응답비교
- Params: lookahead=84, batch=14
- Experiments: experiments/2026-10-16-b14-lookahead84.md
- By: codex@etri-001

### 2026-10-16 05:54+00:00 — PDDISAG-25

- Result: padding620요청검증근거저장:refill139.044→125.811tok/s(-9.52%), 출력32응답31.20→16.62s. lookahead32→84만변경한server30209/client50566시작
- Next: 현재server30209/client50566관측·620요청검증후lookahead대기/슬롯점유tradeoff분석
- Params: batch=14, lookahead=84, queue_policy=padding
- Experiments: experiments/2026-10-16-b14-lookahead84.md
- By: codex@etri-001

### 2026-10-16 05:49+00:00 — PDDISAG-6

- Result: GitHub origin/main연결및동일HEAD확인. README를조직중립라이브러리안내로재작성:설치·CPUquickstart·API·지원범위·문서·라이선스미선언. 링크와예제실행검증,db5ffec커밋
- Next: padding완료결과(처리량회귀·단기지연개선)를근거문서로정리하고후속최적화계속
- By: codex@etri-001

### 2026-10-16 05:33+00:00 — PDDISAG-25

- Result: 사용자fuser빈출력후 actual32byte할당/sync/물리RNGD3검증성공. reset없이padding새retry 서버78658/sidecar71198/client38193시작;compute binding·ATOM사전검사통과
- Next: 서버78658 readiness 및client38193 620요청측정관측, 완료후사건검증·length정책비교
- Experiments: experiments/2026-10-16-b14-padding84.md
- By: codex@etri-001

### 2026-10-16 05:20+00:00 — PDDISAG-25

- Result: padding-probe-r3실패보존(DMA0). 자체서버없음확인, idle sidecar종료요청. 코드·75시험·전력센서준비완료, 새성능수치없음
- Next: 관리자 sudo fuser -v /dev/rngd/npu3* 로점유확인; 해제후실제32byte할당성공검증→새paddingrun
- Blocker: RNGD3 EBUSY가3연속goalturn재현; SMI점유자없음, root fuser는sudo인증필요. 실제장치가용성복구/관리자점유진단필요
- Experiments: experiments/2026-10-16-b14-padding84.md
- By: codex@etri-001

### 2026-10-16 05:19+00:00 — PDDISAG-25

- Result: RNGD EBUSY재현으로padding대기. HTTP구간별보드에너지옵션구현,75시험통과. 센서only2초 RNGD41W/ATOM19.44W 확인(추론효율아님), 전력계측유무혼합비교거부
- Next: RNGD실제할당복구확인후padding타이밍시험; 이후양쪽동일전력계측으로에너지평가
- Experiments: experiments/2026-10-16-iteration-power.md
- By: codex@etri-001

### 2026-10-16 05:16+00:00 — PDDISAG-25

- Result: padding시험은RNGD3 EBUSY로DMA전종료; 최소32byteprobe도재현, SMI소유자없음. 실패원본보존. 완료mixed84처리량·응답그래프0bcd1e3저장/검수
- Next: RNGD가용성재확인후새run으로padding실측; 재시도전실제할당확인, reset/타작업종료금지
- Experiments: experiments/2026-10-16-b14-padding84.md
- By: codex@etri-001

### 2026-10-16 05:14+00:00 — PDDISAG-25

- Result: B12/B14총1240요청exact/ownership0/정상종료. mixed sealed115.103→124.838(+8.46%), refill115.258→139.044(+20.64%). B14refill짧은응답악화도보존
- Next: B14동일graph/lookahead32 padding vs length 실측, 처리량과길이별응답tradeoff검증
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 05:10+00:00 — PDDISAG-25

- Result: B14warm151.45/첫mixedsealed125.464tok/s, client66721계속. B12offline모형padding후보발견(실측아님), 3검사통과·근거기록
- Next: B14나머지반복관측·620요청검증후비교; 다음동일lookahead32 length/padding실측
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 05:08+00:00 — PDDISAG-25

- Result: B12변동원인계측:worker85.8–87.3ms/step대비 총576–704step, 빈lane/tail주요후보. B14server57263 HTTP_READY/client66721실측중, 시작소스자동기록확인
- Next: client66721완료/620요청검증후B12비교; 혼합배치tail감소정책·lookahead후보검증
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 05:06+00:00 — PDDISAG-25

- Result: B12혼합620요청검증완료. B14동조건server57263/client66721 초기화진행, source기록자동저장확인. 전력연속측정은아직미수행
- Next: live client66721 완료후사건재생검증 및 B12/B14처리량·길이별응답비교
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 05:06+00:00 — PDDISAG-22 ✅

- Result: ABBA4프로세스1472요청exact/ownership0/정상종료. B12중앙요약134.265 B14 151.488tok/s(+12.83%); 역순쌍+12.34%. 전체근거41fdbc6, 시작실패도보존
- Next: PDDISAG-25 혼합부하·전력·지연 평가 계속; 일반성은미확인
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 05:05+00:00 — PDDISAG-2

- Result: B12 mixed620요청exact/ownership0/정상종료. sealed115.103 refill115.258tok/s(+0.135%), 출력32응답24.53→33.28s악화. 시작소스기록누락은사후기록명시, 자동수집fddd45e추가. B14서버57263시작
- Next: B14서버57263 준비·client측정 관측 후동조건비교; B12사후provenance한계명시
- Metrics: b12_mixed_sealed_tok_s=115.103, b12_mixed_refill_tok_s=115.258
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 04:57+00:00 — PDDISAG-2

- Result: mixed試験コードaf4aa37コミット。B12 HTTP_READY/32요청probe완료,35iteration실패0. client90110 워밍업진행; 비교도구기존pair회귀통과
- Next: client90110 완료까지관측 후620요청전체재생검증; 서버76994정상종료 후B14실측
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 04:55+00:00 — PDDISAG-2

- Result: B12 mixed84 서버76994 WORKER_READY/WEIGHTS_LOADED; client90110 준비대기 후 자동측정, sidecar86634. CPU68시험통과. 구성당620요청
- Next: 현재서버76994/client90110 관측 계속; 완료후 사건검증·정상종료하고 B14동조건실행
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 04:54+00:00 — PDDISAG-2

- Result: ABBA증빙41fdbc6커밋. 혼합84요청 sealed/refill 전용모드와 사건검증/양배치설정 준비; 기존368요청재생통과, 신규실측아직없음
- Next: ATOM2/RNGD3 점유확인 후 B12/B14 mixed84 동일조건 실측, 길이별응답·실제슬롯점유비교
- Experiments: experiments/2026-10-16-b12-b14-mixed84.md
- By: codex@etri-001

### 2026-10-16 04:50+00:00 — PDDISAG-2

- Result: ABBA1472요청exact/ownership0/lane100%/모든서버정상종료. B14두쌍이득13.32%,12.34%,프로세스요약134.265→151.488tok/s(+12.83%). 전체그래프·근거저장
- Next: B14 혼합길이·도착분포에서 refill/길이기반 스케줄링 최적화 및 지연·전력검증
- Metrics: b14_abba_tok_s=151.488, abba_gain_percent=12.827
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:44+00:00 — PDDISAG-2

- Result: B12-r2서버21594 weights loaded·client92697 준비대기. ABBA분석기 오류거부시험추가,68개통과, SDK설정기록과함께c274afa저장
- Next: B12-r2 측정완료·정상종료 후1472요청ABBA검증 및 보고
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:42+00:00 — PDDISAG-2

- Result: B14-r2 중앙151.448tok/s,368요청exact/ownership0/lane100%/정상종료. 이전B14 151.528과근접. 마지막B12-r2서버21594 시작
- Next: B12-r2 client완료 후1472요청ABBA통합검증 및 개선율 정리
- Metrics: b14_repeat_pd84_tok_s=151.448
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:36+00:00 — PDDISAG-2

- Result: B14역순retry3 장치확보·ATOM/RNGD 준비통과(server16087),client54713 readiness대기. ABBA분석기추가·중복run거부확인
- Next: client54713 84요청완료 검증 후 B12-r2 실행,4run순서교차분석
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:35+00:00 — PDDISAG-2

- Result: RNGD32byte probe npu3확보성공. ATOM일시컨텍스트로retry2사전중단·보존,재조회idle후retry3서버16087 및 client연결(sidecar56795)
- Next: retry3 준비/84요청측정 완료 확인 후 B12역순비교 및 ABBA검증
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:33+00:00 — PDDISAG-2

- Result: 읽기전용진단: fusion8/Mesh2/논리3개,이전성공npu3채널기록확인. smi/PE fuser소유자미표시로EBUSY원인미확정. server에할당전SDK설정기록추가
- Next: 추가장치상태진단 및 역순비교 분석기준비; 실제획득 가능 확인 전 하드웨어재시작 보류
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:31+00:00 — PDDISAG-2

- Result: 역순B14 재현2시도 모두 RNGD3 EBUSY로compute acquisition실패(DMA0),실패보존. smi/fuser 소유자미표시·sudo인증없음; 우리sidecar만종료
- Next: 읽기전용 RNGD매핑/점유 진단 및 CPU 비교분석 준비; 장치획득 가능해지면 역순 재현
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:30+00:00 — PDDISAG-2

- Result: 역순84요청 비교 준비: B14-r2서버83135/sidecar31243 시작,4a2a5a7 소스manifest보존. B12-r2동일조건설정 준비
- Next: B14-r2 client 완료 후 검증·정상종료, B12-r2 실행하여 ABBA비교
- Experiments: experiments/2026-10-16-b12-b14-abba84.md
- By: codex@etri-001

### 2026-10-16 04:29+00:00 — PDDISAG-2

- Result: B14/84 중앙151.528 vs B12 133.715tok/s(+13.32%),두구성lane100%.736요청 exact/ownership0/DMA완료, 전서버·sidecar정상종료.48요청 부정결과 유지 및 비교그래프작성
- Next: 역순 B14→B12 프로세스 재현으로 순서효과 확인 후 혼합부하·전력·지연평가
- Metrics: b14_pd84_tok_s=151.528, b14_vs_b12_gain_percent=13.322
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:23+00:00 — PDDISAG-2

- Result: B14/84 서버61189 ATOM준비·RNGDcompile 완료, client26882 readiness대기.84요청 그래프지원 및 queue용량 비교가드 추가·기존결과회귀통과
- Next: B14/84 준비완료 후368요청 측정/검증, B12/84와 비교그래프 생성
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:21+00:00 — PDDISAG-2

- Result: B12/84 중앙133.715tok/s,368요청 exact/ownership0/lane100%/정상종료. B14/84 서버61189 시작 및 readiness 후 client연결
- Next: B14/84 전체실행 완료·로그검증 후 동일84요청 처리량 비교
- Metrics: b12_pd84_tok_s=133.715
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:18+00:00 — PDDISAG-2

- Result: B12/84 warm134.802tok/s·실제12x7배치 확인; 본측정r0 135.313tok/s, 나머지2회 진행
- Next: B12/84 완료 후 전체검증 및 B14/84 순차실행
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:16+00:00 — PDDISAG-2

- Result: B12/84 probe32요청 완료,84요청 warm 진행(client81942). 48요청 비교그래프 생성·검수; 포화분석기 반복별요청수/출력 검증 강화 및 변조거부 확인
- Next: B12/84 전체측정 완료 후 로그검증/종료, B14/84 실행
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:14+00:00 — PDDISAG-2

- Result: B14/context448 코드·224요청검증을50a0b8d에 보존, 단위시험63개통과. B12/84 서버25454 weights loaded, client81942 준비대기(소스manifest보존)
- Next: B12/84 client81942 측정완료/사건검증 후 B14/84 순차시험
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:13+00:00 — PDDISAG-2

- Result: B14 48요청 중앙129.706tok/s,224요청 exact/ownership0/정상종료, lane85.714%. B12 대비 개선미확인. B12 84요청 서버25454 시작(sidecar2294 유지)
- Next: B12 84요청 준비 후 client 실행·검증, 이어 B14 동일84요청 비교
- Metrics: b14_pd48_tok_s=129.706
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:11+00:00 — PDDISAG-2

- Result: B14 48요청 r0=128.343tok/s 단회, 실제배치14/14/14/6 확인. 나머지2회 진행. B12/B14 공통84요청·queue96 설정 및 근거문서 준비
- Next: client23811 종료 후 3회중앙값/전체 로그 검증, 안전종료 후84요청 비교 시작
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:09+00:00 — PDDISAG-2

- Result: B14 P2P probe32요청 완료: 요청별64/64CQE 및 완료iteration logits exact 확인. client23811 warm/본측정 진행. 후속84요청 공배수 비교 도구 준비
- Next: client23811 전체 완료 후 안전종료·전체 로그 검증, B12와48요청 비교 및84요청 포화시험
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:08+00:00 — PDDISAG-2

- Result: B14 FP32 자체이력510/512, maxregret0.03692. P2P서버46475에서 ATOM14슬롯 export/WORKER_READY 및 RNGD weights loaded 확인, host참조 준비 중
- Next: 준비 후 동일48요청 HTTP검증; client/서버 종료 및 전체 DMA/로그 재생 검증
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:06+00:00 — PDDISAG-2

- Result: B14 decode157.538tok/s 중앙, 4회 B12/full host-reference logits exact. FP32검사3758 시작; B14/context448 PD stride/worker 연결 및 경계검증 추가
- Next: FP32검사 완료 후 B14 실제 P2P host parity/수명 검증과 동일 요청 조건 성능 비교
- Metrics: b14_decode_tok_s=157.538
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:04+00:00 — PDDISAG-2

- Result: B14/S448 compile128.83초 및 eager KV할당/decode2step finite 통과. 측정도구 context/비4배수 배치 수정, 단위시험61개 통과. 128토큰 측정57523 시작
- Next: B14 decode측정 완료 및 FP32 품질 확인 후 context448 P2P layout/worker 연결
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:02+00:00 — PDDISAG-2

- Result: Kernel15.011GB와 선택가중치2byte추정15.010GB 일치에 가까워 가중치 주원인 추론. B14/S448 opt-in 추가 후 compile95561 시작
- Next: B14 compile 완료 후 eager 메모리/2step 검증; 통과 시 context 및 비4배수 배치 PD 경로 연결
- Experiments: experiments/2026-10-16-b14-context448.md
- By: codex@etri-001

### 2026-10-16 04:00+00:00 — PDDISAG-2

- Result: 메모리 단계 실측: runtime14.1GiB, KV60개 성공 후61번째 실패(총64개 필요). decode/DMA없음; 메모리 중복 원인은 아직 미확정
- Next: 컴파일 메모리 분류를 확인하고 예약 영역 감소 또는 지원되는 KV 공유 경로 검토
- Experiments: experiments/2026-10-16-b16-context400.md
- By: codex@etri-001

### 2026-10-16 03:59+00:00 — PDDISAG-2

- Result: B16/S448 전체compile 143.95초 성공, eager KV malloc 실패(exit1), decode/DMA 미시작. 종료 후 장치 context 없음 확인
- Next: runtime 예약 메모리와 external KV 중복 진단; B12 기준 유지하며 지원 가능한 메모리 절감 경로 검토
- Experiments: experiments/2026-10-16-b16-context400.md
- By: codex@etri-001

### 2026-10-16 03:53+00:00 — KONI B16 PD 최적화

- Result: S416 전체compile실패보존. 단일attention분리시험:400/416실패,448/512성공(런타임/DMA없음). B16/S448전체compile시작
- Next: S448 전체compile완료 후 eager메모리 적합성 검증
- Experiments: experiments/2026-10-16-b16-context400.md
- By: codex@etri-001

### 2026-10-16 03:49+00:00 — KONI B16 PD 최적화

- Result: B16 S400 compile가 Invalid output channel shape로abort(exit134), 실행/DMA없음·로그보존. 별도S416 정렬후보 compile53255시작; context512기본유지
- Next: S416 compile 결과확인 후 eager메모리/출력검증 또는 제약기록
- Experiments: experiments/2026-10-16-b16-context400.md
- By: codex@etri-001

### 2026-10-16 03:46+00:00 — KONI B16 PD 최적화

- Result: ABBA896요청 전체exact/소유권위반0/정상종료. full의두쌍이득+4.50%,+3.77%, 프로세스중앙요약126.499→131.730tok/s(+4.14%). 포화긴출력의다음실험기준으로선택
- Next: 전체 결과·코드 정리 후 메모리·배치 확대 또는 mixed부하 적용의 다음 병목 최적화
- Metrics: full_abba_tok_s=131.730
- Experiments: experiments/2026-10-16-projection-abba.md
- By: codex@etri-001

### 2026-10-16 03:39+00:00 — KONI B16 PD 최적화

- Result: ABBA3 grouped 중앙126.621tok/s, 직전grouped126.377과유사. 224요청exact/소유권위반0/정상종료. 마지막full 서버54635시작
- Next: full4 완료 후 ABBA 두비교쌍 및 전896요청 검증·통합보고
- Metrics: abba3_grouped_tok_s=126.621
- Experiments: experiments/2026-10-16-projection-abba.md
- By: codex@etri-001

### 2026-10-16 03:33+00:00 — KONI B16 PD 최적화

- Result: ABBA2 grouped 중앙126.377tok/s, 첫full132.070대비 full+4.50%. 224요청exact/소유권위반0/정상종료. ABBA3 grouped 서버55950시작
- Next: grouped3 및 full4 완료 후 두비교쌍·반복변동 통합분석
- Metrics: abba2_grouped_tok_s=126.377
- Experiments: experiments/2026-10-16-projection-abba.md
- By: codex@etri-001

### 2026-10-16 03:27+00:00 — KONI B16 PD 최적화

- Result: ABBA1 full 완료: 132.070/131.897/132.495 중앙132.070tok/s, 224요청exact/소유권위반0/정상종료. grouped2 서버32306시작
- Next: grouped2 완료검증 후 grouped3/full4 이어서 순서교차비교
- Metrics: abba1_full_tok_s=132.070
- Experiments: experiments/2026-10-16-projection-abba.md
- By: codex@etri-001

### 2026-10-16 03:21+00:00 — KONI B16 PD 최적화

- Result: 동일48요청/출력128워밍업 및 full-grouped-grouped-full 순서교차시험 준비. warmup조건불일치거부 추가. 첫server97516/client70023 시작
- Next: ABBA1종료/사건검증 후 grouped2·3, full4 순차실행
- Experiments: experiments/2026-10-16-projection-abba.md
- By: codex@etri-001

### 2026-10-16 03:19+00:00 — KONI B16 PD 최적화

- Result: full projection PD224요청exact/소유권위반0. 처리량119.727/126.833/132.645 중앙126.833vs기존127.249(-0.33%): 전체PD개선미확인, 기본채택보류. 정상종료
- Next: 동일48요청128출력 워밍업 및 graph순서교차로 반복변동/장치구간 확인
- Metrics: full_projection_pd_tok_s=126.833
- Experiments: experiments/2026-10-16-b12-full-projection.md
- By: codex@etri-001

### 2026-10-16 03:13+00:00 — KONI B16 PD 최적화

- Result: Full projection 독립FP32 자체이력510/512일치, maxregret0.03692(응용품질우위아님). 새graph 같은기준P2P+48요청시험 server72679 시작
- Next: 새graph HTTP시험 결과/사건검증 후 기존B12포화127.249tok/s와비교
- Experiments: experiments/2026-10-16-b12-full-projection.md
- By: codex@etri-001

### 2026-10-16 03:11+00:00 — KONI B16 PD 최적화

- Result: B12 full projection decode-only144.151tok/s, 이전138.394대비+4.16%. strict기존graphparity실패(exit1)보존, 전출력finite. 독립FP32검사10572시작
- Next: FP32결과확인 및 같은이력검사 필요성판단 후 새graph PD검증
- Metrics: b12_full_projection_decode_tok_s=144.151
- Experiments: experiments/2026-10-16-b12-full-projection.md
- By: codex@etri-001

### 2026-10-16 03:09+00:00 — KONI B16 PD 최적화

- Result: B12 full projection 컴파일121.058초 성공, eager2step finite통과. 128토큰3회 decode실측 session58535 시작; 이전graph출력차이와 독립품질검증 분리
- Next: 58535 결과확인 및 저장logits의 FP32검사 후 PD시험 여부판정
- Experiments: experiments/2026-10-16-b12-full-projection.md
- By: codex@etri-001

### 2026-10-16 03:05+00:00 — KONI B16 PD 최적화

- Result: 48요청 비교 B8 94.621→B12 127.249tok/s(+34.48%), 실제lane모두100%, 총426요청exact/소유권위반0/정상종료. B12 full-output-projection 별도compile 시작
- Next: full projection compile 결과 후 eager 실행/수치/처리량 독립검증
- Metrics: b12_sat48_tok_s=127.249
- Experiments: experiments/2026-10-16-b8-b12-saturation48.md
- By: codex@etri-001

### 2026-10-16 02:59+00:00 — KONI B16 PD 최적화

- Result: B8 48요청3회 중앙94.621tok/s, 실제슬롯점유100%, 202요청exact/소유권위반0/정상종료. 동일B12 서버88841/client93642 시작
- Next: B12 48요청 완료 후 동조건비교 및 full projection 별도compile 시험
- Metrics: b8_sat48_tok_s=94.621
- Experiments: experiments/2026-10-16-b8-b12-saturation48.md
- By: codex@etri-001

### 2026-10-16 02:56+00:00 — KONI B16 PD 최적화

- Result: B8 48요청 시험 probe/warm 완료, 본측정 진행. 다음 후보full_output_projection의 과거residentB24와 현재external-KV 차이 확인, 명시compile옵션 추가(미실행)
- Next: client51763 완료 후 사건/점유검증, B12 48요청 비교; 이후 single projection 별도artifact 검증
- Experiments: experiments/2026-10-16-b8-b12-saturation48.md
- By: codex@etri-001

### 2026-10-16 02:54+00:00 — KONI B16 PD 최적화

- Result: 48요청×출력128×3회 포화시험 구현, 기존B12 사건재생 회귀검증통과. 장치idle확인 후 B8 server74091/client51763 시작
- Next: B8 완료/실제배치점유검증 후 B12 동일48요청 시험
- Experiments: experiments/2026-10-16-b8-b12-saturation48.md
- By: codex@etri-001

### 2026-10-16 02:52+00:00 — KONI B16 PD 최적화

- Result: B12 FP32 top1 506/512. Eager decode-profile exact 통과, 중앙138.394tok/s, runtime호출83.849/전체85.987ms. 입력/출력 overhead 제한적 확인
- Next: 동일48요청 B8/B12로 부분배치 비용과 포화처리량 분리 검증
- Metrics: b12_decode_tok_s=138.394
- Experiments: experiments/2026-10-16-b12-profile-quality.md
- By: codex@etri-001

### 2026-10-16 02:48+00:00 — KONI B16 PD 최적화

- Result: 동일32요청 B8/B12 비교완료: refill 출력12894.30→110.17(+16.82%), 혼합90.75→100.13(+10.34%). 혼합sealed91.36→82.64(-9.55%). 총1258요청exact/소유권위반0. 서버종료, B12 FP32검사시작
- Next: CPU quality session83620 결과확인 후 품질/처리량근거 통합 및 다음 최적화 선택
- Metrics: b12_refill_fixed128_gain_percent=16.82, b12_refill_mixed_gain_percent=10.34
- Experiments: experiments/2026-10-16-b8-control32.md
- By: codex@etri-001

### 2026-10-16 02:38+00:00 — KONI B16 PD 최적화

- Result: B12 처리량·길이별지연 그래프 생성/시각검수. 혼합 출력32 응답16.907→4.984초, 출력12822.388→22.696초. 독립FP32 품질검사 도구 준비; B8 시험은계속진행
- Next: B8 client57159 완료 뒤 baseline비교, CPU간섭 없이 B12 FP32검사 실행
- Experiments: experiments/2026-10-16-b12-pd-http.md
- By: codex@etri-001

### 2026-10-16 02:37+00:00 — KONI B16 PD 최적화

- Result: B12 640요청 전체exact/소유권위반0/실제refill80건 검증완료. 출력128 sealed109.930/refill110.165, 혼합82.638/100.130tok/s. 서버정상종료. 동일32요청 B8 control 시작
- Next: B8 server7437/client57159 완료확인 후 동조건 비교 및 B12 독립품질 검증
- Metrics: b12_fixed128_refill_tok_s=110.165, b12_mixed_refill_tok_s=100.130
- Experiments: experiments/2026-10-16-b12-pd-http.md
- By: codex@etri-001

### 2026-10-16 02:32+00:00 — KONI B16 PD 최적화

- Result: B12 실측 계속 진행. 고정32요청의 실제 배치12+12+8, 유효 lane88.89% 확인; 부분배치 비용을 포함한 비교 유지
- Next: client66971 완료 후 전 요청 사건검증, B8 동일32요청 baseline 실행
- Metrics: b12_fixed_useful_lane_fraction=0.888889
- Experiments: experiments/2026-10-16-b12-pd-http.md
- By: codex@etri-001

### 2026-10-16 02:30+00:00 — KONI B16 PD 최적화

- Result: B12 HTTP 반복 실측 진행: 20시나리오 중7완료, 첫 sealed 출력128109.022tok/s(단회). 동일32요청 B8 비교설정 및 완료결과 비교도구 준비
- Next: live client66971 완료 확인 후 전체 사건 재생 검사 및 B8 control 실행
- Experiments: experiments/2026-10-16-b12-pd-http.md
- By: codex@etri-001

### 2026-10-16 02:28+00:00 — KONI B16 PD 최적화

- Result: B12 PD 32요청 probe 완료: 전체 logits exact, DMA2048/2048 CQE 정상. HTTP sealed/refill 3회 반복 실측 진행 중
- Next: 실측 session66971 완료 확인, 사건 재생 검사 후 B8 동일32요청 비교
- Metrics: b12_probe_requests=32
- By: codex@etri-001

### 2026-10-16 02:24+00:00 — KONI B16 PD 최적화

- Result: B12 context512 컴파일145.289초 성공. eager 할당+decode2step finite 통과, 관측15.6/15.7GiB. DMA 및 품질 검증은 미수행
- Next: B12 전체128토큰 host 기준 생성 및 P2P/슬롯 교체 검증
- Metrics: b12_compile_seconds=145.289
- Experiments: experiments/2026-10-16-b12-eager-probe.md
- By: codex@etri-001

### 2026-10-16 02:21+00:00 — KONI B16 PD 최적화

- Result: B16 PD eager KV 할당 실패; DMA0, 정상 장치 반환. 실패 원본 보존. context512 유지 B12 컴파일 시작
- Next: B12 컴파일 결과 확인 후 eager 메모리·출력 검증
- Experiments: experiments/2026-10-16-b16-pd-allocation.md
- By: codex@etri-001

### 2026-10-16 02:18+00:00 — KONI B16 PD 최적화

- Result: B16 decode-only 중앙값177.805 대 B8 104.713 tok/s; strict B8 parity 실패와 FP32 진단 별도 보존. opt-in B16 PD 경로 연결, 단위시험61개 통과; P2P 실측 미완료
- Next: B16 동일 배치 host/P2P 출력 검증 및 슬롯 재사용 후 동일32요청 B8/B16 HTTP 비교
- Metrics: decode_b16_tok_s=177.805, decode_b8_tok_s=104.713
- By: codex@etri-001

### 2026-10-16 02:12+00:00 — PDDISAG-1 ✅

- Result: 한국어 STATUS 초기화 완료: 최신 2ATOM B8 sealed 출력128 177.73 tok/s, 혼합 sealed/refill 147.01/117.31 tok/s 및 근거·한계 정리
- Next: 최신 B8/sealed와 동일 카드 예산 단독·복제 기준을 실제 trace에서 비교하고 지연·전력·품질 검증
- By: codex@etri-001
