# Optimus Prime 이종 PD 통합 설계안

2026-10-19. 상태: 구현 계획. 현재 코드 기준 `72ce576` 이후 문서 변경.
이 문서의 제안 API·디렉터리·기능은 구현 완료를 뜻하지 않는다.

## 1. 결정 요약

Optimus Prime을 **실행 경로 선택·요청 스케줄링·엔진 조합의 주체**로 둔다.
독립 `hetero-npu-p2p`는 **모델을 모르는 전송 및 버퍼 수명 계약 라이브러리**로
유지한다. 제조사 런타임은 계산·내부 캐시·내부 배치를 소유하며, Optimus
어댑터가 지원되는 기능만 공통 계약으로 노출한다.

주력 구현은 이미 검증된 RNGD custom → P2P → ATOM external-cache 경로다.
Native ATOM은 별도 engine 종류로 유지한다. 캐시 export/visibility/lifetime이
검증되기 전에는 native 엔진을 P2P 가능으로 광고하지 않는다. Host 경로는
명시적으로 선택하는 별도 transport다. P2P 실패 시 조용히 host로 전환하지 않는다.

새 저장소를 여러 개 더 만들지 않는다. 당장은 기존 두 패키지에 책임을 정리하고,
제조사 어댑터 배포주기가 실제로 분리될 때만 추가 패키지 분리를 검토한다.

## 2. 현재 구현과 간극

| 현재 코드 | 확인한 상태 | 통합에 필요한 변경 |
|---|---|---|
| `opt_prime/inference.py`, `opti_pri.py`, `schedule_infer.py` | GPU/Torch pipeline 실행, CUDA mesh 경로 포함 | 기존 API 유지. 새로운 공통 진입점에서 별도 실행 backend로 선택 |
| `opt_prime/pd/contracts.py`, `runtime.py` | SDK-free 계약과 단일 진행 요청의 PD 수명 관리 | capability, 명시적 lease ID/generation, abort/drain 계약 추가 |
| `opt_prime/pd/npu/engines.py` | RNGD custom 및 ATOM worker 어댑터 | SDK·모델·RPC 책임을 나누고 worker launch를 examples에서 library로 이동 |
| `opt_prime/pd/hetero_p2p.py` | 독립 P2P 라이브러리의 CopySession 연결 | 검증된 export/import descriptor만 허용하는 factory 연결 |
| `opt_prime/pd/npu/native_atom.py` | B1/context512, host KV 쓰기와 native decode 정합성 검증 | admission/decode 분리, 요청별 소유권과 종료를 공통 계약에 맞춤 |
| `opt_prime/pd/llmd.py` | stage 서비스 및 owner 실행, 기존 경로 sidecar 검증 | 공통 service factory 이용. 엔진별 분기와 SDK import 금지 |
| `examples/pd/qwen_atom_worker.py`, `validate_qwen_pd.py` | P2P 구동·참조 비교·측정이 혼합됨 | 운영 worker와 실험 harness 분리 |
| `hetero_p2p/atom_experimental.py` | pinned thunk/live allocation capture 기반 export | experimental backend로 격리. 공식 지원 API처럼 표현하지 않음 |
| `npu/shared_kv.py` | 호스트 memfd 전달용 모듈 | 선택형 진단/host transport. P2P 구현으로 취급하지 않음 |

기존 GPU PP와 PD는 같은 개념이 아니다. PP는 모델 레이어 분할, PD는 요청의
prefill/decode 단계 분리다. 향후 각 PD endpoint 내부가 PP/TP를 쓰는 것은
가능하지만 현재 GPU pipeline을 바로 외부 KV endpoint로 쓸 수 있다는 근거는 없다.
먼저 whole-request backend로 감싸고, KV 경계 노출 후 PD endpoint를 검증한다.

## 3. 라이브러리 경계와 의존 방향

| 계층 | 소유할 기능 | 소유하지 않을 기능 |
|---|---|---|
| `hetero-npu-p2p` | allocation descriptor, 전송 plan, 범위·정렬 검사, DMA backend, completion receipt, transfer lease | tokenizer, 모델 config, 요청 배치 정책, HTTP, logits, 품질 판정 |
| `opt_prime.accel.pd` core | 모델/KV 계약, capability 협상, 요청 상태, endpoint 선택, admission·decode·release 조율 | vendor SDK import, 임의 주소 탐색, DMA ioctl 구현 |
| Optimus model adapters | checkpoint identity, head/layer/RoPE/layout 계약, KV codec·지원 검사 | 전송 장치 제어, HTTP 라우팅 |
| Optimus engine adapters/workers | compile artifact 검증, vendor 실행, cache reserve/export/import/visibility, 내부 batch 연계 | 전역 라우팅, 독립적인 중복 배치 스케줄러 |
| Optimus service adapters | llm-d stage/HTTP, request ID·취소·응답/추후 streaming | 캐시 주소·FD 해석, 모델 계산 |
| examples/tests/benchmarks | 실행 예제, full-logit oracle, 품질·성능 실험 | 운영에서 매 요청 참조 추론 강제 실행 |

의존 방향: service → Optimus core → engine/transport contracts.
구체 transport adapter → hetero-npu-p2p. 구체 engine worker → vendor SDK.
P2P 라이브러리가 Optimus를 import하는 역방향 의존은 만들지 않는다.

기존 `hetero-npu-p2p` 안의 KONI/llm-d/스케줄링 코드를 바로 삭제하지 않는다.
먼저 Optimus 쪽 정책 구현의 정본을 정하고 호환 wrapper·deprecation을 둔 뒤
중복을 단계적으로 제거한다. ABI exporter는 장치 기능이므로 모델 코드와 분리한다.

## 4. 제안 모듈 구조

```text
opt_prime/
  execution.py                 # 제안: public factory, lazy backend selection
  inference.py                 # 기존 GPU pipeline API 보존
  pd/
    contracts.py               # request, semantic KV, lease, capabilities
    runtime.py                 # fenced lifecycle / failure quarantine
    registry.py                # 제안: 명시 등록, 지원 조합 검사
    planning.py                # 제안: engine/model/transport compatibility
    engines/
      rngd_custom.py           # 기존 RNGD graph adapter 이동
      atom_external.py         # 검증된 P2P destination
      atom_native.py           # native host 우선, export 검증 전 P2P 금지
      torch.py                 # GPU/CPU 참조; 실제 검증 수준 표시
    models/
      dense.py                 # 공통 semantic manifest
      qwen2.py, koni.py         # 모델별 codec/artifact policy
    transports/
      p2p.py                   # hetero-npu-p2p 이용
      host.py                  # 명시적 host admission, memfd 등
    workers/
      protocol.py              # bounded RPC, lease ID/generation, completion
      supervisor.py            # 각 SDK Python/컨테이너 lifecycle
    serving/
      llmd.py                  # control-plane stage adapter
    metrics.py                 # 공통 측정 경계와 실제 route 기록
```

위 구조로 한 번에 파일을 옮기지 않는다. 기능별 PR에서 옮기고 기존 import 경로를
재노출한다. 공통 패키지 import만으로 torch/CUDA/NPU를 초기화하지 않는다.
SDK는 서로 충돌하므로 worker마다 Python/컨테이너와 패키지 lock을 분리한다.

## 5. 계약: 모델 의미와 실제 메모리를 구분

현재 `KVSpec`은 모델과 물리 layout을 함께 검사한다. 이를 호환성을 유지하면서
아래 세 수준으로 확장한다.

1. **SemanticKVSpec**: checkpoint revision/hash, architecture, layer/head 순서,
   RoPE 설정, 위치 의미, context, dtype/quantization 정보. 같은 모델 이름만으로
   호환이라고 판정하지 않는다.
2. **PhysicalKVSpec**: encoding(BF16/DLFloat 등), 실제 shape/stride, block size,
   shard mapping, alignment, tensor order. 서로 다르면 등록된 codec이 필요하다.
3. **KVLease**: request ID, lease ID, generation, endpoint instance ID, capacity와
   valid prefix, owner reference, opaque descriptor, completion 상태.

Capabilities는 선언과 런타임 검증을 함께 사용한다. 최소 항목:
`prefill`, `decode`, `host_admit`, `device_export`, `device_import`,
`external_write_visible`, `static/paged_cache`, `batch_limit`, `context_limit`,
`dtype/layout`, `cancel/drain`, 검증 SDK/driver 범위.

Source export만으로 P2P 가능이라고 결정하지 않는다. 정확한 방향의 backend,
destination import/export 계약, PCIe/장치 binding, codec, completion까지 모두
검사한다. 장치 두 개가 같은 노드에 있다는 사실만으로 direct DMA를 약속하지 않는다.
현재 RNGD→ATOM은 **ATOM destination export 후 RNGD write**라는 방향별 계약이다.

Native ATOM의 `set_kv_cache_for_layer(host_tensor)`는 host admission 능력이다.
`device_export`를 뜻하지 않는다. export handle·KV offset/layout·외부 write 반영·
수명 고정이 모두 확인되어야 native P2P adapter를 활성화한다.

## 6. 요청 실행과 소유권

```text
client / llm-d control request
  → Optimus validates model + chooses an explicit route
  → decode worker reserves destination lease
  → prefill worker computes prefix and retains source lease
  → transport submits copies / verifies completion
  → decode worker makes external writes visible
  → decode starts at the validated token position
  → decode completion / drain
  → release exports and source/destination leases
```

이것은 목표 순서다. 현재 runtime은 prefill 후 reserve한다. 목적지 부족 시 낭비를
줄이기 위해 reserve-first를 별도 변경으로 도입하고, prefill 실패 시 예약 해제를
검증한다. cache 데이터는 llm-d HTTP를 지나가지 않는다. 같은 노드의 FD는 UNIX
socket SCM_RIGHTS로 넘기고 JSON에는 raw 포인터를 전달하지 않는다. 장치 주소는
해당 owner/backend가 검증한 범위 안에서만 해석한다.

Single-flight/단일 슬롯을 먼저 보존한다. 슬롯 풀과 이중 버퍼는 요청별 lease가
완성된 뒤 추가한다. 소스는 DMA 완료까지, destination은 decode 완료까지 유지한다.
취소는 즉시 free가 아니다. 큐 제거 → 제출 작업 fence/drain → lease 해제 순서다.
worker 사망·timeout·불확실한 completion은 quarantine 처리한다. 같은 allocation을
재할당하거나 generation을 재사용하지 않는다. 격리/복구 범위는 실제 장치 context
소유권에 따라 정의하며 자동 장치 reset은 하지 않는다.

## 7. 제조사 엔진과 llm-d 연결 정책

- 제조사 엔진이 continuous batching을 제공하면 Optimus는 요청을 전달하고
  placement/backpressure를 담당한다. 제조사 내부 token scheduler를 외부에서
  동시에 조작하지 않는다.
- external-cache custom 엔진은 Optimus가 검증된 sealed batch/iteration 정책을
  사용한다. native와 custom의 batch capability를 같은 값으로 가정하지 않는다.
- llm-d 연결은 control-plane adapter다. transfer receipt와 request lease를
  연결하고 decode-ready가 확인된 요청만 전달한다.
- 현재 B1 native probe는 private production worker가 아니다. `generate()`를
  `reserve/admit/decode/release`로 분리하고 reference 비교는 harness로 옮긴다.
- Furiosa native connector, Mooncake 등은 별도 선택 adapter 후보다. 현재 custom
  RNGD graph/P2P 경로를 그 기능으로 구현했다고 표시하지 않는다. 새 connector는
  문서·설치 API 확인과 실제 양쪽 endpoint 호환시험 후 지원 목록에 추가한다.

제안 설정(아직 지원되는 CLI가 아님):

```yaml
mode: pd
model: {id: pinned-model, revision: exact-revision}
prefill: {engine: rngd_custom, worker: rngd-worker-0}
decode: {engine: atom_external, worker: atom-worker-0}
transport: {kind: hetero_p2p, required: true, fallback: disabled}
serving: {adapter: llmd}
```

Host/native 조합은 별도 설정에 `atom_native`와 `host`를 명시한다. 지원 불가능한
`atom_native + required P2P`는 startup 오류로 처리한다. 자동 선택은 지원 행렬과
측정된 비용 모델이 생긴 이후에만 도입하고, 실제 선택 경로를 항상 노출한다.

## 8. 구현 순서와 완료 기준

| 단계 | 작업 | 완료 판정 |
|---|---|---|
| 1 — 최우선 | 기존 P2P worker/launcher를 library factory로 이동, source/target 선택과 manifest 검사를 공통화 | 기존 KONI/Qwen 검증 조건에서 실제 DMA receipt·full-logit parity 유지, SDK 없는 core import 가능 |
| 2 | capabilities 및 semantic/physical spec, request-bound leases, reserve-first·abort/drain 도입 | stale generation, 잘못된 revision/layout, worker failure, 취소에서 오사용/조기 해제 없음 |
| 3 | 기존 P2P 핫패스 최적화: persistent worker/session/입력, 검증된 packing·배치, 불필요한 참조 추론 제거 | 동일 요청·카드·정밀도 A/B/B/A, 정합성 후 실제 지연/처리량 개선. 느려진 결과도 기록 |
| 4 | llm-d stage factory를 공통 경로에 연결, backpressure·취소 연결 | 직접 실행 및 sidecar 결과 동일, route별 trace와 bounded queue 확인 |
| 5 — 병행 조사 | native cache export 계약 확인. 공식/검증 가능한 adapter가 있을 때만 P2P 확장 | live allocation 소유권·범위·completion·visibility 검증. 불충족이면 native host-only 유지 |
| 6 | 추가 모델/GPU/backend와 슬롯풀·전송/계산 overlap | 각각 capability matrix 통과, 요청 혼입·품질·메모리 압박 회귀 통과 |

우선 단계1~2를 작게 구현한다. 네이티브 P2P 연구가 막혀도 검증된 외부 캐시
P2P 경로의 제품화·최적화는 진행할 수 있다. 큰 batch/멀티카드 지원을 B1 지원으로
추론하지 않고, 전용 시험을 통과한 조합만 활성화한다.

## 9. 검증·성능 보고

- 동일 입력/출력/도착 분포, 카드 수, checkpoint, 정밀도, 컴파일 옵션을 명시한다.
- reference 경로의 전체 logits와 전송 bytes/완료 수를 검증한다. FP32 품질·응용
  정확도는 별도 항목이다. 운영 핫패스에서 reference decode를 반복하지 않는다.
- prefill, codec, transfer submission/completion, visibility, decode, queue,
  TTFT/TPOT·p99·SLO goodput을 분리한다. host wall time을 device kernel time으로
  표시하지 않는다. overlap이 생기면 구간 합을 end-to-end로 쓰지 않는다.
- P2P, host, standalone을 구분하며 카드 예산이 다른 비교는 명시한다. MFU와 전력은
  합의된 FLOPs/peak/동시 측정 구간이 확보된 경우에만 보고한다.
- 최적화 실험은 워밍업과 순서 교차를 포함하고 원시 결과·실패·환경 manifest를
  남긴다. 소규모4prompt 개선을 전체 모델·부하의 개선으로 일반화하지 않는다.
- GPU, 타 모델, 다른 SDK/드라이버, 다중 노드는 개별 검증 전까지 planned로 둔다.

## 10. 바로 검토할 변경 단위

첫 PR의 범위는 `P2PWorkerClient + factory + 기존 examples의 위임`으로 제한한다.
검증 harness의 oracle 기능은 examples에 남기고 라이브러리는 토큰 생성/완료만
담당하게 한다. 기존 public import를 유지하고 native/host 기능을 자동 선택하지
않는다. 두 번째 PR에서 capability·lease 계약을 확장한다.

참고: [코드 지도](CODE_REVIEW_GUIDE_KO.md), [native 검증](NATIVE_NO_PRIMER.md),
[host 실험과 P2P 구분](NATIVE_SHARED_DIAGNOSTIC.md).
