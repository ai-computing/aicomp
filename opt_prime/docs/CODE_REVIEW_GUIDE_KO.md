# 이종 PD 코드 검토 안내

현재 작업 저장소는 `aicomp-llmd`, 브랜치는 `feat/optimus-llmd`입니다.
`opt_prime/opt_prime/pd/`가 Optimus Prime에 추가한 PD 라이브러리입니다.
루트 실험 디렉터리는 실행 산출물이며 라이브러리 진입점이 아닙니다.

## 먼저 볼 코드

| 순서 | 파일 (`opt_prime/` 기준) | 확인할 내용 |
|---|---|---|
| 1 | `opt_prime/pd/npu/native_atom.py` | 새 네이티브 ATOM 경로. `validate_slot_config`, `NativeAtomDecodeLane.generate`: 버전·캐시 규격 검사, BF16→DLFloat 변환, 네이티브 KV 쓰기, 명시적 블록/마스크, 실패 시 poisoned 상태 |
| 2 | `tests/test_pd_native_atom.py` | 긴 요청 다음 짧은 요청의 mask 초기화, 잘못된 KV 거부, 쓰기 실패 시 버퍼 보존 |
| 3 | `examples/pd/validate_native_no_primer.py` | 실제 ATOM 검증. 첫 요청 전 prefill 금지, 이후 참조 방식과 전체 vocabulary logits 비교 |
| 4 | `examples/pd/native_rngd_prefix_worker.py` | 별도 Furiosa SDK 프로세스에서 요청마다 실제 RNGD prefill. 현재는 NPZ 파일을 이용한 호스트 전달 검증 도구 |
| 5 | `opt_prime/pd/contracts.py`, `runtime.py` | 모델 revision·물리 KV 규격, 수명, prefill→reserve→transfer→decode→finish 계약 |
| 6 | `opt_prime/pd/npu/engines.py`, `hetero_p2p.py` | 기존 커스텀 NPU 엔진과 P2P 전송을 공통 런타임에 연결하는 어댑터 |
| 7 | `opt_prime/pd/llmd.py` | llm-d stage 서비스, 요청/취소 상태, 소유 스레드 실행과 HTTP 경계 |

## 제조사 제공 기능과 우리 코드

- ATOM 제조사: 컴파일된 모델, Optimum decode runtime, `get/set_kv_cache_for_layer`, decoder의 block table·attention mask 인자.
- 우리 코드: RNGD KV의 포맷 변환과 검증, 두 SDK 프로세스의 연결, 명시적인 native decode 상태 준비, 소유권·실패 처리, 이종 전송 및 llm-d 연계 어댑터.
- RNGD의 이번 prefill은 Furiosa SDK 위의 `npu/rngd.py` 커스텀 Qwen 그래프입니다. **Furiosa-LLM 전체 네이티브 엔진을 연결한 것은 아닙니다.**

## 현재 두 경로를 구분해서 검토

기존 경로: RNGD 커스텀 prefill → 직접 P2P → ATOM 커스텀 decode → 공통 PD/llm-d.

새 검증 경로: RNGD 커스텀 prefill → 호스트 NPZ → ATOM 제조사 native decode.
새 경로는 prefill 중복 실행을 제거했지만 아직 `PDRuntime`/llm-d 엔진 선택에
연결하지 않았습니다. 파일 전달은 정합성 검증용이며 고성능 전송 구현이 아닙니다.

## 중점 검토 항목

- native lane은 B1/context512/static 1블록 및 고정 SDK 버전만 허용합니다. 이 제한을 풀려면 별도 검증이 필요합니다.
- 동일 모델 객체를 외부 scheduler와 동시에 사용하면 안 됩니다. lane 내부 lock만으로 외부 호출을 막지는 못합니다.
- 요청 토큰과 KV의 의미적 대응은 호출자가 보장합니다. shape 검사는 다른 요청의 KV 혼입을 검출하지 못하므로 운영 연결에는 request ID/revision 결합이 필요합니다.
- 실패한 슬롯은 자동 복구하지 않습니다. 런타임을 재생성해야 하며 native P2P import/export는 아직 포함하지 않습니다.
- 전체 logits 일치는 동일 RNGD KV에 대한 실행 경로 동등성입니다. FP32 품질이나 단독 NPU 대비 성능 우위를 의미하지 않습니다.
- 다음 통합에서 호스트 전달을 명시적 transport로 분리하고, admission과 decode를 분리하여 기존 `KVLease` 및 finish 계약과 맞춰야 합니다.

결과: [사전 prefill 제거](NATIVE_NO_PRIMER.md), [실시간 RNGD 연결](NATIVE_LIVE_VALIDATION.md).
