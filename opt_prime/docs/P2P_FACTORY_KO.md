# P2P worker/factory 라이브러리화

2026-10-19. 통합 설계의 첫 구현 단위. 기존 external-cache P2P 경로를 유지한다.
Native ATOM 내부 캐시 export, 범용 capability registry, reserve-first와 취소/drain은
후속 단계이며 이 변경에 포함하지 않는다.

## 변경

- `opt_prime.pd.npu.factory.create_qwen_p2p`: 준비된 RNGD engine/binding/packer/backend와
  ATOM worker를 조합해 `P2PSession`을 반환한다. `session.runtime`이 기존 `PDRuntime`이다.
- `P2PWorkerClient`: gdb capture 기반 실험적 exporter 시작, UNIX socket FD 수신,
  모델/프로토콜/FD 크기 검사, 정상 종료를 담당한다. vendor SDK를 import하지 않는다.
- `opt_prime.pd.npu.worker`: 기존 examples worker를 라이브러리의 모듈 실행으로 이동했다.
  참조 모듈이 없으면 host reference 명령을 거부하며, 전체 logits 이력을 저장하지 않는다.
- `examples/pd/qwen_reference.py`: 테스트용 host admission과 full-logit oracle만 담당한다.
  신뢰하는 로컬 config의 `validation_hook`으로만 선택한다. 운영 config에는 넣지 않는다.
- `QwenAtomWorkerEngine`: 요청 ID/증가하는 generation/완료/출력 길이를 확인한다.
  검증 모드에서는 기존 reference_checked 및 max_abs_diff==0도 필수다.
- 기존 `examples/pd/qwen_atom_worker.py`는 호환 진입점으로 남겼다.

## 사용 흐름

```python
from opt_prime.pd.npu.factory import create_qwen_p2p

session = create_qwen_p2p(
    config_path, rngd_engine, rngd_binding, kv_packer, dma_backend, worker_directory
)
# 성공한 요청은 DMA completion 및 decode 완료까지 동기적으로 실행된다.
tokens = session.runtime.generate(request)
# 실패한 lane은 강제 close하지 않는다. poisoned 상태의 소유권을 보존한다.
session.close()
```

Config는 기존 `qwen_pd_config.example.json`의 artifact/worker/장치 설정을 사용한다.
`socket`은 사용하지 않는 UNIX socket 경로, worker_directory는 신규 디렉터리다.
`decode_engine`은 atom_external, `transport`는 p2p만 허용한다(생략 시 이 값).
정상 종료는 반복 호출 가능하며 active/poisoned runtime은 close를 거부한다.
초기화 실패 시 DMA는 제출되지 않았지만 exporter 프로세스가 남을 수 있다.
`startup-failure.json`과 worker log의 PID를 확인해야 하며 자동 장치 reset은 없다.

소스 SDK·artifact·codec 초기화는 여전히 caller 책임이다. factory는 현재 Qwen
external-cache 구성용이며 모든 NPU/모델에 대한 자동 factory가 아니다.
라이브러리 내부는 기존 별도 `hetero-npu-p2p`를 사용한다. 호스트 fallback은 없다.
GDB/live-allocation/pinned-thunk 의존도 아직 제거되지 않았다.

## 검증

- Qwen0.5B: 직접16요청320토큰, DMA768개, host-reference 전체 logits exact.
- Qwen1.5B: 직접16요청320토큰, DMA896개, host-reference 전체 logits exact.
- Qwen1.5B llm-d sidecar:8요청160토큰, direct와 토큰 exact, 정상 종료.
- 운영 모드: 별도 결과 `validation/p2p-factory/production/result.json` 참조.
  worker oracle 없이 이전 검증 run의 토큰과 비교하며 full-logit 검증과 구별한다.
- SDK 없는 factory import, 지원되지 않는 native P2P 거부, poisoned close 차단,
  응답 request ID 및 reference 정책 회귀 포함42개 테스트 통과.

기존 핫패스를 추출한 변경이다. 새 속도 개선율·native P2P·일반 품질 보장은 주장하지 않는다.
