# PD capability, lease identity와 협력적 취소

2026-10-19. 기존 P2P factory 위의 다음 증분 구현이다.

## 구현한 계약

- `EngineCapabilities`: prefill/decode 역할, 입력/context 한도, 명시적 transport
  집합, 검증된 external-KV 여부. Qwen factory에서 P2P 조합을 협상한다.
  지원되지 않는 조합은 host fallback 없이 거부한다. 레거시 adapter의 capability는
  선택 항목으로 유지한다. 전체 SDK/장치 자동 발견 registry는 아직 아니다.
- `KVLease.bind`: request ID, 양수 generation, lane UUID를 불변 lease에 연결한다.
  이미 다른 요청/세대/lane에 바인딩된 lease는 거부한다. descriptor/owner를 복사하거나
  메모리를 재할당하지 않는다. worker RPC의 기존 request ID/generation 검사도 유지한다.
- `PDRuntime`: 목적지를 reserve한 다음 prefill한다. reserve/precompute 오류 후에는
  보수적으로 poisoned 상태와 소유권을 보존하며 자동 예약 해제/재시도하지 않는다.
- `CancellationToken`: 실행 전에 취소되었으면 장치 작업 없이 종료한다. 이미 시작된
  prefill/transfer는 동기 반환까지 진행한다. transfer fence 이후 취소가 관찰되면
  decode를 생략하고 transport.finish 후 lease를 해제한다. decode 중 취소는 decode가
  정상 반환한 뒤 finish한다. 결과는 `RequestCancelled`로 전달한다.

```python
from opt_prime.pd.runtime import CancellationToken, RequestCancelled

token = CancellationToken()
# 다른 스레드에서 token.cancel() 가능. 토큰은 제출마다 새로 만든다.
try:
    tokens = session.runtime.generate(request, cancellation=token)
except RequestCancelled:
    # 정상 취소는 runtime.state == 'idle'; 같은 lane을 다시 쓸 수 있다.
    pass
```

취소가 요청되어도 DMA 오류/worker 오류/finish 실패를 정상 취소로 숨기지 않는다.
불확실한 completion은 poisoned로 남으며 버퍼를 유지한다. device abort/reset,
실행 중 token 단위 중단, 취소 시간 상한을 제공하지 않는다. `finish`는 동기 계약의
정상 작업을 마무리하는 drain 경계이며 별도 비동기 장치 drain API 구현은 아니다.

llm-d의 기존 prepared-prefill 취소/TTL은 그대로다. 새 Python cancellation token을
HTTP 중단/timeout에 연결하지 않았다. 실제 분산 취소 프로토콜과 재시작 후 lease
복구, 전체 semantic/physical KV 분리, native-cache P2P는 후속 작업이다.

## 검증 범위

- 47 CPU 회귀: SDK-free 계약, 지원 조합/용량 거부, stale/foreign lease 거부,
  reserve-before-prefill, transfer 중 취소 후 finish/재사용, 불확실 DMA 격리,
  실행 전 취소.
- 1.5B 실제 P2P16요청320토큰896DMA, 전체 logits exact. llm-d8요청160토큰 exact.
- 실제 취소 시험은 `validation/pd-lifecycle/cancel/result.json`에 기록한다.
  완료된 DMA 직후 cancellation을 주입하며 decoder generation이 증가하지 않고
  lane이 idle로 돌아오는지 확인한 뒤16요청을 다시 실행한다.
- 변경된 시간 항목: `prefill_seconds`는 prefill만, `reserve_validate_seconds`는
  선행 reserve와 후행 검사 합. end-to-end는 벽시계로 별도 측정한다.

검증은 동기 single-flight/B1 경로다. 멀티슬롯/비동기 엔진·GPU·다중 노드까지
지원한다고 해석하지 않는다. 속도 개선이나 품질 개선을 주장하는 변경도 아니다.
