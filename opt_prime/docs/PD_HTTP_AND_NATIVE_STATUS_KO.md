# HTTP 취소·semantic KV 및 native P2P 상태

2026-10-19. 기존 RNGD→ATOM external-cache P2P와 llm-d를 확장했다.

## HTTP 계약

각 stage의 localhost HTTP endpoint에 `POST /v1/cancel`을 추가했다.
`x-request-id`는 취소할 요청의 ID다. 실행/큐 대기 중이면 HTTP 스레드는 해당
CancellationToken만 설정하며 장치나 버퍼를 조작하지 않는다.

- 202 + `drained:false`: 취소 요청 접수. 장치 작업 완료/자원 해제를 의미하지 않는다.
- 200 + `drained:true`: owner가 완료된 prepared prefill을 해제했다.
- 원래 completion 요청의499: 동기 완료 경계에서 취소되어 응답 생성 중단.
- 잘못된 ID400, 일치하는 준비 요청 없음409, owner 미가동503.
- 이미 완료와 경합한 취소는 best-effort다. 202가 항상499를 보장하지 않는다.
  동시에 여러 논리 요청에 같은 ID를 재사용하지 않는다.
- 연결 끊김/timeout은 자동 취소가 아니다. 취소는 명시적으로 요청한다.
- transport/engine 완료가 불확실하면500 및 poisoned 보존. 강제 device abort 없음.

```bash
curl -X POST http://127.0.0.1:DECODE_STAGE_PORT/v1/cancel \
  -H 'x-request-id: your-request-id'
```

이는 Optimus stage의 확장 API다. upstream llm-d proxy가 이 endpoint를 자동 전달한다고
주장하지 않는다. 일반 요청의 sidecar 연결과 stage 직접 취소를 각각 검증했다.

## Semantic KV

`KVSpec.semantic_id`를 추가했다. Qwen adapter는 DenseModel의 revision·geometry·
RoPE 등 manifest를 정규화해 SHA256으로 식별한다. 물리 shape가 같아도 지문이 다르면
PDRuntime이 거부한다. 기존 adapter는 빈 값으로 호환 유지한다. 가중치 파일 전체를
새로 해싱하거나 모든 모델의 semantic codec을 구현한 것은 아니다.

## 실측

1.5B direct16요청320토큰896DMA 전체 logits exact, llm-d8요청160토큰 exact.
실제 HTTP 취소:56DMA/917504bytes 완료 후202 접수,499 종료, decode 생략,
동일 lane 후속 요청 토큰 일치.52CPU회귀 및 정적 검사 통과.
근거: `validation/pd-http/q15/result.json`, `http-cancel/result.json`.

## Native P2P 감사와 기존 설명 보완

공식 최신 문서에는 `nixl-rbln` DMA-BUF/RDMA 경로가 있다. 따라서 제조사에 공식
경로가 전혀 없다고 일반화하면 안 된다. 이 기능은 `vllm` 구현에 한정되며,
구현별 장치 표에서는 Rebel100을 대상으로 한다. 현재 ATOM Optimum 정적 캐시
및 RNGD PDMA와의 직접 호환성은 확인되지 않았다.

출처: [PD 기능](https://docs.rbln.ai/latest/software/model_serving/vllm_support/features/disaggregated-prefill-decode.html),
[구현별 장치·설치](https://docs.rbln.ai/latest/software/model_serving/vllm_support/index.html).

로컬 확인: vllm-rbln0.11.0, rebel0.11.0, nixl-rbln/nixl 없음. 기존 설치 커넥터는
host staging 코드를 제공한다. 기본 index에서는 nixl-rbln 배포를 찾지 못했고,
제조사 index 다운로드는 인증 요구/키링 실패로 완료하지 못했다. 설치 변경 없음.
`rdma link show`와 `/sys/class/infiniband`에 장치 없음. GPU driver도 접근 불가.
근거: `validation/pd-http/native-audit.json`.

Native ATOM P2P는 **미완료**다. 필요한 것은 단순 SDK 버전 증가가 아니라 현재
ATOM에서 사용 가능한 캐시 export 계약·해당 패키지·엔진 지원 및 RNGD 연동 증거다.
기존 custom/external-cache P2P는 계속 지원한다. 전체 GPU/NPU, 멀티슬롯 overlap,
분산 취소·재시작 복구 및 모든 모델 확장은 이번 검증 범위를 넘는다.
