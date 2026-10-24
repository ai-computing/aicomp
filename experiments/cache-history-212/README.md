# Prefix-cache page geometry and request history (#212)

## Background

On the four selected checkpoints, prefix-cache reuse can change greedy output relative to prior requests. A preceding512/2048 homogeneous request matrix did not reproduce the problem: Qwen3-0.6B page64/page256 C0+ABBA controls completed220responses with100/100 token and100/100 SSE comparisons equal to cache-off. That pass did not cover the original128-token partially warmed prefix.

## Setup and criterion

Runtime a3497e57ac9d8286ad20032db4c842f26272e961 (PR266), one A10080GB PCIe, driver580.178.04, torch2.13.0+cu130, Transformers5.15.0. This evidence branch is based on main fd8926565217fdc675705cdf9d4477c1b0dbd57f; measurements use the explicitly named experimental runtime, not main. Frozen checkpoints: Qwen3-0.6B, Qwen3-4B, Gemma3-1B and KONI-Llama3.1-8B-Instruct-20241024; exact revisions in summary.json.

BF16, four execution slots, context9216, prefill chunk256, prefix cache enabled. First collect serial128/2048/8192-token baselines; then mixed-length bursts of4 and8 requests, output32, greedy, ignore EOS. Validate HTTP readiness, completion counts, cancellation/disconnect recovery and empty-prompt rejection separately. All token comparisons below are independently recomputed from saved arrays. Throughput is not the correctness criterion.

Only page size changes between configurations. Qwen0.6 runs64,256, then256,64 after other model pairs; each other model runs64 then256 once. Fresh server/process per arm, cached history retained within each arm. No competing GPU benchmark. Actual prefix-hit counters must be positive. Ten child exits and coordinator states checked; GPU compute-process list empty afterward.

## Results

|Model|Page|Concurrent equals own serial|Serial equals archived cache-off|Prefix-hit tokens|Mixed c8 wall(s)|
|---|---:|---:|---:|---:|---:|
|qwen06b|64|8/12|2/3|35072|1.498|
|qwen06b|256|12/12|3/3|32768|1.445|
|qwen4b|64|8/12|1/3|35072|1.961|
|qwen4b|256|12/12|3/3|32768|1.956|
|gemma|64|12/12|2/3|35072|1.761|
|gemma|256|12/12|3/3|32768|1.811|
|koni|64|7/12|2/3|35072|1.667|
|koni|256|12/12|1/3|32768|1.781|
|qwen06b|256|12/12|3/3|32768|1.513|
|qwen06b|64|8/12|2/3|35072|1.465|

Qwen0.6 page64 reproduces the archived symptom: four2K responses diverge at zero-based output index1, token374 versus594, and all three serial baselines equal the archived cache-enabled run. Page256 removes the discrepancy in this fixture. The reverse-order repeat checks reproducibility. Qwen4 and KONI also require separate within-route and cache-off comparisons; consult summary.json for each actual first divergence.

Gemma page64 now has12/12 within-run matches, while its2K serial output differs from archived cache-off at index24, after the first EOS at index2. The archived cache-enabled experiment had8/12 matches. This changed observation must not be represented as a reproduced Gemma within-run failure or as general answer-quality improvement. Page256 comparisons remain scoped to these saved arrays.

Current-runtime cold controls clarify the KONI historical mismatch: cold page64 and page256 match3/3, and page256 cache-on matches current cold3/3. Both current cold configurations differ from archived128/2K references. Thus that historical difference is not attributable to page256; its exact environmental/runtime cause remains unisolated. Page64 cache-on still differs from current cold at2K.

All four models additionally complete current page256/cache-off controls:12/12 concurrent matches each, zero cache hits, and cache-on/cache-off serial baselines match3/3 each. Across the10cache-on arms and5cache-off controls, all15HTTP lifecycle checks pass; numerical comparisons are180concurrent responses with17mismatches, all in page64 cache-on arms (Qwen0.6 twice4, Qwen4 four, KONI five). These current controls use the same checkpoint, runtime and environment, rather than depending on archived comparisons.

This supports evaluating page256/chunk256 as a workload-specific mitigation candidate. It does not establish a general numerical fix. Coarser pages reduce token reuse (32768 versus35072hit tokens in this fixture) and can change memory/recompute costs. Burst times are single measurements with uncontrolled execution order for three models; they are not stable speedup estimates. No production defaults or engine code were changed.

## Reproduction

Use validate.py from runtime a3497e57ac9d8286ad20032db4c842f26272e961 at opt_prime/examples/pd/http_matrix/validate.py, SHA2563d400c279f338fdba8e1f3f97d7f83da21271908116360279e1b0ca400de1798. The local diagnostic copies changed only the hard-coded page size and the import path to serve_batched. A portable equivalent is:

```bash
export RUNTIME=/path/to/runtime-checkout
export PAGE=64 # repeat with256
export COPY=/path/to/new-validation-copy.py
python3 - <<'PYCODE'
import os,pathlib,hashlib
src=pathlib.Path(os.environ['RUNTIME'])/'opt_prime/examples/pd/http_matrix/validate.py'
s=src.read_text()
assert hashlib.sha256(src.read_bytes()).hexdigest()=='3d400c279f338fdba8e1f3f97d7f83da21271908116360279e1b0ca400de1798'
assert s.count('        page_size=64,')==1
s=s.replace('        page_size=64,',f"        page_size={int(os.environ['PAGE'])},")
s=s.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))','')
with open(os.environ['COPY'],'x') as f:f.write(s)
PYCODE
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$RUNTIME/opt_prime:$RUNTIME/opt_prime/examples/pd" "$PYTHON" "$COPY" --model "$FROZEN_MODEL" --prefix-cache --output "$NEW_RESULT"
```

Set PYTHON to the torch/Transformers environment above and FROZEN_MODEL to the frozen checkpoint. Use an unused output path for every arm; keep originals. No HTTP module override. Raw result filenames/hashes and first divergences are in summary.json; site-specific commands/logs/configs remain local.

## Remaining acceptance work

#212 stays open. Generalization requires different partial-prefix lengths, shared/diverging prefixes, cache eviction/host reload, chunk/page combinations, all-model numerical controls and answer-quality checks. Aligning page and chunk sizes does not mathematically guarantee identical computation for arbitrary prompt histories. Engine-owned changes require the maintainer's implementation and regression review. Related #201; this is evidence, not a completed four-model GPU/RNGD/ATOM certification.
