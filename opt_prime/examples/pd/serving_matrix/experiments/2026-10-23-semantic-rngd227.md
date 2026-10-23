# Four-model semantic RNGD prefill handoff

- **Date**: 2026-10-23
- **Task**: aicomp #227, parent #201
- **Host / device**: exclusive RNGD and ATOM allocations on RNGD+ATOM host; RNGD128GiB memory limit, fusion4/Mesh2, Furiosa2026.4; ATOM64GiB, Rebel0.11/Optimum0.11.0.post1. GPU A10080GB BF16 SDPA.
- **Commit**: e16c256 plus semantic producer/import-checker changes in exp/227-semantic-matrix.

## Question

Can the exact short chat fixtures from the native GPU/ATOM semantic suite run through RNGD prefill and resume to EOS on both decoder types without exporting padded state?

## Setup

Reuse the pinned model revisions and exact saved token IDs/EOS sets from the preceding semantic-eos227 record. Select capital-0, arithmetic-0, korean-0 and retrieval-0. Gemma/Qwen compiled width128; KONI partitioned width512, RMS tile512,group1. Gemma selects hidden state n-1 and exports only n KV positions. Qwen and KONI reuse their existing true-length export paths. Three finite/repeat-exact prefill runs per case. GPU decoder checker and both ATOM importers now stop on exported EOS IDs, maximum64 outputs. Gemma additionally verifies native raw-state restoration.

## Command

```bash
bash run-gemma223.sh --width 128 --fixtures semantic227-gemma-fixtures.json --cases capital-0 arithmetic-0 korean-0 retrieval-0 --output semantic227-gemma-rngd128
bash run-qwen227.sh --model Qwen/Qwen3-0.6B --width 128 --fixtures semantic227-qwen06b-fixtures.json --cases capital-0 arithmetic-0 korean-0 retrieval-0 --output semantic227-qwen06b-rngd128
bash run-qwen227.sh --model Qwen/Qwen3-4B --width 128 --fixtures semantic227-qwen4b-fixtures.json --cases capital-0 arithmetic-0 korean-0 retrieval-0 --output semantic227-qwen4b-rngd128
bash run-koni219.sh --width 512 --partitioned --group 1 --rms-tile 512 --fixtures semantic227-koni-fixtures-eos.json --cases capital-0 arithmetic-0 korean-0 retrieval-0 --output semantic227-koni-rngd512
# Each export stem produces -0.pt through -3.pt, containing the true input IDs and EOS IDs.
python check_rngd_gpu.py --model "$GPU_MODEL" --exports "$EXPORT0" "$EXPORT1" "$EXPORT2" "$EXPORT3" --eos-aware --max-output 64 --output "$GPU_REPORT"
bash run-atom-handoff227.sh --model "$MODEL_ID" --cache-dir atom-artifacts --kv-blocks "$BLOCKS" --exports "$EXPORT0" "$EXPORT1" "$EXPORT2" "$EXPORT3" --eos-aware --max-output 64 --output "$ATOM_REPORT"
# Gemma uses run-gemma-atom223.sh and atom-artifacts-global225 instead.
```

## Results

- CPU extraction test fails before helper implementation, then passes. Full tiny Gemma oracle now covers unpadded and right-padded inputs; combined selected equation/cache/export/EOS suite20passes.
- Gemma:4/4 native/imported EOS continuations match on GPU and ATOM;52slot readback and native-state/logit roundtrip exact.
- Qwen0.6B and4B:4/4 GPU continuations match;4/4 ATOM native matches for both importers. Importer logits exact;56 and72 slots read back exactly respectively.
- KONI:4/4 EOS outputs match GPU and ATOM native; both importers preserve64slots and yield identical decoder logits. Input lengths137–143, width512.
- All four producers:48 finite/repeat-exact runs. Exact saved input IDs and true KV lengths verified for all16exports. All32 model/case/decoder comparisons match through EOS. Native GPU and ATOM sequences also match for these16 short fixtures.
- Warm prefill+host-export ranges: Gemma44.09–52.09ms; Qwen0.6B43.13–46.80ms; Qwen4B94.08–96.22ms; KONI228.33–234.85ms. Different input lengths/compiled widths and concurrent host activity mean these are not normalized engine comparisons or controlled serving benchmarks.
- First KONI GPU checker invocation had a mistyped fourth export path and failed with FileNotFoundError after the first three checks. Corrected path rerun completes all four. Keep the original failure log; no compiler or model change was needed.
- All hardware jobs terminal; owned RNGD/ATOM Pods released. Scoped Ruff/format checks pass;20CPU tests pass.

Evidence: ../evidence227/semantic227-rngd-summary.json, semantic227-*-rngd-gpu.json, semantic227-*-rngd-atom.json and producer jsonl. Large tensor exports remain local and can be regenerated from committed fixtures and commands.

## Conclusion

Short diverse prompts now reach GPU/ATOM through RNGD without forced post-EOS output. This preserves existing native-model behavior, including shared wrong answers from the previous record; it does not prove factual quality. Long semantic cases, production PD/cache/cancellation, batching and real compute/transfer overlap remain open. These are full-prefix exports through host staging, not overlapped streaming.
