# Gemma long-prefix producer and ATOM hybrid-state handoff

- **Date**: 2026-10-23
- **Task**: aicomp #223, parent #201; DraftPR224 follow-up
- **Host / device**: RNGD+ATOM host; separate exclusively owned RNGD and ATOM Pods. RNGD128GiB host-memory limit, fusion4/Mesh2, Furiosa2026.4; ATOM64GiB host-memory limit, Rebel0.11/Optimum0.11.0.post1, B1/context9216/k3 hybrid artifact.4requested/16limited CPUs each.
- **Commit**: baseline90e4a7d; exp/223-gemma-rngd working tree. Producer code unchanged from validated #223.

## Question

Does Gemma preserve its local/global cache semantics at8K? Can all52 ATOM cache slots be restored exactly and receive RNGD KV, including512-position local state, global full-prefix state and decoder mask/absolute positions?

## Setup

Same pinned Gemma checkpoint and causal-prefill artifact as the preceding records. Source configuration listsfloat32 cache metadata, local slots[1,1,512,256] and global slots[3,1,9216,256]. Actual runtime physical size is two bytes per element for all52slots. The experiment verifies this for every allocation; it does not assume source metadata describes physical storage. Reuse the existing validated BF16-to-DLFloat encoding for incoming RNGD KV and test it on hardware. Both tables select exclusively owned block/lane0. Before every continuation reset the decoder prompt mask and pass absolute cache_position/position_ids.

First run native prefill, read all52 raw cache slots, generate32tokens, restore the exact native raw state, and require all31 decoder-logit vectors/tokens to match. Then import RNGD state, using the latest min(prompt,512) tokens for local slots and the whole prompt for global slots. Read back all52slots before decode. Compare native/PD outputs, without treating raw transport success as numerical equivalence.

## Command

```bash
bash run-gemma223.sh --width 8192 --output rngd-gemma-1b-8192-223
bash run-gemma-atom223.sh --cache-dir atom-artifacts --exports rngd-gemma-1b-128-223-fixed.pt rngd-gemma-1b-2048-223-fixed.pt --output rngd-gemma-atom223-raw16.json
```

The ATOM launcher uses the existing container_bootstrap and check_gemma_atom_handoff.py in the same isolated environment as the earlier matrix. No shared SDK or other workload is modified.

## Results

- Initial metadata-driven staging fails before inference: allocating float32 by the physical byte count cannot reshape65536elements to[1,1,512,256]. Preserve this failed run.
- Explicit physical16-bit staging: native raw-state roundtrip at128/2048 passes52slot readback and all31 decoder-logit vectors exactly. This verifies state restoration, including local cache ordering and mask positions, for these fixtures.
- RNGD import:52slots read back exactly at both lengths. At2K the32-token PD output equals the prior GPU native/imported output. At128 the first5tokens agree with GPU but fixed-length output later diverges. Native ATOM vs imported differs at generated index1 at both lengths; do not claim cross-engine numerical equivalence. The selected outputs deliberately continue beyond EOS, so raw-token parity is a capacity fixture, not a chat-quality score.
-8K RNGD completes: initialization783.381s, full26-layer prefill+export1.553709/1.547400/1.552140s,3finite/bit-exact repeats. GPU BF16 native/imported32tokens match;52cache slots exact with22local511/4global8192positions. Retained K/V/logit relativeRMS8.1114%/5.6739%/1.4114%.8K is faster than the2K compiled plan in these runs; profile before attributing the difference.
-8K ATOM: native raw-state roundtrip exact for52slots/all31logit vectors, RNGD52slots read back exact. Native/imported output diverges at index2; imported output must be compared with the GPU result separately. Native logit108=20.90625 vs107=20.53125; imported107=20.40625 vs108=20.34375. This is not a tie.
-GPU FP32 references at128/2K also yield32-token native/imported matches; retained K/V/logit relativeRMS0.687/0.725/0.966% at128 and4.539/2.806/1.279% at2K.
-**Native reference invalidation:** follow-up #225 proves the existing ATOM artifact violates prefix causality after its first global attention layer. Changing only the final64tokens of a128-token prompt leaves layers0–5prefixKV exact but changes about99% of prefixKV from layer6onward. Therefore previous native-versus-PD mismatches are not safely attributable to rounding or quality differences. Global AttentionOp/FlashAttentionOp text-prefill flags are now covered by upstream issue831; a fresh fixed artifact is compiling. Retain these baseline results as failure evidence, not a valid causal reference.

Private logs: ../evidence225/rngd-gemma-atom223.log (failed float32 staging), rngd-gemma-atom223-raw16.log and rngd-gemma-1b-8192-223.log. Raw encoded-bit equality fractions are recorded only as diagnostics, not relative floating-point errors.

## Conclusion

Hybrid ATOM raw-state restoration and RNGD handoff now execute at128/2K under exclusive ownership. Production scheduler integration, prefix/host state restoration, larger coverage, numerical divergence analysis and transfer overlap remain open. Full four-model goal remains active.

## Follow-up provenance

The RNGD8K allocation was released after its terminal run. The owned ATOM allocation is retained for #225's fresh-artifact validation. Exact upstream-dev global forward methods pass48operator-boundary cases after2failures/46passes on the installed implementation; combined global/sliding suite60passes. This verifies flag selection, not the pending rebuilt hardware result. Upstream: RBLN-SW/optimum-rbln#831 and draftPR827; downstream aicomp#225.
