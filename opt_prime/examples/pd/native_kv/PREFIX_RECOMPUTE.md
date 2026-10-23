# GPU prefix recomputation and native suffix transfer (#162, in progress)

Parent d3d54af / PR161. The native exporter can omit a prefix using the control
request start-position field. The API below supports bounded suffix receive; PREFIX_OVERLAP.md now documents a
working single-request GPU-prefix/native-suffix prototype and its measurements. Full
transfer remains the serving default; rolling serving is unchanged.

```python
packet, stats = client.prefill(model, tokens, 'views', start_pos=prefix_length)
```

start_pos must be an integer, excluding bool, in [0,len(tokens)-2]. The packet owns
len(tokens)-1-start_pos positions. Destination block IDs restart at0; the source suffix
starts at start_pos. Preserve all existing request-ID, slot, payload, final-frame and
ACK validation. No missing prefix is silently filled: a caller must supply and validate
the omitted prefix before this cache can drive decode. Stats report both start_pos and
exported_positions. Invalid offsets fail before network submission. The existing
observed full-request wire fixture remains byte-identical.

## Validation

Same hardware/model/FXB/software as NATIVE_EXPORT_PROFILE.md. Source hashes in
suffix_api_manifest.json. Full suite52 tests passes, including exact shorter-packet
assembly in views/fast/legacy modes, ACK contents, offset bounds and unchanged full
wire compatibility. Separate targeted protocol/receiver suite20 passes.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
# Start the dedicated default native endpoint as in GPU_LOCAL_PAIRED.md.
python3 probe_suffix_transfer.py --out suffix_api_hardware.json
```

Real RNGD:128/64,512/256,1024/512 input/skipped-prefix pairs all return suffixes byte-equal
to the corresponding full-native KV in all72 slots. These are independent executions
on synthetic repeated-token prompts. Full requests run first and include cold costs;
latencies in the artifact are not an end-to-end speedup claim. Owned server stopped.

## Remaining work

The overlap prototype is implemented and measured in PREFIX_OVERLAP.md. Integrate a
bounded opt-in serving path, including joint prefix/suffix ownership, chunked GPU work,
active-decode interference and cancellation/overload validation. The prototype alone
does not establish native HTTP performance. Parent diagnostic manifests describe the
parent version; suffix_api_manifest.json identifies this API's initial validation and
prefix_overlap_manifest.json identifies the current prototype.
