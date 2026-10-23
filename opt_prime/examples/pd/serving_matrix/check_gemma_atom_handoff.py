"""Exclusive B1 Gemma hybrid-cache roundtrip, then RNGD producer handoff."""

import argparse
import json
from pathlib import Path

import gemma_causal_compat  # noqa: F401 - validated process-local SDK compatibility
import numpy as np
import torch
from atom_stream_upload import bf16_to_dlfloat
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from rebel.kv_cache import aligned_tensor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache-dir", required=True)
p.add_argument("--exports", nargs="+", required=True)
p.add_argument("--output", required=True)
p.add_argument("--eos-aware", action="store_true")
p.add_argument("--max-output", type=int, default=32)
a = p.parse_args()
devices = list(Path("/dev").glob("rbln[0-9]*"))
assert len(devices) == 1
engine = AtomRblnEngine(
    devices=[int(devices[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=3,
    cache_dir=a.cache_dir,
)
rows = []
try:
    engine.start("unsloth/gemma-3-1b-it")
    b = engine.batcher(prefix_cache=False)
    model = b.model
    decoder = b.decoder
    rc = model.rbln_config
    assert rc.cache_impl == "hybrid" and rc.batch_size == 1 and rc.sliding_window == 512
    metas = rc.kvcache_metas
    assert all(isinstance(m, dict) and m["dtype"] == "float32" for m in metas)
    assert len(metas) == 52
    runtime = model.model[1]
    buffers = []
    for meta in metas:
        count = int(np.prod(meta["shape"]))
        size = runtime.get_kv_cache_size_for_layer(meta["name"])
        assert size == count * 2, (meta["name"], size, count)
        buffers.append(
            aligned_tensor(
                count, dtype=np.uint16, alignment=2 << 20, tensor_type="np"
            ).reshape(meta["shape"])
        )
    global_table = torch.zeros((1, 1), dtype=torch.int16)
    local_table = torch.zeros((1, 1), dtype=torch.int16)

    def decode(first, n):
        decoder.dec_attn_mask.zero_()
        decoder.dec_attn_mask[0, :n] = 1
        tokens = [first]
        vectors = []
        for j in range(a.max_output - 1):
            if tokens[-1] in eos:
                break
            pos = torch.tensor([[n + j]], dtype=torch.int32)
            out = decoder(
                input_ids=torch.tensor([[tokens[-1]]]),
                cache_position=pos,
                position_ids=pos.clone(),
                block_tables=global_table,
                local_block_tables=local_table,
            )
            z = out.logits[0, -1].float().clone()
            assert bool(torch.isfinite(z).all())
            vectors.append(z)
            tokens.append(int(z.argmax()))
        return tokens, torch.stack(vectors) if vectors else torch.empty((0,))

    def restore(arrays, n):
        exact = 0
        for meta, buf, data in zip(metas, buffers, arrays):
            kept = min(n, meta["shape"][2])
            assert data.shape == tuple(meta["shape"][1:2] + [kept] + meta["shape"][3:])
            assert np.isfinite(data).all()
            buf[0, :, :kept] = data
            runtime.set_kv_cache_for_layer(meta["name"], 0, 0, kept, buf)
            buf.fill(0)
            runtime.get_kv_cache_for_layer(meta["name"], 0, 0, kept, buf)
            assert np.array_equal(buf[0, :, :kept], data)
            exact += 1
        return exact

    with torch.inference_mode():
        for path in a.exports:
            r = torch.load(path, weights_only=True)
            eos = set(r["eos_ids"]) if a.eos_aware else set()
            assert not a.eos_aware or eos
            assert a.max_output > 0
            n = len(r["ids"])
            assert r["keys"].shape == r["values"].shape == (26, 1, n, 256)
            assert 0 < n <= 9216 - a.max_output
            out = model.prefill_decoder(
                input_ids=torch.tensor([r["ids"]]),
                cache_position=torch.arange(n, dtype=torch.int32)[None],
                batch_idx=0,
                block_tables=global_table[0],
                local_block_tables=local_table[0],
            )
            first_logits = out.logits[0, -1].float().clone()
            first = int(first_logits.argmax())
            saved = []
            incoming = []
            errors = []
            for i, (meta, buf) in enumerate(zip(metas, buffers)):
                kept = min(n, meta["shape"][2])
                runtime.get_kv_cache_for_layer(meta["name"], 0, 0, kept, buf)
                native = buf[0, :, :kept].copy()
                saved.append(native)
                value = (
                    r["keys" if i % 2 == 0 else "values"][i // 2, :, -kept:]
                    .to(torch.bfloat16)
                    .contiguous()
                )
                encoded = bf16_to_dlfloat(
                    value.view(torch.int16).numpy().view(np.uint16)
                )
                incoming.append(encoded)
                errors.append(float(np.mean(encoded == native)))
            native_tokens, native_logits = decode(first, n)
            roundtrip_slots = restore(saved, n)
            rt_tokens, rt_logits = decode(first, n)
            assert native_tokens == rt_tokens and torch.equal(
                native_logits, rt_logits
            ), "native state roundtrip differs"
            imported_slots = restore(incoming, n)
            tokens, logits = decode(int(r["logits"].argmax()), n)
            different = next(
                (i for i, (x, y) in enumerate(zip(native_tokens, tokens)) if x != y),
                None,
            )
            divergence = None
            if different is not None:
                ref = first_logits if different == 0 else native_logits[different - 1]
                got = r["logits"] if different == 0 else logits[different - 1]
                divergence = {
                    "native_top5": [
                        {"token": int(i), "logit": float(ref[i])}
                        for i in ref.topk(5).indices
                    ],
                    "imported_top5": [
                        {"token": int(i), "logit": float(got[i])}
                        for i in got.topk(5).indices
                    ],
                }
            row = {
                "divergence": divergence,
                "input_tokens": n,
                "export": Path(path).name,
                "case": r.get("metadata", {}).get("case"),
                "eos_ids": sorted(eos),
                "native_roundtrip_exact": True,
                "roundtrip_slots": roundtrip_slots,
                "imported_slots": imported_slots,
                "slot_encoded_equal_fraction": errors,
                "native_tokens": native_tokens,
                "pd_tokens": tokens,
                "tokens_equal": tokens == native_tokens,
                "first_difference": different,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
finally:
    engine.close()
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
