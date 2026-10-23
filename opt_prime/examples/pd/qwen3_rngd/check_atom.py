import json
from pathlib import Path

import numpy as np
import torch
from atom_stream_upload import import_atom_span
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "Qwen/Qwen3-0.6B"
physical = int(next(Path("/dev").glob("rbln[0-9]*")).name[4:])
e = AtomRblnEngine(
    devices=[physical],
    max_seq_len=512,
    batch_size=1,
    kv_blocks=3,
    cache_dir=str(Path("qwen3-197-atom").resolve()),
)
e.start(mid)
records = torch.load("qwen3_197_kv.pt", weights_only=True)
ref = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(mid)
b = e.batcher(prefix_cache=False)
rows = []


def decode(first, n):
    out = [first]
    vectors = []
    for j in range(15):
        z = (
            b.decoder(
                input_ids=torch.tensor([[out[-1]]]),
                cache_position=torch.tensor([[n + j]], dtype=torch.int32),
                block_tables=b.blocks.table([0]),
            )
            .logits[0, -1]
            .float()
        )
        assert torch.isfinite(z).all()
        vectors.append(z.clone())
        out.append(int(z.argmax()))
    return out, torch.stack(vectors)


with torch.inference_mode():
    for r in records:
        ids = r["ids"]
        n = len(ids)
        b.blocks.admit(0, ids)
        block = b.blocks.lane_block[0]
        nl = b._prefill(0, ids).logits[0, -1].float()
        native, _ = decode(int(nl.argmax()), n)
        hf = ref(torch.tensor([ids]), use_cache=True)
        hl = hf.logits[0, -1].float()
        ll = r["logits"].float()
        ck = torch.stack([l.keys[0] for l in hf.past_key_values.layers])
        cv = torch.stack([l.values[0] for l in hf.past_key_values.layers])
        kr = float(
            (r["keys"].float() - ck).square().mean().sqrt() / ck.square().mean().sqrt()
        )
        vr = float(
            (r["values"].float() - cv).square().mean().sqrt()
            / cv.square().mean().sqrt()
        )
        variants = {}
        saved = []
        for name in ["original", "portable"]:
            if name == "original":
                b.blocks.import_span(0, 0, r["keys"], r["values"])
            else:
                import_atom_span(b.blocks, 0, 0, r["keys"], r["values"])
            exact = 0
            for meta, buf in zip(b.blocks.metas, b.blocks._staging()):
                expected = buf[block, :, :n].copy()
                buf.fill(0)
                b.blocks.io_rt.get_kv_cache_for_layer(meta["name"], block, 0, n, buf)
                assert np.array_equal(expected, buf[block, :, :n])
                exact += 1
            tokens, vectors = decode(int(ll.argmax()), n)
            saved.append(vectors)
            variants[name] = {
                "tokens": tokens,
                "text": tok.decode(tokens),
                "readback_exact_slots": exact,
                "native_equal": tokens == native,
            }
        row = {
            "prompt": r["prompt"],
            "prompt_tokens": n,
            "first_cpu": int(hl.argmax()),
            "first_rngd": int(ll.argmax()),
            "first_atom": int(nl.argmax()),
            "key_relative_rms": kr,
            "value_relative_rms": vr,
            "logits_relative_rms": float(
                (ll - hl).square().mean().sqrt() / hl.square().mean().sqrt()
            ),
            "prefill_s": r["prefill_s"],
            "native_tokens": native,
            "native_text": tok.decode(native),
            "variants": variants,
            "importer_logits_exact": bool(torch.equal(*saved)),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        b.blocks.release(0, None)
Path("qwen3_197_atom.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
)
e.close()
