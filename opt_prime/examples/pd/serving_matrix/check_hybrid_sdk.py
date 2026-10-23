"""Compare the hybrid table probe with SDK-owned tables on the same compiled graph."""

import argparse
import json
from pathlib import Path

import gemma_causal_compat  # noqa: F401 - process-local upstream fix
import torch
from hybrid_tables import install
from inputs import exact_prompt
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cache-dir", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
model = "unsloth/gemma-3-1b-it"
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
pre = [tok.bos_token_id] if tok.bos_token_id is not None else []
fill = tok.encode(
    "This is background context. The red box contains apples. The blue box contains books.\n",
    add_special_tokens=False,
)
tail = tok.encode(
    "\nQuestion: What is the capital of France?\nAnswer:", add_special_tokens=False
)
nodes = list(Path("/dev").glob("rbln[0-9]*"))
assert len(nodes) == 1
engine = AtomRblnEngine(
    devices=[int(nodes[0].name[4:])],
    max_seq_len=9216,
    batch_size=1,
    kv_blocks=3,
    cache_dir=a.cache_dir,
)
rows = []
try:
    engine.start(model)
    b = engine.batcher(prefix_cache=False)
    decoder = b.decoder
    install(b)
    with torch.inference_mode():
        for n in [128, 512, 2048, 8192]:
            ids = exact_prompt(pre, fill, tail, n)
            vectors = []
            tokens = []
            for mode in ["sdk", "external"]:
                if mode == "sdk":
                    out = b.model.prefill_decoder(
                        input_ids=torch.tensor([ids]),
                        cache_position=torch.arange(n, dtype=torch.int32)[None],
                        batch_idx=0,
                    )
                else:
                    b.blocks.admit(0, ids)
                    out = b._prefill(0, ids)
                values = []
                generated = []
                for step in range(32):
                    z = out.logits[0, -1].float().clone()
                    assert torch.isfinite(z).all()
                    values.append(z)
                    token = int(z.argmax())
                    generated.append(token)
                    if step == 31:
                        break
                    pos = torch.tensor([[n + step]], dtype=torch.int32)
                    kw = {
                        "input_ids": torch.tensor([[token]]),
                        "cache_position": pos,
                        "position_ids": pos.clone(),
                    }
                    if mode == "external":
                        kw["block_tables"] = b.blocks.table([0])
                    out = (decoder if mode == "sdk" else b.decoder)(**kw)
                vectors.append(torch.stack(values))
                tokens.append(generated)
                if mode == "external":
                    b.blocks.release(0, None)
            rows.append(
                {
                    "input_tokens": n,
                    "logits_exact": torch.equal(*vectors),
                    "max_abs_difference": float((vectors[0] - vectors[1]).abs().max()),
                    "tokens_equal": tokens[0] == tokens[1],
                    "tokens": tokens,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
finally:
    engine.close()
Path(a.output).write_text(json.dumps(rows, indent=2) + "\n")
