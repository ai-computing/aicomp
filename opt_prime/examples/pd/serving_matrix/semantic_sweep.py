"""EOS-aware GPU/ATOM direct inference using identical serialized input IDs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from semantic_inputs import eos_ids, fixtures, generate_until_stop
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--backend", choices=["gpu", "atom"], required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--fixtures", required=True)
p.add_argument("--cache-dir")
p.add_argument("--kv-blocks", type=int, default=3)
p.add_argument("--limit", type=int, default=64)
a = p.parse_args()
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
if a.backend == "gpu":
    cfg = GenerationConfig.from_pretrained(a.model, local_files_only=True)
    eos = eos_ids(cfg.eos_token_id, tok.eos_token_id)
    data = {"eos_ids": eos, "cases": list(fixtures(tok))}
    Path(a.fixtures).write_text(json.dumps(data, ensure_ascii=False, indent=2))
else:
    data = json.loads(Path(a.fixtures).read_text())
assert data["eos_ids"] and all(isinstance(x, int) for x in data["eos_ids"])
assert all(0 < len(r["ids"]) <= 9216 - a.limit for r in data["cases"])
engine = None
try:
    if a.backend == "gpu":
        model = (
            AutoModelForCausalLM.from_pretrained(
                a.model,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
    else:
        from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

        nodes = list(Path("/dev").glob("rbln[0-9]*"))
        assert len(nodes) == 1
        engine = AtomRblnEngine(
            devices=[int(nodes[0].name[4:])],
            max_seq_len=9216,
            batch_size=1,
            kv_blocks=a.kv_blocks,
            cache_dir=a.cache_dir,
        )
        engine.start(a.model)
        b = engine.batcher(prefix_cache=False)
        model, decoder = b.model, b.decoder
    with torch.inference_mode(), open(a.output, "w", buffering=1) as output:
        for r in data["cases"]:
            ids = r["ids"]
            previous = None
            for repeat in range(2):
                start = time.perf_counter()
                if a.backend == "gpu":
                    out = model(
                        input_ids=torch.tensor([ids], device="cuda"),
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cache = out.past_key_values

                    def step(token):
                        global cache
                        o = model(
                            input_ids=torch.tensor([[token]], device="cuda"),
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                        cache = o.past_key_values
                        assert bool(torch.isfinite(o.logits).all())
                        return int(o.logits[0, -1].argmax())
                else:
                    table = torch.zeros((1, 1), dtype=torch.int16)
                    hybrid = model.rbln_config.cache_impl == "hybrid"
                    extra = {"local_block_tables": table[0]} if hybrid else {}
                    out = model.prefill_decoder(
                        input_ids=torch.tensor([ids]),
                        cache_position=torch.arange(len(ids), dtype=torch.int32)[None],
                        batch_idx=0,
                        block_tables=table[0],
                        **extra,
                    )
                    decoder.dec_attn_mask.zero_()
                    decoder.dec_attn_mask[0, : len(ids)] = 1
                    position = len(ids)

                    def step(token, table=table, hybrid=hybrid):
                        global position
                        pos = torch.tensor([[position]], dtype=torch.int32)
                        extra = {"local_block_tables": table} if hybrid else {}
                        o = decoder(
                            input_ids=torch.tensor([[token]]),
                            cache_position=pos,
                            position_ids=pos.clone(),
                            block_tables=table,
                            **extra,
                        )
                        position += 1
                        assert bool(torch.isfinite(o.logits).all())
                        return int(o.logits[0, -1].argmax())

                assert bool(torch.isfinite(out.logits).all())
                first = int(out.logits[0, -1].argmax())
                ttft = time.perf_counter() - start
                tokens, reason = generate_until_stop(
                    first, step, set(data["eos_ids"]), a.limit
                )
                assert previous is None or tokens == previous, "repeat mismatch"
                previous = tokens
                row = {
                    "case": r["case"],
                    "backend": a.backend,
                    "repeat": repeat,
                    "input_tokens": len(ids),
                    "input_sha256": hashlib.sha256(
                        json.dumps(ids).encode()
                    ).hexdigest(),
                    "tokens": tokens,
                    "stop_reason": reason,
                    "text": tok.decode(tokens, skip_special_tokens=True),
                    "ttft_s": ttft,
                    "e2e_s": time.perf_counter() - start,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "tokens"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
finally:
    if engine is not None:
        engine.close()
