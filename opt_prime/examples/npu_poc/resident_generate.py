"""NPU execution of the resident cache path (1) -- **generation with resident cache × TP** (PP=1).

    cd opt_prime/examples            # from the repository root (the directory containing npu_poc/)
    TP=2 NEW=32 SEQ=16 python3 -m npu_poc.resident_generate
    (env vars: TP · NEW (tokens to generate) · SEQ (prompt length, <= BLOCK) · GDT=bfloat16)
    ATOM+ only — run in the ATOM+ environment (see the poc.py header and README.md).

`resident.py` rewrites attention into the vendor paged op; this file compiles
the result and runs 1 prefill + N decode steps.  To add PP, use
`resident_pp_generate.py`.  Prefill accepts only prompts up to the block size
(`NPU_POC_BLOCK`, default 128) -- `resident.check_prefill_len` rejects longer
ones.  The decode op correctly includes the new token (cos 0.999996).

Note: **the design is simpler than the functional cache path (`generate.py`)** --
the cache **disappears from the graph**:

    functional   decode(input_ids, position_ids, attention_mask, 32 past K/V)
                 → (logits, 32 updated K/V)      ← host round trip every step
    resident     decode(input_ids, position_ids, seq, block_table)
                 → (logits,)                    ← the cache lives only on the device

No `past_key_values` is passed to the HF model.  The rewritten attention
**reads and writes the resident cache directly**, so the graph only sees the
one new token.  No buckets are needed either -- the shape is always `(1,1)`.

    prompt P <= BLOCK · generate N tokens · TP

Reference: **CPU greedy decoding of the same model and prompt**.  As in
`generate.py`, per-step accuracy is judged by **teacher forcing**; free-running
generation is reported separately because errors amplify.
"""
import contextlib
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # opt_prime/

NEW = int(os.environ.get("NEW", "16"))
SEQ = int(os.environ.get("SEQ", "16"))
TP = int(os.environ.get("TP", "1"))
B = 1
DT = {torch.float32: "float32", torch.bfloat16: "bfloat16",
      torch.int16: "int16", torch.int32: "int32", torch.int64: "int64"}
# Note: default is float32 -- if unset, the path is bit-identical to earlier measurements.
GDT = torch.bfloat16 if os.environ.get("GDT") == "bfloat16" else torch.float32


def main():
    import rebel
    from transformers import AutoConfig, AutoModelForCausalLM
    from npu_poc import capture as CAP, compat, models as M, resident as R
    from npu_poc.backends.rbln import RblnBackend

    MODEL = "meta-llama/Llama-3.2-1B"
    cfg = AutoConfig.from_pretrained(MODEL)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=GDT).eval()
    NL, NH = cfg.num_hidden_layers, cfg.num_attention_heads
    KVH, HD = cfg.num_key_value_heads, cfg.hidden_size // NH
    R.check_prefill_len(SEQ)
    ids = M.text_inputs(MODEL, batch=B, seq=SEQ, n=1)[0]["input_ids"]
    cap_total = SEQ + NEW
    nblk = (cap_total + R.BLOCK - 1) // R.BLOCK
    print(f"[resident] prompt {SEQ} + generate {NEW} · BLOCK {R.BLOCK} · blocks {nblk} · "
          f"TP {TP} · layers {NL} · dtype {DT[GDT]}", flush=True)

    backend = RblnBackend(verbose=False)
    plan = compat.build_plan(backend.caps(), dtype=GDT)

    def build(seq_len, prefill):
        with contextlib.redirect_stdout(io.StringIO()):
            cap = CAP.capture_and_split(model, 1, plan, batch=B, seq=seq_len,
                                        use_kv_cache=True)
        sub = cap.split_gm.get_submodule(cap.stage_names[0])
        st = R.rewrite(sub, kvh=KVH, n_heads=NH, head_dim=HD, batch=B,
                       prefill=prefill, impl="vendor", verbose=False)
        R.prune_kv_outputs(sub, keep=1)
        if st.skipped or st.replaced != NL:
            raise SystemExit(f"[resident] FAIL rewrote {st.replaced}/{NL}")
        return sub, st

    pre, stp = build(SEQ, True)
    dec, std = build(1, False)
    print(f"[resident] prefill: {stp}", flush=True)
    print(f"[resident] decode : {std}", flush=True)

    caches = R.cache_tensors(NL, KVH, HD, nblk, dtype=GDT)
    bt = torch.arange(nblk, dtype=torch.int16).reshape(B, nblk)
    cn = [f"past_key_values_{'kv'[i % 2]}_{i // 2}" for i in range(2 * NL)]
    names = ["input_ids", "position_ids", "seq", "block_table"] + cn

    def spec(n_tok, pos0):
        t = [torch.zeros(B, n_tok, dtype=torch.long),
             torch.zeros(B, n_tok, dtype=torch.long),
             torch.tensor([[pos0]], dtype=torch.int32), bt, *caches]
        return t, [(a, list(x.shape), DT[x.dtype]) for a, x in zip(names, t)]

    for g in (pre, dec):
        ph = [n.target for n in g.graph.nodes if n.op == "placeholder"]
        if ph != names:
            raise SystemExit(f"[resident] FAIL placeholder\n  {ph}\n  {names}")

    ctx = rebel.CompileContext(use_weight_sharing=True)
    for i, t in enumerate(caches):
        ctx.mark_static_address(t, f"kv_cache_{i}")     # shared by both binaries
    kw = {"tensor_parallel_size": TP} if TP > 1 else {}
    tp_, ip_ = spec(SEQ, 0)
    td_, id_ = spec(1, SEQ)
    t0 = time.time()
    cp = rebel.compile_from_torch(pre, input_info=ip_, example_inputs=tp_,
                                  compile_context=ctx, **kw)
    cd = rebel.compile_from_torch(dec, input_info=id_, example_inputs=td_,
                                  compile_context=ctx, **kw)
    comp = time.time() - t0
    dev = list(range(TP)) if TP > 1 else 0
    rp = rebel.Runtime(cp, device=dev, tensor_type="pt")
    rd = rebel.Runtime(cd, device=dev, tensor_type="pt")
    print(f"[resident] compiled {comp:.1f}s · 2 binaries (no buckets — decode is "
          f"always (1,1))", flush=True)

    # ── CPU reference greedy ─────────────────────────────────────────
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=NEW, do_sample=False,
                             use_cache=True,
                             pad_token_id=getattr(cfg, "eos_token_id", None))
    ref_new = ref[0, SEQ:].tolist()

    def gen(force=None):
        pos = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)
        t1 = time.time()
        o = rp.run(ids, pos, torch.tensor([[0]], dtype=torch.int32), bt)
        pre_ms = (time.time() - t1) * 1000
        lg = o[0] if isinstance(o, (list, tuple)) else o
        cur = int(lg[0, -1].argmax().item())
        out, ms, valid = [cur], [], SEQ
        for i in range(NEW - 1):
            if force is not None and i < len(force):
                cur = force[i]
            t1 = time.time()
            o = rd.run(torch.tensor([[cur]], dtype=torch.long),
                       torch.tensor([[valid]], dtype=torch.long),
                       torch.tensor([[valid]], dtype=torch.int32), bt)
            ms.append((time.time() - t1) * 1000)
            lg = o[0] if isinstance(o, (list, tuple)) else o
            valid += 1
            cur = int(lg[0, -1].argmax().item())
            out.append(cur)
        return out, pre_ms, ms

    free, pre_ms, ms = gen()
    n = min(len(free), len(ref_new))
    same = sum(1 for i in range(n) if free[i] == ref_new[i])
    tf, _, _ = gen(force=ref_new)
    tsame = sum(1 for i in range(n) if tf[i] == ref_new[i])
    host = sum(float(c.abs().sum()) for c in caches)
    tok_s = 1000.0 / (sum(ms) / max(len(ms), 1)) if ms else 0.0
    print(f"[resident] prefill {pre_ms:.1f}ms · decode {min(ms):.1f}~{max(ms):.1f}ms "
          f"({len(ms)} steps) · {tok_s:.0f} tok/s", flush=True)
    print(f"[resident] host cache sum {host:.4f}  (0 = on device only — resident)")
    print(f"[resident] teacher-forced match {tsame}/{n} · free-running {same}/{n}")
    ok = tsame >= max(1, int(n * 0.9)) and host == 0.0
    print(f"[resident] RESULT {'PASS' if ok else 'FAIL'} resident-generate "
          f"SEQ={SEQ} NEW={NEW} TP={TP} dt={DT[GDT]} teacher_forced={tsame}/{n}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
