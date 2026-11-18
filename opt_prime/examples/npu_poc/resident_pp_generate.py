"""NPU execution of the resident cache path (2) -- **PP × TP × generation.**

    cd opt_prime/examples            # from the repository root (the directory containing npu_poc/)
    PP=2 TP=2 NEW=16 SEQ=16 python3 -m npu_poc.resident_pp_generate   # needs PP×TP cards
    ATOM+ only — run in the ATOM+ environment (see the poc.py header and README.md).

`resident_generate.py` is PP=1.  This file combines it with PP generation (`generate.py`).

    opt_prime split → each stage keeps the cache for its own layers resident →
    each stage compiled with TP → **the top-level FX forward is the pipeline**

Two preconditions confirmed by measurement:
  · **prefill (seq=S) and decode (seq=1) split identically** -- per-stage SDPA
    counts and K/V output indices match → **the two graphs can share the cache**.
  · stage outputs mix `[boundary values..., K/V...]` → `prune_outputs_at` removes
    only the K/V positions, and the wrapper **puts dummies back** to keep the
    top-level `getitem` indices valid.

Stage boundary inputs are **captured with hooks during a CPU run** -- building
them by hand breaks the experimental control.

    PP=2 × TP=2  (4 cards = PP × TP)
"""
import contextlib
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # opt_prime/

PP = int(os.environ.get("PP", "2"))
TP = int(os.environ.get("TP", "2"))
SEQ = int(os.environ.get("SEQ", "16"))
NEW = int(os.environ.get("NEW", "16"))
B = 1
DT = {torch.float32: "float32", torch.int16: "int16",
      torch.int32: "int32", torch.int64: "int64"}
STATE = {"seq": 0}            # changes every step -- read by the wrapper


def main():
    import rebel
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM
    from npu_poc import capture as CAP, compat, models as M, placement
    from npu_poc import resident as R
    from npu_poc.backends.rbln import RblnBackend

    MODEL = "meta-llama/Llama-3.2-1B"
    cfg = AutoConfig.from_pretrained(MODEL)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32).eval()
    NH, KVH = cfg.num_attention_heads, cfg.num_key_value_heads
    HD = cfg.hidden_size // NH
    R.check_prefill_len(SEQ)
    backend = RblnBackend(verbose=False)
    caps = backend.caps()
    stage_devs = placement.assign(PP, TP, None, caps)
    ids = M.text_inputs(MODEL, batch=B, seq=SEQ, n=1)[0]["input_ids"]
    nblk = (SEQ + NEW + R.BLOCK - 1) // R.BLOCK
    bt = torch.arange(nblk, dtype=torch.int16).reshape(B, nblk)
    print(f"[resident-pp] PP={PP} × TP={TP} · prompt {SEQ} + generate {NEW} · "
          f"BLOCK {R.BLOCK} · placement {stage_devs}", flush=True)
    plan = compat.build_plan(caps, dtype=torch.float32)

    def capture(seq_len, x, p):
        """Capture and split, then **record stage boundary inputs/outputs with hooks during a CPU run**."""
        with contextlib.redirect_stdout(io.StringIO()):
            cap = CAP.capture_and_split(model, PP, plan, batch=B, seq=seq_len,
                                        use_kv_cache=True)
        inp, out = {}, {}
        hs = []
        for n in cap.stage_names:
            m_ = cap.split_gm.get_submodule(n)
            hs.append(m_.register_forward_pre_hook(
                lambda mod, a, n=n: inp.__setitem__(n, [t.clone() for t in a])))
            hs.append(m_.register_forward_hook(
                lambda mod, a, o, n=n: out.__setitem__(
                    n, [t for t in (o if isinstance(o, (list, tuple)) else [o])])))
        with torch.no_grad():
            ref = cap.split_gm(x, p)
        for h in hs:
            h.remove()
        kv = {n: [i for i, t in enumerate(out[n])
                  if t.dim() == 4 and t.shape[1] == KVH and t.shape[3] == HD]
              for n in cap.stage_names}
        tot = {n: len(out[n]) for n in cap.stage_names}
        return cap, inp, kv, tot, (ref[0] if isinstance(ref, (list, tuple)) else ref)

    pos = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)
    cpf, inpf, kvf, totf, ref_logits = capture(SEQ, ids, pos)
    cpd, inpd, kvd, totd, _ = capture(
        1, torch.zeros(B, 1, dtype=torch.long),
        torch.zeros(B, 1, dtype=torch.long))
    if kvf != kvd:
        raise SystemExit(f"[resident-pp] FAIL splits differ\n{kvf}\n{kvd}")

    # Warning: **one separate context per stage** -- if several cards share one,
    #   `create_runtime` fails (measured).  Sharing is
    #   only needed between prefill ↔ decode of the same stage.
    ctxs, caches, ci = {}, {}, 0
    for nm in cpf.stage_names:
        cs = R.cache_tensors(len(kvf[nm]) // 2, KVH, HD, nblk)
        caches[nm] = cs
        ctxs[nm] = rebel.CompileContext(use_weight_sharing=True)
        for j, t in enumerate(cs):
            ctxs[nm].mark_static_address(t, f"kv_{j}")   # shared by prefill↔decode
            ci += 1
    print(f"[resident-pp] layers per stage {[len(kvf[n])//2 for n in cpf.stage_names]} · "
          f"{ci} resident caches · {nblk} blocks", flush=True)

    class Wrap(nn.Module):
        """Compiled stage + dummy restoration.  `seq` is read from STATE."""

        def __init__(self, rt, drop, total, skip):
            super().__init__()
            self.rt, self.drop, self.total = rt, drop, total
            self.skip = set(skip)      # input positions made unused by the rewrite

        def forward(self, *a):
            a = [x for i, x in enumerate(a) if i not in self.skip]
            o = self.rt.run(*a, torch.tensor([[STATE["seq"]]],
                                             dtype=torch.int32), bt)
            o = list(o) if isinstance(o, (list, tuple)) else [o]
            return R.restore_dummies(o, self.drop, self.total)

    def build(cap, inp, kvidx, tot, prefill, tag):
        tsum = 0.0
        for nm, devs in zip(cap.stage_names, stage_devs):
            sub = cap.split_gm.get_submodule(nm)
            nl = len(kvidx[nm]) // 2
            st = R.rewrite(sub, kvh=KVH, n_heads=NH, head_dim=HD, batch=B,
                           prefill=prefill, impl="vendor", verbose=False)
            if st.skipped or st.replaced != nl:
                raise SystemExit(f"[resident-pp] FAIL {nm} rewrote {st.replaced}/{nl}")
            R.prune_outputs_at(sub, kvidx[nm])
            # Note: first remove boundary inputs made unused by dropping the mask --
            #   for NPUs that remove unused inputs at compile time, the call's input
            #   count would otherwise mismatch.
            skip = R.drop_unused_placeholders(sub)
            tens = ([x for i, x in enumerate(inp[nm]) if i not in set(skip)]
                    + [torch.tensor([[0]], dtype=torch.int32), bt]
                    + list(caches[nm]))
            ph = [n.target for n in sub.graph.nodes if n.op == "placeholder"]
            if len(ph) != len(tens):
                raise SystemExit(f"[resident-pp] FAIL {nm} inputs {len(ph)} != {len(tens)}")
            info = [(n, list(t.shape), DT[t.dtype]) for n, t in zip(ph, tens)]
            t0 = time.time()
            c = rebel.compile_from_torch(sub, input_info=info,
                                         example_inputs=tens,
                                         compile_context=ctxs[nm],
                                         **({"tensor_parallel_size": TP}
                                            if TP > 1 else {}))
            tsum += time.time() - t0
            rt = rebel.Runtime(c, device=devs if len(devs) > 1 else devs[0],
                               tensor_type="pt")
            if skip:
                print(f"[resident-pp]   {nm}: removed {len(skip)} unused inputs {skip}",
                      flush=True)
            setattr(cap.split_gm, nm, Wrap(rt, kvidx[nm], tot[nm], skip))
        print(f"[resident-pp] OK {tag} compiled {tsum:.1f}s · {len(cap.stage_names)} stages",
              flush=True)

    build(cpf, inpf, kvf, totf, True, "prefill")
    build(cpd, inpd, kvd, totd, False, "decode ")

    with torch.no_grad():
        refg = model.generate(ids, max_new_tokens=NEW, do_sample=False,
                              use_cache=True,
                              pad_token_id=getattr(cfg, "eos_token_id", None))
    ref_new = refg[0, SEQ:].tolist()

    def gen(force=None):
        STATE["seq"] = 0
        t0 = time.time()
        with torch.no_grad():
            o = cpf.split_gm(ids, pos)
        pre_ms = (time.time() - t0) * 1000
        lg = o[0] if isinstance(o, (list, tuple)) else o
        cur, out, ms, valid = int(lg[0, -1].argmax()), [], [], SEQ
        out.append(cur)
        for i in range(NEW - 1):
            if force is not None and i < len(force):
                cur = force[i]
            STATE["seq"] = valid
            t0 = time.time()
            with torch.no_grad():
                o = cpd.split_gm(torch.tensor([[cur]], dtype=torch.long),
                                 torch.tensor([[valid]], dtype=torch.long))
            ms.append((time.time() - t0) * 1000)
            lg = o[0] if isinstance(o, (list, tuple)) else o
            valid += 1
            cur = int(lg[0, -1].argmax())
            out.append(cur)
        return out, pre_ms, ms

    free, pre_ms, ms = gen()
    n = min(len(free), len(ref_new))
    same = sum(1 for i in range(n) if free[i] == ref_new[i])
    tf, _, _ = gen(force=ref_new)
    tsame = sum(1 for i in range(n) if tf[i] == ref_new[i])
    host = sum(float(t.abs().sum()) for cs in caches.values() for t in cs)
    tok_s = 1000.0 / (sum(ms) / max(len(ms), 1)) if ms else 0.0
    print(f"[resident-pp] prefill {pre_ms:.1f}ms · decode {min(ms):.1f}~{max(ms):.1f}ms "
          f"({len(ms)} steps) · {tok_s:.0f} tok/s", flush=True)
    print(f"[resident-pp] host cache sum {host:.4f}  (0 = on device only — resident)")
    print(f"[resident-pp] teacher-forced match {tsame}/{n} · free-running {same}/{n}")
    ok = tsame >= max(1, int(n * 0.9)) and host == 0.0
    print(f"[resident-pp] RESULT {'PASS' if ok else 'FAIL'} resident-pp-generate "
          f"PP={PP} TP={TP} SEQ={SEQ} NEW={NEW} teacher_forced={tsame}/{n}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
