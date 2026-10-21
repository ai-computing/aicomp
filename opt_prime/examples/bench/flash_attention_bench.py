"""SDPA vs FlashAttention-2 on A100 (#109): measure before adopting.

    pip install kernels      # prebuilt FA2 from the Hub (kernels-community/flash-attn2); flash-attn has no wheel here
    python examples/bench/flash_attention_bench.py --model unsloth/Llama-3.2-1B-Instruct

Four measurements, all bf16 (FA2 has no fp32 path):
  kernel   causal prefill attention, forward and forward+backward, S = 512 ... 8192 (Llama-3.2-1B head geometry)
  prefill  model forward at S = 512 / 2048 / 4096 with attn "sdpa" vs FA2; logits compared with an fp32 reference
  train    one training step (forward + backward) at S = 2048, B = 2; loss and grad norm compared
  decode   one decode step of 32 sequences with mixed lengths (128-2047): paged flex attention (the batcher's #108
           path, page 64), padded SDPA with a mask (the slot path), FA2 flash_attn_with_kvcache with a block table
           (page 256: FA2's paged KV needs pages that are multiples of 256)
"""
import argparse, json, time
import torch
import torch.nn.functional as F
import kernels
from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.masking_utils import AttentionMaskInterface

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--repeat", type=int, default=20)
a = ap.parse_args()
fa = kernels.get_kernel("kernels-community/flash-attn2", version=2)
dev, bf = "cuda", torch.bfloat16


def timed(fn, repeat=a.repeat):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1e3


def _fa2(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):     # HF layout [B, H, S, D]
    out = fa.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                             softmax_scale=scaling, causal=query.shape[2] > 1)
    return out, None


AttentionInterface.register("fa2_hub", _fa2)
AttentionMaskInterface.register("fa2_hub", lambda *x, **k: None)     # causal, no padding: the kernel masks
results = {}

# ---- kernel ----
H, KV, D = 32, 8, 64
rows = []
for S in (512, 2048, 4096, 8192):
    q = torch.randn(1, S, H, D, device=dev, dtype=bf, requires_grad=True)
    k = torch.randn(1, S, KV, D, device=dev, dtype=bf, requires_grad=True)
    v = torch.randn(1, S, KV, D, device=dev, dtype=bf, requires_grad=True)
    sd = lambda: F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                is_causal=True, enable_gqa=True)
    f2 = lambda: fa.flash_attn_func(q, k, v, causal=True)
    err = (sd().transpose(1, 2).float() - f2().float()).abs().max().item()
    with torch.no_grad():
        fw = timed(sd), timed(f2)
    bw = timed(lambda: sd().sum().backward()), timed(lambda: f2().sum().backward())
    rows.append(dict(S=S, fwd_ms_sdpa=round(fw[0], 3), fwd_ms_fa2=round(fw[1], 3), fwd_x=round(fw[0] / fw[1], 2),
                     fwdbwd_ms_sdpa=round(bw[0], 3), fwdbwd_ms_fa2=round(bw[1], 3), fwdbwd_x=round(bw[0] / bw[1], 2),
                     max_abs_diff=round(err, 4)))
results["kernel"] = rows
print(json.dumps(rows), flush=True)

# ---- model prefill / train ----
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=bf).to(dev)
ref = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
vocab = model.config.vocab_size
g = torch.Generator().manual_seed(0)


def use(impl):
    model.config._attn_implementation = impl


rows = []
model.eval()
for S in (512, 2048, 4096):
    ids = torch.randint(100, vocab - 100, (1, S), generator=g).to(dev)
    with torch.inference_mode():
        r = ref(ids, logits_to_keep=64).logits.float()
        row = dict(S=S)
        for impl in ("sdpa", "fa2_hub"):
            use(impl)
            row[f"ms_{impl}"] = round(timed(lambda: model(ids, logits_to_keep=1), 10), 2)
            out = model(ids, logits_to_keep=64).logits.float()
            row[f"err_vs_fp32_{impl}"] = round((out - r).abs().max().item(), 3)
            row[f"argmax_agree_{impl}"] = round((out.argmax(-1) == r.argmax(-1)).float().mean().item(), 3)
        row["x"] = round(row["ms_sdpa"] / row["ms_fa2_hub"], 2)
    rows.append(row)
results["prefill"] = rows
print(json.dumps(rows), flush=True)
del ref
torch.cuda.empty_cache()

model.train()
ids = torch.randint(100, vocab - 100, (2, 2048), generator=g).to(dev)
row = {}
for impl in ("sdpa", "fa2_hub"):
    use(impl)
    def step():
        model.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        return loss
    torch.cuda.reset_peak_memory_stats()
    row[f"ms_{impl}"] = round(timed(step, 5), 1)
    torch.manual_seed(0)
    loss = step()
    row[f"loss_{impl}"] = round(loss.item(), 5)
    row[f"gradnorm_{impl}"] = round(torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]).item(), 4)
    row[f"peak_gib_{impl}"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
row["x"] = round(row["ms_sdpa"] / row["ms_fa2_hub"], 2)
results["train"] = row
print(json.dumps(row), flush=True)
model.zero_grad(set_to_none=True)
model.eval()

# ---- decode, mixed lengths ----
from opt_prime.accel.pd.paged import PagedKV, _flex
B = 32
lengths = torch.randint(128, 2048, (B,), generator=g).tolist()
row = dict(batch=B, mean_len=round(sum(lengths) / B))
q = torch.randn(B, H, 1, D, device=dev, dtype=bf)
# paged flex (page 64), one layer
kv = PagedKV(1, B, KV, D, bf, dev, pages=B * 2048 // 64, page_size=64, max_context=2048)
for s, n in enumerate(lengths):
    kv.alloc(s, n)
kv.k.normal_(); kv.v.normal_()
mask = kv.block_mask(list(range(B)), lengths)
flex = _flex("cuda")
o_flex = flex(q, kv.k[0][None], kv.v[0][None], block_mask=mask, enable_gqa=True)
row["us_flex_paged"] = round(1e3 * timed(lambda: flex(q, kv.k[0][None], kv.v[0][None], block_mask=mask, enable_gqa=True), 100), 1)
row["us_flex_paged_with_mask_build"] = round(1e3 * timed(lambda: kv.block_mask(list(range(B)), lengths), 100), 1)
# gather the same KV per sequence for the other two paths
ks = [kv.k[0][:, kv.physical([s] * n, torch.arange(n, device=dev))] for s, n in enumerate(lengths)]
vs = [kv.v[0][:, kv.physical([s] * n, torch.arange(n, device=dev))] for s, n in enumerate(lengths)]
L = max(lengths)
kp = torch.zeros(B, KV, L, D, device=dev, dtype=bf); vp = torch.zeros_like(kp)
am = torch.zeros(B, 1, 1, L, device=dev, dtype=torch.bool)
for s, n in enumerate(lengths):
    kp[s, :, :n], vp[s, :, :n], am[s, ..., :n] = ks[s], vs[s], True
sd = lambda: F.scaled_dot_product_attention(q, kp, vp, attn_mask=am, enable_gqa=True)
row["us_sdpa_padded"] = round(1e3 * timed(sd, 100), 1)
# FA2 paged (page 256)
P2 = 256
npg = [-(-n // P2) for n in lengths]
kc = torch.zeros(sum(npg), P2, KV, D, device=dev, dtype=bf); vc = torch.zeros_like(kc)
table = torch.zeros(B, max(npg), dtype=torch.int32, device=dev)
base = 0
for s, n in enumerate(lengths):
    for i in range(npg[s]):
        table[s, i] = base + i
        m = min(P2, n - i * P2)
        kc[base + i, :m] = ks[s][:, i * P2:i * P2 + m].transpose(0, 1)
        vc[base + i, :m] = vs[s][:, i * P2:i * P2 + m].transpose(0, 1)
    base += npg[s]
seqlens = torch.tensor(lengths, dtype=torch.int32, device=dev)
f2 = lambda: fa.flash_attn_with_kvcache(q.transpose(1, 2), kc, vc, cache_seqlens=seqlens, block_table=table)
row["us_fa2_paged256"] = round(1e3 * timed(f2, 100), 1)
o_sd, o_f2 = sd(), f2().transpose(1, 2)
row["max_diff_flex_vs_sdpa"] = round((o_flex - o_sd).abs().max().item(), 4)
row["max_diff_fa2_vs_sdpa"] = round((o_f2 - o_sd).abs().max().item(), 4)
results["decode"] = row
print(json.dumps(row), flush=True)
print(json.dumps(dict(model=a.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      fa2="kernels-community/flash-attn2 v2", results=results)))
