#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# ============================================================================
# pp_pretrain_llama.py
#   Single-file MLPerf-methodology pretraining example for HuggingFace
#   Llama-3.1 8B (from scratch) on C4, using ETRI opt_prime (PP x TP x DP).
#
#   This is the ONLY file for the opt_prime MLPerf pretraining example: the
#   Megatron .bin/.idx dataset adapter AND the tiny-C4 data preparer are
#   consolidated here (no separate opt_prime/data/ module, no prepare_c4_tiny.py).
# ============================================================================
#
# WHAT IT DOES
#   - Builds Llama-3.1 8B from *config* (random init = from-scratch pretraining;
#     --from-pretrained loads real weights for debugging only).
#   - Reads a Megatron *indexed* dataset (.bin/.idx) — EITHER the real MLPerf-
#     preprocessed C4, OR a small self-generated C4 shard (see DATA below).
#   - Trains with opt_prime: free PP/TP/DP, AdamW + linear-warmup/cosine, bf16.
#   Deferred to the patch plan (docs/mlperf_pretraining_patch_plan.md,
#   docs/pp_pretrain_llama_design.md): log-ppl eval, power/energy, mllog, chunked CE.
#
# ----------------------------------------------------------------------------
# 1) PREREQUISITE SOFTWARE
#   - opt_prime: no pip install — run from opt_prime/examples/ (this script adds
#     the package to sys.path). See aicomp/opt_prime/README.md for base setup.
#   - PyTorch >= 2.3.1 (tested 2.5.0+cu124) + CUDA GPUs; transformers (tested
#     4.46.2); packaging.  e.g. on a CUDA box:
#         pip install torch==2.5.0 transformers==4.46.2 packaging
#   - megatron-core == 0.10.0   (reads .bin/.idx via GPTDataset for TRAINING, and
#     writes .bin/.idx via IndexedDatasetBuilder for --prepare-tiny)
#         pip install --no-deps megatron-core==0.10.0
#     IMPORTANT: keep "--no-deps" and pin 0.10.x. megatron-core >= 0.16 pins
#     torch>=2.6 and will UPGRADE/replace torch 2.5 (breaking opt_prime).
#     (The datasets C++ helper ships prebuilt in the wheel; needs gcc/g++ if rebuilt.)
#   - HuggingFace `datasets`   (ONLY for --prepare-tiny; streams C4 from the Hub)
#         pip install datasets
#   - scipy >= 1.9   (optional; only for --partitioner milp/hierarchical).
#   - HuggingFace access token for gated Llama-3.1 — needed even for from_config
#     (config.json is gated) AND for the Llama-3.1 tokenizer.  Any one of:
#         huggingface-cli login
#         export LLAMA_ACCESS_TOKEN=<token>
#         pass <token> as the positional arg
#     (For a quick non-gated smoke, use --model-id gpt2 --tokenizer gpt2 with
#      --tp-size 1 — TP is Llama-only; and prepare tiny data with the SAME gpt2 tokenizer.)
#
# ----------------------------------------------------------------------------
# 2) DATA PREPARATION  (choose ONE; both yield a Megatron .bin/.idx read via --data-prefix)
#   (A) REAL MLPerf C4 (full; train shard ~84GB) — for MLPerf-equivalent runs.
#       The download script lives in a SEPARATE repo (NOT part of opt_prime):
#         MLPerf_Training_Power  (a fork of Dell EMC's "dellemc-mlperf-training-v5.1"
#         MLPerf Training v5.1 submission; layout follows the MLPerf submission
#         convention benchmarks/<bench>/implementations/<system>/).
#       Prereqs: curl + internet, >= ~85GB free disk, NO GPU needed.
#         cd MLPerf_Training_Power/benchmarks/llama31_8b/implementations/nemo_ETRI
#         export DATADIR=/path/for/data ; bash data_scripts/download_8b.sh
#         # -> $DATADIR/8b/c4-train.en_6_text_document.{bin,idx} + $DATADIR/8b/tokenizer/
#       (Resumable: uses the MLCommons R2 downloader = wget --continue + md5 check;
#        if interrupted, just re-run the same command with the same DATADIR.)
#       then train with:
#         --data-prefix $DATADIR/8b/c4-train.en_6_text_document
#         --tokenizer   $DATADIR/8b/tokenizer
#       Canonical MLPerf sources (if you don't have MLPerf_Training_Power):
#         - reference:   github.com/mlcommons/training  ->  small_llm_pretraining/nemo/ (8B)
#                        (405B: large_language_model_pretraining/nemo/)
#         - submissions: github.com/mlcommons/training_results_v5.1
#                        -> <Submitter>/benchmarks/llama31_8b/implementations/nemo/
#       In ALL cases the actual C4 .bin/.idx is fetched from MLCommons R2 storage
#       (training.mlcommons-storage.org); the repo only holds the download script.
#
#   (B) TINY C4 (self-generated, ~MB; for bring-up / smoke tests) — run THIS file
#       in preparation mode (CPU only, NO torchrun). It streams N docs of C4 from
#       the HF Hub, tokenizes (Llama-3.1 by default), and writes a small .bin/.idx
#       at --data-prefix:
#         python3 pp_pretrain_llama.py --prepare-tiny \
#             --data-prefix /path/c4_tiny/c4-train.en_tiny \
#             --tiny-num-docs 20000 --tokenizer meta-llama/Llama-3.1-8B <token>
#       FULLY self-contained within opt_prime (no external repo). Then train (see
#       §3) with the SAME --data-prefix and --tokenizer used above.
#       (NOT the MLPerf reference preprocessing — different shard/order. Bring-up only.)
#
# ----------------------------------------------------------------------------
# 3) RUN TRAINING  (torchrun; run from opt_prime/examples/)
#     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # recommended for long / large-seq
#
#   # 8x A40 (48GB), pure pipeline (simplest memory-safe bring-up):
#     torchrun --nproc_per_node=8 --master_port=29500 pp_pretrain_llama.py \
#         --data-prefix <prefix> --tokenizer <tok> \
#         --pp-size 8 --tp-size 1 --dp-size 1 \
#         --gbs 8 --micro-bs 1 --seq-len 1024 --max-steps 200 \
#         --lr 3e-4 --warmup-steps 30 --dtype bf16 --activation-ckpt <token>
#
#   # 2 GPU, MLPerf 2-GPU parity (TP=2 / PP=1):
#     torchrun --nproc_per_node=2 --master_port=29500 pp_pretrain_llama.py \
#         --data-prefix <prefix> --tokenizer <tok> \
#         --tp-size 2 --pp-size 1 --dp-size 1 --gbs 8 --seq-len 1024 ... <token>
#
# ----------------------------------------------------------------------------
# NOTES
#   - world_size (nproc_per_node x nnodes) MUST equal pp_size * tp_size * dp_size.
#   - Only PP and TP shard the model (DP replicates it). Rough A40-48GB fit for 8B:
#     pp*tp >= 2 at seq~1024 (e.g. tp=2/pp=1 uses ~43GB), but larger seq needs more
#     sharding — seq 8192 needs pp*tp >= 4 (e.g. pp=8, or pp=4/tp=2 for gbs<=16).
#     TP shards the vocab-128256 lm_head, easing the otherwise-heavy last stage.
#   - micro_bs=1 recommended. At PP=1 with 1F1B, num_mb / GBS do NOT affect peak
#     memory (peak = static + ONE micro-batch); memory scales with seq_len.
#   - --cache-dir <dir>: caches the .npy sample-index. Recommended for the large
#     real dataset (first run builds it over the 84GB shard; reused afterwards).
#   - force_free_mem=True (set below) is REQUIRED for long runs (frees per-step
#     buffers; otherwise OOM after a few hundred steps).
#   - Gradient clipping is auto-disabled under TP (clip_grad_norm_ cannot handle
#     mixed DTensor/plain-Tensor params); opt_prime's TP examples also skip it.
#
# *** Tested target: torch 2.5.0, transformers 4.46.2, megatron-core 0.10.0 ***
#

import os
import sys
import math
import time
import logging
import argparse
from typing import Optional

import torch
import torch.distributed as dist
from packaging import version
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p

logging.basicConfig(level=logging.ERROR)


# ============================================================================
# Megatron indexed-dataset adapter  (consolidated from opt_prime/data/)
# ----------------------------------------------------------------------------
# Reads the real MLPerf-preprocessed C4 (or a tiny self-generated shard) — a
# Megatron indexed dataset (.bin/.idx, pre-tokenized) — and exposes it to
# opt_prime as a map-style Dataset yielding (input_ids, labels) int64 tensors.
#
# Compatibility: MLPerf uses reset_position_ids / reset_attention_mask /
# eod_mask_loss = False -> standard full-causal attention + all-position
# next-token loss, exactly what opt_prime computes from a single input_ids
# tensor. GPTDataset returns tokens(=text[:-1]) & labels(=text[1:]) via
# add_extra_token_to_sequence=True, matching opt_prime's CE(ignore_index=-100).
# So only this thin format adapter is needed; no change to opt_prime's loss.
# ============================================================================
class _EodTokenizer:
    """Minimal tokenizer stub for GPTDatasetConfig.

    GPTDataset.__getitem__ only touches config.tokenizer.eod (to build the
    loss_mask/position_ids that opt_prime discards). Since reset_* / eod_mask
    are all False, the concrete eod value never changes tokens/labels.
    """

    def __init__(self, eod_id: int, vocab_size: Optional[int] = None):
        self.eod = int(eod_id)
        self._vocab_size = int(vocab_size) if vocab_size is not None else None
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)

    @property
    def unique_identifiers(self):
        # megatron hashes the config (incl. tokenizer) into a cache key via
        # json.dumps(..., default=lambda o: o.unique_identifiers).
        return {"class": type(self).__name__, "eod": self.eod, "vocab_size": self._vocab_size}


class C4PackedDataset(Dataset):
    """Wrap a megatron GPTDataset so each item is (input_ids, labels) int64 [seq]."""

    def __init__(self, gpt_dataset):
        self.ds = gpt_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]  # {"tokens", "labels", ...}
        return sample["tokens"].long(), sample["labels"].long()


def build_c4_pretrain_dataset(data_prefix, seq_length, num_samples, eod_id,
                              vocab_size=None, seed=1234, cache_dir=None, split="1,0,0"):
    """Build a train C4PackedDataset from a Megatron indexed dataset prefix.

    data_prefix : path WITHOUT the .bin/.idx suffix.
    num_samples : #samples for the sample-index (set >= gbs * max_steps).
    eod_id      : EOD token id of the tokenizer that built the .bin (inert here,
                  but required by GPTDatasetConfig).
    cache_dir   : dir for cached .npy index mappings (like MLPerf /npy_index).
    NOTE: build() is collective when torch.distributed is initialized -> call on
    all ranks together (do NOT rank-0-gate it: deadlocks on internal barriers).
    """
    try:
        from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
        from megatron.core.datasets.blended_megatron_dataset_builder import (
            BlendedMegatronDatasetBuilder,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "requires megatron-core (tested 0.10.0). Install: "
            "pip install --no-deps megatron-core==0.10.0  "
            "(do NOT upgrade torch; >=0.16 needs torch>=2.6). "
            f"Original error: {e!r}"
        )

    config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=seq_length,
        blend=([data_prefix], [1.0]),
        split=split,
        path_to_cache=cache_dir,
        tokenizer=_EodTokenizer(eod_id, vocab_size),
        reset_position_ids=False,      # MLPerf policy -> matches opt_prime default
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,   # opt_prime relies on the traced causal mask
        mmap_bin_files=True,
    )
    builder = BlendedMegatronDatasetBuilder(
        GPTDataset, [num_samples, 0, 0], lambda: True, config)
    train_ds, _valid, _test = builder.build()
    return C4PackedDataset(train_ds)


# ============================================================================
# Tiny-C4 preparation  (consolidated from prepare_c4_tiny.py) — CPU only
# ----------------------------------------------------------------------------
# Streams the first N docs of C4 from the HF Hub, tokenizes (Llama-3.1 by
# default), appends EOD per doc, and writes a small Megatron .bin/.idx at
# output_prefix — the exact format build_c4_pretrain_dataset() reads.
# ============================================================================
def prepare_c4_tiny(output_prefix, num_docs, tokenizer_id, access_token=None,
                    dataset="allenai/c4", config="en", split="train",
                    append_eod=True, log_every=2000):
    import numpy as np
    from datasets import load_dataset
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    out_dir = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(out_dir, exist_ok=True)
    bin_path, idx_path = output_prefix + ".bin", output_prefix + ".idx"

    print(f"[prepare-tiny] tokenizer: {tokenizer_id}")
    tok = AutoTokenizer.from_pretrained(tokenizer_id, token=access_token)
    eod_id = tok.eos_token_id
    vocab_size = len(tok)
    # Llama-3.1 vocab (128256) exceeds uint16 range -> int32 required.
    print(f"[prepare-tiny] eod_id={eod_id} vocab_size={vocab_size} bin_dtype=int32")
    print(f"[prepare-tiny] streaming {num_docs} docs from {dataset}/{config}:{split}")

    ds = load_dataset(dataset, config, split=split, streaming=True)
    builder = IndexedDatasetBuilder(bin_path, dtype=np.int32)
    n_docs = n_tokens = 0
    for rec in ds:
        if n_docs >= num_docs:
            break
        ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
        if append_eod:
            ids = ids + [eod_id]
        if len(ids) == 0:
            continue
        builder.add_item(torch.IntTensor(ids))
        builder.end_document()
        n_docs += 1
        n_tokens += len(ids)
        if n_docs % log_every == 0:
            print(f"  ... {n_docs} docs, {n_tokens:,} tokens")
    builder.finalize(idx_path)

    print(f"[prepare-tiny] DONE: {n_docs} docs, {n_tokens:,} tokens")
    print(f"[prepare-tiny] wrote: {bin_path} ({os.path.getsize(bin_path):,} B) + {idx_path}")
    for s in (1024, 2048, 4096, 8192):
        print(f"    ~{n_tokens // (s + 1):,} packed samples at seq_len={s}")
    print(f"[prepare-tiny] train with:  --data-prefix {output_prefix}")


# ============================================================================
# Arguments
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="opt_prime Llama-3.1 8B C4 pretraining (single file)")
    p.add_argument("token", nargs="?", default=None, help="HF access token (gated Llama)")
    # data (Megatron indexed .bin/.idx) — used to READ (train) or WRITE (--prepare-tiny)
    p.add_argument("--data-prefix", required=True,
                   help="indexed dataset prefix WITHOUT .bin/.idx "
                        "(read target for training; write target for --prepare-tiny)")
    p.add_argument("--cache-dir", default=None,
                   help="dir for cached .npy index mappings (like MLPerf /npy_index)")
    p.add_argument("--eod-id", type=int, default=None,
                   help="EOD token id (default: tokenizer.eos_token_id; inert given reset/eod=False)")
    # --- tiny-data preparation mode (CPU only, no torchrun) ---
    p.add_argument("--prepare-tiny", action="store_true",
                   help="PREP MODE: stream C4 from HF, tokenize, write a small .bin/.idx "
                        "at --data-prefix, then exit. Run WITHOUT torchrun. See header (2B).")
    p.add_argument("--tiny-num-docs", type=int, default=20000, help="[prep] #C4 docs to take")
    p.add_argument("--tiny-dataset", default="allenai/c4", help="[prep] HF dataset id")
    p.add_argument("--tiny-config", default="en", help="[prep] dataset config name")
    p.add_argument("--tiny-split", default="train", help="[prep] train | validation")
    p.add_argument("--no-append-eod", action="store_true", help="[prep] don't append EOD per doc")
    p.add_argument("--log-every", type=int, default=2000, help="[prep] progress log interval (docs)")
    # model
    p.add_argument("--model-id", default="meta-llama/Llama-3.1-8B",
                   help="HF model/config id (arch source; weights only with --from-pretrained)")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer id/path (default: --model-id). Used for eod/vocab (& prep tokenize).")
    p.add_argument("--from-pretrained", action="store_true",
                   help="load pretrained weights instead of random init (debug; not pretraining)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    # parallelism (free; must satisfy world_size == pp*tp*dp)
    p.add_argument("--pp-size", type=int, default=8)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--activation-ckpt", action="store_true", default=False)
    p.add_argument("--swap-opt", action="store_true", default=False,
                   help="offload optimizer state to host during fwd/bwd (memory relief)")
    p.add_argument("--partitioner", default="auto",
                   choices=["auto", "simple", "milp", "hierarchical", "llama-tp-split"],
                   help="pipeline partitioner. 'auto'=llama-tp-split if Llama+tp>1 else simple. "
                        "'milp'/'hierarchical' minimize cross-stage activation volume. "
                        "Llama+tp>1 always forces llama-tp-split.")
    # batch / schedule
    p.add_argument("--gbs", type=int, default=8, help="global batch size")
    p.add_argument("--micro-bs", type=int, default=1, help="micro batch size")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-interval", type=int, default=10, help="[train] log every N steps")
    return p.parse_args()


args = parse_args()
if args.token:
    os.environ["LLAMA_ACCESS_TOKEN"] = args.token
access_token = os.getenv("LLAMA_ACCESS_TOKEN")


# ============================================================================
# Preparation mode (CPU only, no torchrun) — write tiny C4 and exit
# ============================================================================
if args.prepare_tiny:
    prepare_c4_tiny(
        output_prefix=args.data_prefix,
        num_docs=args.tiny_num_docs,
        tokenizer_id=(args.tokenizer or args.model_id),
        access_token=access_token,
        dataset=args.tiny_dataset,
        config=args.tiny_config,
        split=args.tiny_split,
        append_eod=(not args.no_append_eod),
        log_every=args.log_every,
    )
    sys.exit(0)


# ============================================================================
# Training mode (torchrun)
# ============================================================================
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def rank0(msg):
    if rank == 0:
        print(msg, flush=True)


# ---- version guard ----
if version.parse(torch.__version__.split("+")[0]) < version.parse("2.3.1"):
    raise ValueError(f"torch >= 2.3.1 required, found {torch.__version__}")

# ---- derive batch geometry ----
assert world_size == args.pp_size * args.tp_size * args.dp_size, (
    f"world_size({world_size}) must equal pp*tp*dp "
    f"({args.pp_size}*{args.tp_size}*{args.dp_size})")
assert args.gbs % args.dp_size == 0, "gbs must be divisible by dp_size"
batch_size = args.gbs // args.dp_size            # per-DP-replica batch
assert batch_size % args.micro_bs == 0, "per-replica batch must be divisible by micro_bs"
num_mb = batch_size // args.micro_bs             # microbatches per step
assert num_mb >= 1
if num_mb < args.pp_size:
    rank0(f"[warn] num_mb({num_mb}) < pp_size({args.pp_size}) — large pipeline bubble")

dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

rank0(f"> world={world_size} pp={args.pp_size} tp={args.tp_size} dp={args.dp_size} "
      f"| gbs={args.gbs} micro_bs={args.micro_bs} batch_size(per-dp)={batch_size} num_mb={num_mb} "
      f"| seq_len={args.seq_len} dtype={args.dtype} partitioner={args.partitioner}")

# ---- tokenizer (eod / vocab) ----
tok_id = args.tokenizer or args.model_id
tokenizer = AutoTokenizer.from_pretrained(tok_id, token=access_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
eod_id = args.eod_id if args.eod_id is not None else tokenizer.eos_token_id
vocab_size = len(tokenizer)
rank0(f"> tokenizer={tok_id} eod_id={eod_id} vocab_size={vocab_size}")

# ---- model (from scratch) ----
config = AutoConfig.from_pretrained(args.model_id, token=access_token)
config.use_cache = False
if args.from_pretrained:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, token=access_token, config=config, torch_dtype=dtype)
    rank0("> model: loaded pretrained weights (debug mode, not from-scratch)")
else:
    # Create params directly in target dtype (avoids a transient fp32 copy; for
    # 8B x nproc that fp32 intermediate would spike host RAM). bf16 random init
    # is fine for a from-scratch run.
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_config(config)   # random init -> pretraining
    finally:
        torch.set_default_dtype(_prev_dtype)
    rank0(f"> model: from_config (random init, from scratch, dtype={args.dtype})")
model = model.to(dtype)

assert config.vocab_size >= vocab_size, (
    f"model vocab_size({config.vocab_size}) < tokenizer vocab({vocab_size})")
rank0(f"> total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- engine ----
optimus_p = Optimus_p(
    model, num_mb, use_gpu=True,
    pp_size=args.pp_size, tp_size=args.tp_size, dp_size=args.dp_size,
    activation_ckpt=args.activation_ckpt,
    swap_opt_in_fwdbwd=args.swap_opt,
    partitioner=args.partitioner,
    force_free_mem=True,         # REQUIRED for long runs: gates clean_run_info()
                                 # (frees per-step buffers; else OOM after ~100s of steps).
    grad_accum_normalize=True,   # mean over num_mb (standard PP convention)
)
optimus_p.train()

# ---- optimizer + schedule ----
# foreach=False required for TP/DTensor compatibility (mixed DTensor/Tensor).
optimus_p.optimizer = torch.optim.AdamW(
    optimus_p.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-5,
    weight_decay=args.weight_decay, foreach=False)


def lr_lambda(step):
    if step < args.warmup_steps:
        return float(step + 1) / float(max(1, args.warmup_steps))
    progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


scheduler = torch.optim.lr_scheduler.LambdaLR(optimus_p.optimizer, lr_lambda)

# ---- dataset (Megatron .bin/.idx via --data-prefix; real MLPerf or tiny) ----
# build after Optimus_p so torch.distributed is initialized (build() is collective).
num_samples = args.gbs * (args.max_steps + 2)
rank0(f"> building C4 dataset: prefix={args.data_prefix} num_samples={num_samples}")
dataset = build_c4_pretrain_dataset(
    data_prefix=args.data_prefix, seq_length=args.seq_len, num_samples=num_samples,
    eod_id=eod_id, vocab_size=vocab_size, seed=args.seed, cache_dir=args.cache_dir)
dataloader = optimus_p.prepare_dataloader(dataset, batch_size)
rank0(f"> dataset ready: {len(dataset)} samples, {len(dataloader)} batches")


# ---- train loop ----
def train():
    optimus_p.train()
    total_loss = 0.0
    tokens_per_step = args.gbs * args.seq_len
    clip_disabled = False
    start = time.time()
    for step, batch in enumerate(dataloader):
        if step >= args.max_steps:
            break

        data, labels = None, None
        if optimus_p.is_first_stage():
            data = batch[0]      # input_ids       [batch_size, seq_len]
            labels = batch[1]    # next-token labels [batch_size, seq_len]

        labels = optimus_p.move_labels2last_stage(labels)

        optimus_p.optimizer.zero_grad()
        optimus_p.run(data, labels, mode="1f1b")

        if args.clip_grad and args.clip_grad > 0 and not clip_disabled:
            try:
                torch.nn.utils.clip_grad_norm_(
                    optimus_p.parameters(), args.clip_grad, foreach=False)
            except RuntimeError as e:
                # Under TP, parameters() mixes DTensor (sharded proj) and plain
                # Tensor (norms); clip_grad_norm_ can't compute a joint norm.
                # Disable (opt_prime's TP examples also skip clipping). Future work.
                clip_disabled = True
                if rank == 0:
                    print(f"[warn] grad clipping disabled under TP "
                          f"(mixed DTensor/Tensor): {type(e).__name__}", flush=True)

        optimus_p.optimizer.step()
        scheduler.step()

        if optimus_p.is_last_stage():
            loss = sum(optimus_p.get_loss()) / optimus_p.num_mb
            total_loss += float(loss)
            if step % args.log_interval == 0 and step > 0:
                elapsed = time.time() - start
                cur_loss = total_loss / args.log_interval
                step_time = elapsed / args.log_interval
                tok_s = tokens_per_step / step_time
                print("| step {:6d}/{:6d} | lr {:.3e} | s/step {:6.3f} | "
                      "tok/s {:9.0f} | loss {:7.4f} | ppl {:9.2f}".format(
                          step, args.max_steps, scheduler.get_last_lr()[0],
                          step_time, tok_s, cur_loss, math.exp(min(cur_loss, 20))),
                      flush=True)
                total_loss = 0.0
                start = time.time()


t0 = time.time()
train()
if rank == 0:
    print(f"Time elapsed: {time.time() - t0:.3f} sec", flush=True)

# peak GPU memory per rank (stages differ: embedding/lm_head stages are heavier)
if torch.cuda.is_available():
    print(f"[mem] rank {rank} cuda:{torch.cuda.current_device()} "
          f"peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
          f"peak_reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB", flush=True)

# ---- cleanup ----
if dist.is_initialized():
    try:
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Cleanup on rank {rank}: {e}", flush=True)

rank0("[done] pp_pretrain_llama.py")
