"""Models under test and their driver inputs.

Always fix the seed — without one, rel varies by 2.7x even with the same
binary and the PASS/FAIL verdict flips (measured).  All inputs go through --seed.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def build_synthetic(hidden: int = 5120) -> nn.Module:
    """Same topology as examples/pp_train_synthetic.py.

    It does not depend on transformers, so a failure here points to torch —
    this lets us observe issues separately from the model library.
    """

    class SyntheticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(hidden, hidden)
            self.linear2 = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
            self.linear3 = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
            self.linear4 = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
            self.linear5 = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
            self.linear6 = nn.Linear(hidden, hidden)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            x = self.relu(self.linear1(x))
            for m in self.linear2: x = self.relu(m(x))
            for m in self.linear3: x = self.relu(m(x))
            for m in self.linear4: x = self.relu(m(x))
            for m in self.linear5: x = self.relu(m(x))
            return self.relu(self.linear6(x))

    return SyntheticModel()


def load(model_id: str, *, dtype: torch.dtype, attn: str,
         use_cache: bool = False):
    """Load the model.

    Warning: on ATOM+, `attn_implementation` is a first-class variable that
      decides pass/fail, and it is model-dependent (e.g. GPT-2 is used with
      sdpa).  So the default stays at transformers' default, sdpa, and
      converting the boundary mask to float (C6) is the general rule.  Always
      log the value and the reason.
    """
    if model_id == "synthetic":
        return build_synthetic(), "SyntheticModel"

    from transformers import AutoConfig, AutoModelForCausalLM
    tok = os.getenv("LLAMA_ACCESS_TOKEN")
    # With use_cache=True, HF returns past_key_values as outputs — for 32
    # layers that is 64 (K,V) tensors leaving the graph.  Whether the PP
    # split and the vendor compiler can handle them is the question this
    # experiment asks.
    cfg = AutoConfig.from_pretrained(model_id, use_cache=use_cache, token=tok)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, config=cfg, torch_dtype=dtype,
        attn_implementation=attn, token=tok).eval()
    return m, type(m).__name__


# Real-sentence inputs — to avoid the "flat logits" of random tokens.
#
# A token sequence from torch.randint is meaningless, so the model is not
# confident and the top two candidates are often nearly tied.  With NPU
# arithmetic that is not bit-identical to the CPU such positions flip, so argmax
# drops because of the input.
# The opposite trap also happened — a reproducer that used torch.randn inputs
# underestimated the error.  Either way, measure
# with the distribution the model actually sees.
#
# Every sentence is at least 16 tokens and is truncated to seq — padding would
# create an attention_mask and drag in the boundary dtype axis, giving
# two variables instead of one.
_SENTENCES = [
    "The capital city of France is Paris, which sits on the river Seine in the north.",
    "Machine learning models are usually trained on very large datasets so that they can recognize statistical patterns.",
    "In the morning the fishermen pushed their small wooden boats out into the calm bay.",
    "A compiler translates the source code written by a programmer into the machine instructions that a processor runs.",
    "She opened the old wooden door and found a narrow staircase leading down to a cellar.",
    "The experiment measured the temperature of the sample every five minutes for an hour.",
    "Economists still disagree about whether raising interest rates will actually slow down inflation very quickly.",
    "He walked along the quiet beach at sunset, listening to the waves and the gulls above.",
    "The library on the corner keeps a collection of maps drawn before the railway was built.",
    "Nitrogen makes up roughly seventy eight percent of the air that we breathe every day.",
    "After the storm passed, the farmers went out to check the damage to their orchards.",
    "A good translator has to understand not only the words but also the customs behind them.",
    "The committee met for three hours and still could not agree on the wording of the clause.",
    "Volcanic ash from the eruption drifted across the ocean and reached the mainland by evening.",
    "Most birds that migrate south in autumn return to the same nesting grounds each spring.",
    "The engineer sketched the bridge on a napkin before anyone had approved the project.",
    "Reading aloud to young children helps them learn the rhythm of sentences long before school.",
    "The museum acquired a painting that had been hidden in a private collection for decades.",
    "Salt was once so valuable that soldiers in some armies were partly paid with it.",
    "She practised the same passage on the piano every evening until her fingers knew it.",
    "The river changed course after the flood and left the old harbour several miles inland.",
    "Astronomers measure the distance to nearby stars by watching how they shift against the background.",
    "He kept a notebook of every repair he made, with the date and the cost beside it.",
    "The treaty was signed in a small town that few people outside the region had heard of.",
    "Bees communicate the direction of a food source by dancing in a figure of eight.",
    "The new policy took effect in January and was revised again before the summer ended.",
    "Glass is made by heating sand with soda and lime until the mixture becomes a thick liquid.",
    "Her grandfather had crossed the mountains on foot when the roads were still unpaved.",
    "The report concluded that the shortage was caused by poor planning rather than bad weather.",
    "Sailors once navigated by the stars, and a cloudy night could cost them many days.",
]


def text_inputs(model_id: str, *, batch: int, seq: int, n: int) -> list:
    """Build n requests of shape (batch, seq) from real sentences.

    When seq is long one sentence is not enough, so distinct sentences are
    concatenated into a document and truncated to seq.  Padding is avoided
    because it would create an attention_mask and drag in the boundary dtype
    axis.

    Each document starts at a different offset in the pool to reduce overlap.
    For very long seq the pool wraps around, but the order still differs per
    document, so inputs do not become identical.  Note: long seq values do
    reuse sentences.
    """
    import os
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id,
                                        token=os.getenv("LLAMA_ACCESS_TOKEN"))
    pool = _SENTENCES
    out, cursor = [], 0
    for i in range(n):
        rows = []
        for _ in range(batch):
            ids, used = [], 0
            while len(ids) < seq:
                text = pool[cursor % len(pool)]
                cursor += 1
                used += 1
                piece = tok(text, return_tensors="pt").input_ids[0]
                # Keep BOS only on the first piece so it reads as one document.
                ids.extend(piece.tolist() if not ids else piece.tolist()[1:])
                if used > len(pool) * 4:
                    raise SystemExit(f"[models] could not fill seq={seq}")
            rows.append(torch.tensor(ids[:seq], dtype=torch.long))
        out.append({"input_ids": torch.stack(rows)})

    # Warning: control check — "worst of M" is only meaningful if the
    #   requests differ.  As seq grows the sentence pool runs out, documents
    #   overlap and eventually all become identical (measured: with a 30-sentence
    #   pool of ≈540 tokens, seq=512 collapses 8 requests into 1 distinct one).
    #   Passing silently would turn "the worst" into "one input measured eight times".
    uniq = {tuple(x["input_ids"].flatten().tolist()) for x in out}
    if len(uniq) != n:
        raise SystemExit(
            f"[models] only {len(uniq)} distinct real-sentence requests ({n} requested) — "
            f"filling seq={seq} × batch={batch} from a pool of {len(pool)} sentences "
            f"makes documents overlap.  Enlarge the pool or reduce seq.")
    return out


def driver_inputs(model, *, batch: int, seq: int) -> dict:
    """Inputs that drive the top-level split GraphModule.

    split_module keeps the original placeholders at the front of the graph,
    so these are also the inputs of stage 0.
    """
    name = type(model).__name__
    if name == "SyntheticModel":
        return {"x": torch.randn(batch, model.linear1.in_features)}
    if name.startswith(("GPT2", "Llama", "Qwen", "Mistral")):
        vocab = getattr(model.config, "vocab_size", 32000)
        return {"input_ids": torch.randint(0, min(vocab, 1000), (batch, seq),
                                           dtype=torch.long)}
    raise NotImplementedError(f"add an input spec for {name}")
