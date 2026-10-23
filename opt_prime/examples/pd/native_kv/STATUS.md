# ATOM staging experiment

State: review. Related issue: https://github.com/ai-computing/aicomp/issues/205

Warm import latency improved by 34–37% on Qwen3-4B at 128/512/2048 tokens. Device readbacks and compared decode logits are exact between importers. See experiments/2026-10-23-atom-staging205.md.

Next: review and integrate into HTTP serving; overlap and full model coverage remain unverified.
