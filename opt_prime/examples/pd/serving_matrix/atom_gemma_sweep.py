"""Compile with upstreamPR827's process-local causal-prefill fix, then run the sweep."""

import runpy

import gemma_causal_compat  # noqa: F401 - apply validated SDK patch before compilation

runpy.run_module("atom_sweep", run_name="__main__")
