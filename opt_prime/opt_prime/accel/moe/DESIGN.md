# moe_offload — heterogeneous MoE expert placement and execution

Goal: run any MoE model whose experts do not fit in one accelerator by spreading each layer's experts over several
*tiers* (GPUs, NPUs, the host CPU), running the tiers concurrently and summing their partial outputs — and make adding
a model, an accelerator type or a kernel a local change.

## Three independent axes

| axis | abstraction | what adding support means |
|---|---|---|
| model | `ModelAdapter` → `ExpertSpec` per MoE layer | standard transformers >= 5 experts modules are handled by `GenericAdapter`; a family with a different experts layout registers one adapter |
| device | `ExpertBackend` (one per device type) | implement `supports / capacity / cost / load / run`; register it under a name (`cuda`, `cpu`, `rngd`, ...) |
| kernel | per-backend kernel registry keyed by `ExpertSpec.signature()` | for devices whose kernels are compiled per shape (NPUs), add a tiling rule; the backend builds and caches the kernel |

`ExpertSpec` is the contract between the axes: `(num_experts, hidden, ffn, activation, gate, has_bias, transposed,
dtype)`. Backends never look at model classes, adapters never look at devices.

## Layers on top

```
               ┌──────────────────────── apply_moe_offload(model, tiers, profile) ────────────────────────┐
 ModelAdapter ─┤ ExpertSpec per layer     Planner (capacity + cost models of the tiers) → placement       │
               │                          OffloadedLayer: split routed pairs by tier, run tiers            │
               │                          concurrently, sum partial outputs                                │
               └───────────────┬──────────────────────────────┬──────────────────────────┬──────────────┘
                         ExpertBackend "cuda"           ExpertBackend "rngd"        ExpertBackend "cpu"
                         (torch: batched/grouped)       (furiosa-opt kernels,       (native AVX-512 BF16 /
                                                         Rust expert server)          torch)
```

* **Tier** = one backend instance on one device (`cuda:0`, `rngd:1`, `cpu`), with a memory budget.
* **Planner**: per layer, assign experts to tiers to minimize the expected critical path
  `max_t cost_t(experts routed to t)` subject to each tier's capacity, using routing counts from a profile. A greedy
  heuristic (hottest experts to the cheapest tier with room, then rebalance) is the default; `top` and `mix:<h>` are
  kept for reproducing earlier hand-made placements.
* **OffloadedLayer**: one host sync per layer to learn which pairs go where; launches device tiers asynchronously, runs
  host tiers in a worker; partial sums are added in the hidden dtype. Pairs a tier does not own are passed with
  weight 0 (or compacted, for tiers whose kernels cost per pair).
* The custom op `opt_prime::offloaded_experts` keeps all of this opaque to `torch.export` / the OptimusPrime IR.

## Backend contract

```python
class ExpertBackend:
    name: str
    def supports(self, spec: ExpertSpec) -> bool          # can this backend run this expert shape?
    def capacity_bytes(self) -> int                       # memory available for expert weights
    def cost(self, spec, n_experts, n_pairs) -> float     # expected seconds for one call (planner input)
    def load(self, spec, weights: ExpertWeights) -> Handle  # upload a subset of experts (local ids 0..n-1)
    def run(self, handle, hidden, local_index, weights) -> Tensor   # [T,H] partial output, hidden dtype
```

`ExpertWeights` is the model-independent view an adapter produces: stacked tensors for the selected experts
(`gate_up[n,2F,H]`, `down[n,H,F]`, optional biases) plus the spec. Cost models start as measured constants
(bytes/s, per-call overhead) and can be calibrated by a `bench()` method per backend.

## Accelerators driven by a graph compiler (RNGD now, ATOM later, other NPUs)

`backends/npu_graph.GraphNPUBackend` holds everything such devices share: the `SlotFFN` graph (K experts on T
tokens, weighted by a gate matrix G[T,K]), token/slot buckets for static shapes, a compiled-graph cache keyed by
`(spec.signature(), T_bucket, K_bucket)`, and two weight-residency modes:

* `inputs` — every expert's weights are device tensors passed as graph inputs; the host picks K per call (pointer
  swap, no weight traffic). RNGD (furiosa-torch) uses this.
* `baked` — one graph per expert with its weights as constants, for runtimes that upload every input per call.

A vendor backend implements three hooks:

```python
@register_backend("my-npu")
class MyNpuBackend(GraphNPUBackend):
    residency = "inputs"                  # or "baked"
    overhead_s, bandwidth_Bps = 300e-6, 500e9          # cost model used by the planner
    def default_capacity(self): ...
    def _compile(self, module, sample_inputs): ...      # e.g. vendor.compile_from_torch(module, samples)
    def _to_device(self, t): ...
    def _to_host(self, t): ...
```

`emulate=True` (or `MOE_OFFLOAD_EMULATE_NPU=1`) runs the same graphs on the host, so placement, slot grouping and
numerics are tested without the device or its SDK.

### Per-shape kernels (RNGD `rngd-opt`)

The npu-kernels package (`npu_kernels.rngd`, separate repository; the kernels need the furiosa-opt toolchain and have their own release cycle) turns an expert shape into a furiosa-opt crate: `plan(H, F)` derives the tiling (gate/up rows
split over 2 clusters x A row groups x H/32 column blocks; down rows over C groups of HR rows with 16-aligned column
blocks; Solar-Open's plan reproduces the hand-written kernel of npu-offload), `generate()` fills `template/` (kernel,
CPU-emulation check binary, expert server binary). The `rngd-opt` backend builds the server once per machine
(`cargo furiosa-opt build`; plain `cargo build` = CPU emulator with `emulate=True`), keeps one resident
`slots_per_layer`-expert table per MoE layer in HBM, and per decode token sends (layer, <=K slots, x) over a pipe.
A new shape needs nothing; a new tiling idea is a template change, checked with `cargo run --bin eval -- check`.
The planner honours `slots_per_layer` (any backend may set it) and picks `rngd` vs `rngd-opt` by cost.

## Adaptive re-placement

Profiles from calibration prompts can miss the routing of served traffic (Qwen3-30B-A3B: profile predicted 75 %
of routed pairs on the GPU at a 15 GB budget, generation reached 45 %). With `adapt_every=N`, each layer keeps an
EMA of per-expert routing counts and every N calls swaps up to `adapt_swaps` of the hottest host experts into each
accelerator tier in place of its coldest ones (`backend.update`, available when `can_update`). The host tier holds
every expert (its local id is the global id), so any expert can be computed there or promoted from there.

## Adding support

| new ... | do |
|---|---|
| model family with the transformers >= 5 experts interface | nothing (GenericAdapter) |
| model family with another experts layout | `@register_adapter` class with `matches / spec / weights` |
| GPU type PyTorch supports | add its `mem_get_info` in `backends/cuda.py` and register an alias of `GpuBackend` |
| graph-compiled NPU | subclass `GraphNPUBackend`, implement the three hooks, `@register_backend` |
| any other device | subclass `ExpertBackend` (`capacity_bytes / load / run`, optional `update`) |
| faster kernel for some shapes | register a backend whose `supports(spec)` accepts those shapes (as `rngd-opt` does); set `slots_per_layer` if its tables are fixed-size |

## Place in opt_prime

`opt_prime.accel.moe` is the vendor-neutral expert-offload feature of `opt_prime.accel`: `accel.spec` describes the
model (ExpertSpec is part of it), the tier contract and registry are `accel.moe.backends`, and each vendor's tier lives
with that vendor in `accel.platforms.<vendor>.experts` (cuda, cpu with the native kernels, rngd, rngd-opt, atom).
`accel.engines` chooses a whole-request engine per model (Optimus with these tiers, or a vendor engine), and
`accel.platforms` leases chips to roles so expert tiers, PD endpoints and engines never share an accelerator.

## Status

| piece | state |
|---|---|
| GenericAdapter (transformers >= 5 experts interface) | 20 families verified (tests) |
| gpu backend (cuda / ROCm via torch.cuda / xpu) | CUDA verified on A100; sync-free gather+bmm for decode |
| cpu backend | native AVX-512 BF16 kernel (~60-70 GB/s, 12 threads), torch fallback |
| N-tier layer (async routing copy, no device stall), presence-aware planner, adaptive re-placement | done, tested; Qwen3-30B-A3B numbers in the README |
| rngd backend (furiosa-torch SlotFFN) | logic verified in emulation; graphs for Qwen3-30B-A3B and Solar-Open shapes compiled and cached without a device; silicon run pending a free NPU |
| rngd-opt (generated furiosa-opt kernels + expert server) | crates pass CPU emulation and silicon; chip pinned through FURIOSA_VISIBLE_CHIPS and opt_prime.accel.platforms leases |
| atom (RBLN SDK, `baked` per-expert graphs) | verified on one RBLN-CA22 (rebel-compiler 0.11): Qwen3-30B-A3B expert shape vs cpu backend PASS (0.008), ~210 us per expert; tiny qwen3_moe end to end with experts on atom + cpu PASS. User rebel.triton kernels do not pass device graph conversion on this SDK |
| rngd-opt on silicon | Qwen3 / Solar shapes PASS, 186-640 us per token (1-8 experts); Qwen3-30B-A3B rngd-opt x2 (64 slots) + cpu 15.4 tok/s (18.4 compiled), Solar-Open-100B rngd-opt x2 + cpu 8.6 tok/s exact |
| native host dispatcher (csrc/dispatch.c) | blocking tiers on persistent C threads, no GIL; +20-40 % end to end |
