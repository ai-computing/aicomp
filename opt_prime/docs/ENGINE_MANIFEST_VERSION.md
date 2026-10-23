# Engine geometry manifests (version 2)

The engine geometry contract now includes ordered per-layer specs and `geometry_version: 2`. This prevents a binary compiled for one sliding/full-attention order, normalization epsilon, bias configuration, or RoPE configuration from being selected for another model that has the same set of block shapes.

Existing RNGD `engine_manifest.json` files must be regenerated with `engine_geometry(model_spec(model))` for the exact model/configuration used to build the engine. Do not merely add a version field to an old manifest: verify its compiled configuration first. Older manifests fail the existing equality check rather than being silently accepted.

RNGD startup enforces `ready_timeout`, including an incomplete READY line. Startup failures stop and reap the child and release acquired leases. ATOM startup failures release acquired leases too. ATOM leases continue to use physical card IDs, while SDK device arguments use the host-group visible NPU IDs discovered from RSD.
