"""Research import target for an exclusively owned, static B1 ATOM cache.

Reuse the existing staging/import arithmetic, without creating a scheduler or
pretending that this one-block artifact supports prefix caching or concurrency.
"""


class SingleBlockTarget:
    def __init__(self, model):
        rc = model.rbln_config
        if (rc.batch_size, rc.kvcache_num_blocks, rc.cache_impl) != (1, 1, "static"):
            raise ValueError("requires static B1 / one-block artifact")
        self.metas = [
            dict(m) if isinstance(m, dict) else vars(m).copy() for m in rc.kvcache_metas
        ]
        if not self.metas or len(self.metas) % 2:
            raise ValueError("requires full key/value pairs")
        shape = self.metas[0]["shape"]
        for i, m in enumerate(self.metas):
            if (
                len(shape) != 4
                or shape[0] != 1
                or shape[2] != rc.max_seq_len
                or min(shape) < 1
                or m["shape"] != shape
                or m["name"] != f"past_key_values_{i}"
                or m["dtype"] != "bfloat16"
                or m.get("layer_type", "full_attention") != "full_attention"
            ):
                raise ValueError("requires uniform full-attention cache geometry")
        self.io_rt = model.model[1]
        self.lane_block = [0]
        self._stage = None

    def _staging(self):
        from opt_prime.accel.platforms.atom.batching import AtomKVBlocks

        return AtomKVBlocks._staging(self)

    def import_span(self, lane, start, keys, values):
        from opt_prime.accel.platforms.atom.batching import AtomKVBlocks

        return AtomKVBlocks.import_span(self, lane, start, keys, values)
