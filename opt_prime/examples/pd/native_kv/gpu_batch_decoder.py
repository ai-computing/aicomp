"""Experimental fixed-width CUDA graph with independent KV positions per lane."""
import types


class BatchGraphDecoder:
    def __init__(self, model, batch_size=4, capacity=2048, adaptive_single=False):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        if not 1 <= batch_size <= 16 or not 256 <= capacity <= 2048 or capacity % 256:
            raise ValueError('unsupported batch/capacity')
        self.model, self.batch_size, self.capacity = model, batch_size, capacity
        self.device = next(model.parameters()).device
        self.input_id = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        self.position = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.active = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.remaining = [0] * batch_size
        self.keys = [torch.zeros((batch_size, capacity, 8, 128), dtype=torch.bfloat16,
                                 device=self.device) for _ in range(36)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        lanes = torch.arange(batch_size, device=self.device)
        cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
        block_table = torch.arange(batch_size, dtype=torch.int32, device=self.device)[:, None]
        lengths = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        owner = self
        for layer in model.model.layers:
            module = layer.self_attn
            original = module.forward
            def forward(attn, hidden_states, position_embeddings, attention_mask,
                        past_key_values=None, cache_position=None, _original=original, **kwargs):
                if past_key_values is not owner:
                    return _original(hidden_states, position_embeddings, attention_mask,
                                     past_key_values, cache_position, **kwargs)
                width = hidden_states.shape[0]
                if hasattr(attn, 'qkv_proj'):
                    q, k, v = attn.qkv_proj(hidden_states).split(attn.qkv_sizes, dim=-1)
                else:
                    q, k, v = attn.q_proj(hidden_states), attn.k_proj(hidden_states), attn.v_proj(hidden_states)
                q = attn.q_norm(q.view(width, 1, -1, 128)).transpose(1, 2)
                k = attn.k_norm(k.view(width, 1, -1, 128)).transpose(1, 2)
                v = v.view(width, 1, 8, 128)
                q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
                kc, vc = owner.keys[attn.layer_idx], owner.values[attn.layer_idx]
                # Inactive lanes keep a bounded position and cannot affect another lane.
                kc[lanes[:width], owner.position[:width]] = k[:, :, 0]
                vc[lanes[:width], owner.position[:width]] = v[:, 0]
                out = flash_attn_varlen_func(q[:, :, 0], kc[:width], vc[:width],
                    max_seqlen_q=1, cu_seqlens_q=cu_q[:width+1], max_seqlen_k=capacity,
                    block_table=block_table[:width], seqused_k=lengths[:width], softmax_scale=attn.scaling,
                    causal=False, fa_version=2)
                return attn.o_proj(out.reshape(width, 1, -1)), None
            module.forward = types.MethodType(forward, module)
        def step(width=batch_size):
            lengths.copy_(self.position + 1)
            logits = model(self.input_id[:width], past_key_values=self, cache_position=self.position[:width],
                position_ids=self.position[:width, None], attention_mask={'full_attention': None},
                use_cache=True, logits_to_keep=1).logits
            self.last_logits = logits
            self.input_id[:width].copy_(torch.where(self.active[:width, None],
                logits[:, -1].argmax(-1, keepdim=True), self.input_id[:width]))
            self.position[:width].add_(self.active[:width]).clamp_(max=capacity - 1)
        with torch.inference_mode():
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3): step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph): step()
            self.single_graph = self.graph
            if batch_size > 1 and adaptive_single:
                with torch.cuda.stream(stream):
                    for _ in range(3): step(1)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                torch.cuda.synchronize(self.device)
                self.single_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.single_graph): step(1)

    def load_lane(self, lane, layers, last_token, count):
        import torch
        if not 0 <= lane < self.batch_size or self.remaining[lane]:
            raise ValueError('lane unavailable')
        if len(layers) != 36: raise ValueError('expected 36 layers')
        n = layers[0][0].shape[2]
        if not 1 <= n < self.capacity or not 1 <= count <= self.capacity - n:
            raise ValueError('prefix/output outside capacity')
        if not 0 <= last_token < self.model.config.vocab_size: raise ValueError('invalid token')
        for pair in layers:
            for t in pair:
                if t.shape != (1, 8, n, 128) or t.dtype != torch.bfloat16:
                    raise ValueError('invalid canonical KV')
        with torch.inference_mode():
            # One pinned upload; staging is fenced before reuse or caller release.
            host = torch.empty((72, n, 8, 128), dtype=torch.bfloat16, pin_memory=True)
            for i, pair in enumerate(layers):
                for j, t in enumerate(pair): host[2*i+j].copy_(t[0].transpose(0, 1))
            packed = host.to(self.device, non_blocking=True)
            for i in range(36):
                self.keys[i][lane, :n].copy_(packed[2*i])
                self.values[i][lane, :n].copy_(packed[2*i+1])
            self.input_id[lane].fill_(last_token)
            self.position[lane].fill_(n)
            self.active[lane].fill_(True)
        torch.cuda.synchronize(self.device)
        self.remaining[lane] = count

    def cancel_lane(self, lane):
        if not 0 <= lane < self.batch_size: raise ValueError('invalid lane')
        self.remaining[lane] = 0
        self.active[lane].fill_(False)

    def step(self, eos_ids=(), ignore_eos=False, paused=()):
        paused = set(paused)
        if any(not 0 <= lane < self.batch_size for lane in paused):
            raise ValueError('invalid paused lane')
        if not any(n and lane not in paused for lane, n in enumerate(self.remaining)): return {}
        for lane in paused: self.active[lane].fill_(False)
        (self.graph if any(self.remaining[1:]) else self.single_graph).replay()
        tokens = self.input_id[:, 0].tolist()
        for lane in paused:
            if self.remaining[lane]: self.active[lane].fill_(True)
        result = {}
        for lane, remaining in enumerate(self.remaining):
            if not remaining or lane in paused: continue
            result[lane] = tokens[lane]
            self.remaining[lane] -= 1
            if not self.remaining[lane] or (not ignore_eos and tokens[lane] in eos_ids):
                self.cancel_lane(lane)
        return result
