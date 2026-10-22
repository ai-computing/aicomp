"""Resumable token-chunked BF16 prefill; one GPU owner schedules its steps."""
import types
import torch


class ChunkedPrefiller:
    def __init__(self, local):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        self.local = local
        owner = self
        for layer in local.model.model.layers:
            original = layer.self_attn.forward
            def forward(attn, hidden_states, position_embeddings, attention_mask,
                        past_key_values=None, cache_position=None, _original=original, **kwargs):
                state = past_key_values
                if not isinstance(state, PrefillSession) or state.owner is not owner:
                    return _original(hidden_states, position_embeddings, attention_mask,
                                     past_key_values, cache_position, **kwargs)
                total = hidden_states.shape[1]
                q = attn.q_norm(attn.q_proj(hidden_states).view(1,total,32,128)).transpose(1,2)
                k = attn.k_norm(attn.k_proj(hidden_states).view(1,total,8,128)).transpose(1,2)
                v = attn.v_proj(hidden_states).view(total,8,128)
                q,k = apply_rotary_pos_emb(q,k,*position_embeddings)
                q = q.transpose(1,2).reshape(total,32,128).contiguous()
                k = k.transpose(1,2).reshape(total,8,128).contiguous()
                keys, values = state.layers[attn.layer_idx]
                if state.paged:
                    batch, capacity = len(state.prompts), len(state.prompts[0])
                    begin, end = state.processed[0], state.ends[0]
                    kc = keys.view(batch,capacity,8,128)
                    vc = values.view(batch,capacity,8,128)
                    kc[:,begin:end].copy_(k.view(batch,end-begin,8,128))
                    vc[:,begin:end].copy_(v.view(batch,end-begin,8,128))
                    out = flash_attn_varlen_func(q,kc,vc,cu_seqlens_q=state.cu_q,
                        block_table=state.block_table,seqused_k=state.key_lengths,
                        max_seqlen_q=state.max_q,max_seqlen_k=capacity,
                        softmax_scale=attn.scaling,causal=True,fa_version=2)
                else:
                    kp, vp = [], []
                    for slot, lane in enumerate(state.active):
                        begin = state.offsets[lane]
                        end = state.ends[lane]
                        a,b = state.q_offsets[slot:slot+2]
                        keys[begin+state.processed[lane]:begin+end].copy_(k[a:b])
                        values[begin+state.processed[lane]:begin+end].copy_(v[a:b])
                        kp.append(keys[begin:begin+end]); vp.append(values[begin:begin+end])
                    # Only initialized prefixes enter attention. FA2 causal aligns the
                    # shorter query suffix with the bottom-right of each key prefix.
                    out = flash_attn_varlen_func(q,torch.cat(kp),torch.cat(vp),
                        cu_seqlens_q=state.cu_q,cu_seqlens_k=state.cu_k,
                        max_seqlen_q=state.max_q,max_seqlen_k=state.max_k,
                        softmax_scale=attn.scaling,causal=True,fa_version=2)
                return attn.o_proj(out.reshape(1,total,-1)),None
            layer.self_attn.forward = types.MethodType(forward,layer.self_attn)

    def start(self, prompts, chunk_tokens, *, paged=True):
        return PrefillSession(self, prompts, chunk_tokens, paged)


class PrefillSession:
    def __init__(self, owner, prompts, chunk_tokens, paged):
        self.owner = owner
        self.model = owner.local.model
        self.device = owner.local.device
        if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 2048:
            raise ValueError('invalid token chunk size')
        if not 1 <= len(prompts) <= 16 or any(not 1 <= len(p) <= 2048 for p in prompts):
            raise ValueError('invalid prompt lengths')
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for p in prompts for t in p):
            raise ValueError('invalid token')
        self.prompts = [list(p) for p in prompts]
        self.chunk_tokens = chunk_tokens
        self.offsets = [0]
        for prompt in prompts: self.offsets.append(self.offsets[-1]+len(prompt))
        self.paged = paged and len({len(p) for p in prompts}) == 1 and len(prompts[0]) % 256 == 0
        self.block_table = torch.arange(len(prompts),dtype=torch.int32,device=self.device)[:,None]
        self.processed = [0]*len(prompts)
        self.layers = [(torch.empty((self.offsets[-1],8,128),dtype=torch.bfloat16,device=self.device),
                        torch.empty((self.offsets[-1],8,128),dtype=torch.bfloat16,device=self.device)) for _ in range(36)]
        self.last_hidden = torch.empty((len(prompts),4096),dtype=torch.bfloat16,device=self.device)
        self.done = False

    @torch.inference_mode()
    def step(self):
        """Return None at a scheduling boundary, or owned final KV/token/logits."""
        if self.done: raise RuntimeError('session already complete')
        self.active = [i for i,p in enumerate(self.prompts) if self.processed[i] < len(p)]
        self.ends = [min(len(p),n+self.chunk_tokens) for p,n in zip(self.prompts,self.processed)]
        ids, positions = [], []
        self.q_offsets = [0]; k_offsets = [0]
        for lane in self.active:
            a,b = self.processed[lane],self.ends[lane]
            ids.extend(self.prompts[lane][a:b]); positions.extend(range(a,b))
            self.q_offsets.append(len(ids)); k_offsets.append(k_offsets[-1]+b)
        self.cu_q = torch.tensor(self.q_offsets,dtype=torch.int32,device=self.device)
        self.cu_k = torch.tensor(k_offsets,dtype=torch.int32,device=self.device)
        self.key_lengths = torch.tensor([self.ends[i] for i in self.active],dtype=torch.int32,device=self.device)
        self.max_q = max(self.ends[i]-self.processed[i] for i in self.active)
        self.max_k = max(self.ends[i] for i in self.active)
        hidden = self.model.model(torch.tensor([ids],device=self.device),past_key_values=self,
            position_ids=torch.tensor([positions],device=self.device),
            cache_position=torch.arange(len(ids),device=self.device),
            attention_mask={'full_attention':None},use_cache=True).last_hidden_state[0]
        for slot,lane in enumerate(self.active):
            self.processed[lane] = self.ends[lane]
            if self.ends[lane] == len(self.prompts[lane]):
                self.last_hidden[lane].copy_(hidden[self.q_offsets[slot+1]-1])
        if any(n < len(p) for n,p in zip(self.processed,self.prompts)): return None
        logits = self.model.lm_head(self.last_hidden)
        tokens = logits.argmax(-1).tolist()
        self.done = True
        return tokens,self.layers,self.offsets,logits
