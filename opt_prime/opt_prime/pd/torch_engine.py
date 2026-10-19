"""Greedy single-sequence Torch execution; model/cache policy lives in adapters.

Models are supplied by the caller. No downloads or vendor SDK imports occur here.
"""
import inspect
import torch
from transformers.cache_utils import DynamicCache
from opt_prime.kv_cache import KVCacheManager
from .contracts import KVLease, KVSpec, PrefillResult
from .torch_models import describe_model
# Compatibility exports: existing users may import transport from this module.
from .torch_cache import TorchCache, TorchCacheTransport, fence


class TorchCausalLMEngine:
    def __init__(self, model, *, model_id, revision, max_context, endpoint, prefix_only=False,
                 optimize_prefill=True, capture_logits=False):
        config = model.config
        self.geometry = describe_model(config)
        self.optimize_prefill = optimize_prefill
        self.capture_logits = capture_logits
        if optimize_prefill and 'logits_to_keep' not in inspect.signature(model.forward).parameters:
            raise ValueError('model forward does not support last-token logits optimization')
        devices = {p.device for p in model.parameters()}
        dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError('one device and one weight precision per engine required')
        self.device, = devices
        self.dtype, = dtypes
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('this adapter supports torch CPU/CUDA devices')
        if type(max_context) is not int or max_context < 2 or max_context > config.max_position_embeddings:
            raise ValueError('context exceeds model capacity')
        self.layers = self.geometry.layers
        self.heads = self.geometry.kv_heads
        self.head_dim = self.geometry.head_dim
        self.spec = KVSpec(model_id, revision, str(self.dtype),
                           f'torch-bhsd-full-attention-v1:{self.geometry.model_type}',
                           (self.layers, 2, 1, self.heads, max_context, self.head_dim))
        self.model = model.eval()
        self.max_context, self.endpoint = max_context, endpoint
        self.prefix_only = prefix_only
        self.last_logits = []

    def _validate(self, request):
        if len(request.input_ids) + request.max_new_tokens > self.max_context:
            raise ValueError('request exceeds cache capacity')
        if any(t >= self.model.config.vocab_size for t in request.input_ids):
            raise ValueError('token outside model vocabulary')
        if self.prefix_only and len(request.input_ids) < 2:
            raise ValueError('prefix prefill requires at least two prompt tokens')

    def reserve(self, request):
        self._validate(request)
        manager = KVCacheManager(self.layers, 1, self.max_context, self.heads,
                                 self.head_dim, dtype=self.dtype, device=self.device)
        self._pending_manager = manager
        manager.allocate(1)
        return KVLease(self.spec, manager, TorchCache(manager), self.endpoint)

    @torch.inference_mode()
    def prefill(self, request):
        lease = self.reserve(request)
        tokens = request.input_ids[:-1] if self.prefix_only else request.input_ids
        kwargs = {'logits_to_keep': 1} if self.optimize_prefill else {}
        output = self.model(torch.tensor([tokens], device=self.device), use_cache=True, **kwargs)
        cache = output.past_key_values
        if not isinstance(cache, DynamicCache) or len(cache.layers) != self.layers:
            raise ValueError('expected complete DynamicCache')
        for i, layer in enumerate(cache.layers):
            lease.payload.manager.update(i, layer.keys, layer.values, position=0)
        lease.payload.manager.set_seq_len(len(tokens))
        fence(self.device)
        self.last_logits = []
        if self.capture_logits and not self.prefix_only:
            self.last_logits.append(output.logits[0, -1].detach().cpu().clone())
        if self.prefix_only:
            return PrefillResult(lease, request.input_ids[-1], len(tokens))
        token = int(output.logits[0, -1].argmax())
        return PrefillResult(lease, token, len(tokens), (token,))

    @torch.inference_mode()
    def decode(self, request, cache, continuation):
        self._validate(request)
        continuation.validate(request)
        manager = cache.payload.manager
        if cache.spec != self.spec or manager.get_seq_len() != continuation.cache_position:
            raise ValueError('cache position or specification mismatch')
        pairs = [manager.get(i) for i in range(self.layers)]
        dynamic = DynamicCache(ddp_cache_data=pairs)
        token = continuation.next_token
        generated = []
        self.last_logits = []
        for _ in range(request.max_new_tokens - len(continuation.output_prefix)):
            output = self.model(torch.tensor([[token]], device=self.device),
                                past_key_values=dynamic, use_cache=True)
            dynamic = output.past_key_values
            logits = output.logits[0, -1]
            if self.capture_logits:
                self.last_logits.append(logits.detach().cpu().clone())
            token = int(logits.argmax())
            generated.append(token)
        fence(self.device)
        return generated
