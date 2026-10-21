"""Greedy single-sequence Torch execution; model/cache policy lives in adapters.

Models are supplied by the caller. No downloads or vendor SDK imports occur here.
"""
import inspect
import torch
from transformers.cache_utils import DynamicCache
from opt_prime.kv_cache import KVCacheManager
from opt_prime.accel.pd.contracts import KVLease, KVSpec, PrefillResult
from opt_prime.accel.pd.sampling import select_token
from opt_prime.accel.pd.torch_models import describe_model
# Compatibility exports: existing users may import transport from this module.
from opt_prime.accel.pd.torch_cache import LayeredKVCache, TorchCache, TorchCacheTransport, fence


class TorchCausalLMEngine:
    supports_sampling = True          # SamplingParams beyond greedy, early stop at stop_token_ids

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
        text = config.get_text_config() if hasattr(config, 'get_text_config') else config
        if type(max_context) is not int or max_context < 2 or max_context > text.max_position_embeddings:
            raise ValueError('context exceeds model capacity')
        if self.geometry.window is not None and max_context > self.geometry.window:
            raise ValueError('context exceeds the sliding window: sliding layers would drop tokens')
        self.layers = self.geometry.layers
        self.heads = self.geometry.kv_heads
        self.head_dim = self.geometry.head_dim
        if self.geometry.uniform:
            layout = f'torch-bhsd-full-attention-v1:{self.geometry.model_type}'
        else:                         # per-layer shapes are part of the contract
            dims = ','.join(f'{h}x{d}' for h, d in self.geometry.layer_dims)
            layout = f'torch-bhsd-per-layer-v1:{self.geometry.model_type}:{dims}'
        self.spec = KVSpec(model_id, revision, str(self.dtype), layout,
                           (self.layers, 2, 1, self.heads, max_context, self.head_dim))
        self.model = model.eval()
        self.max_context, self.endpoint = max_context, endpoint
        self.prefix_only = prefix_only
        self.last_logits = []

    def _validate(self, request):
        if len(request.input_ids) + request.max_new_tokens > self.max_context:
            raise ValueError('request exceeds cache capacity')
        vocab = (self.model.config.get_text_config() if hasattr(self.model.config, 'get_text_config')
                 else self.model.config).vocab_size
        if any(t >= vocab for t in request.input_ids):
            raise ValueError('token outside model vocabulary')
        if self.prefix_only and len(request.input_ids) < 2:
            raise ValueError('prefix prefill requires at least two prompt tokens')

    def reserve(self, request):
        self._validate(request)
        if self.geometry.uniform:
            manager = KVCacheManager(self.layers, 1, self.max_context, self.heads,
                                     self.head_dim, dtype=self.dtype, device=self.device)
        else:
            manager = LayeredKVCache(self.geometry.layer_dims, self.max_context, self.dtype, self.device)
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
        token = select_token(output.logits[0, -1], request.sampling, len(tokens))
        return PrefillResult(lease, token, len(tokens), (token,))

    @torch.inference_mode()
    def decode(self, request, cache, continuation, on_token=None):
        self._validate(request)
        continuation.validate(request)
        manager = cache.payload.manager
        if cache.spec != self.spec or manager.get_seq_len() != continuation.cache_position:
            raise ValueError('cache position or specification mismatch')
        pairs = [manager.get(i) for i in range(self.layers)]
        # Plain layers for every layer: sliding layers hold the whole context (max_context <= window is enforced), so
        # they behave like full ones. (Passing config= is also broken in transformers 5.8 for num_kv_shared_layers=0:
        # layer_types[:-0] is empty.)
        dynamic = DynamicCache(ddp_cache_data=pairs)
        token = continuation.next_token
        generated = []
        self.last_logits = []
        stop = set(request.sampling.stop_token_ids)
        position = continuation.cache_position + 1
        for _ in range(request.max_new_tokens - len(continuation.output_prefix)):
            if token in stop and continuation.output_prefix:          # the prefill already emitted a stop token
                break
            output = self.model(torch.tensor([[token]], device=self.device),
                                past_key_values=dynamic, use_cache=True)
            dynamic = output.past_key_values
            logits = output.logits[0, -1]
            if self.capture_logits:
                self.last_logits.append(logits.detach().cpu().clone())
            token = select_token(logits, request.sampling, position)
            position += 1
            generated.append(token)
            if on_token is not None:
                on_token(token)
            if token in stop:
                break
        fence(self.device)
        return generated
