"""Greedy, single-sequence Torch adapter using OptimusPrime KVCacheManager.

Supports dense full-attention GPT-2/Llama DynamicCache models. Model loading and
weights provenance are caller responsibilities. No vendor SDK is imported here.
"""
from dataclasses import dataclass
import torch
from transformers.cache_utils import DynamicCache
from opt_prime.kv_cache import KVCacheManager
from .runtime import KVLease, KVSpec, PrefillResult


@dataclass
class TorchCache:
    manager: KVCacheManager


def fence(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class TorchCausalLMEngine:
    def __init__(self, model, *, model_id, revision, max_context, endpoint, prefix_only=False):
        config = model.config
        if config.model_type not in ('gpt2', 'llama'):
            raise ValueError('adapter supports full-attention GPT-2 and Llama only')
        if getattr(config, 'sliding_window', None):
            raise ValueError('sliding window cache is not supported')
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
        self.layers = config.num_hidden_layers
        self.heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.spec = KVSpec(model_id, revision, str(self.dtype),
                           'torch-bhsd-full-attention-v1',
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
        output = self.model(torch.tensor([tokens], device=self.device), use_cache=True)
        cache = output.past_key_values
        if not isinstance(cache, DynamicCache) or len(cache.layers) != self.layers:
            raise ValueError('expected complete DynamicCache')
        for i, layer in enumerate(cache.layers):
            lease.payload.manager.update(i, layer.keys, layer.values, position=0)
        lease.payload.manager.set_seq_len(len(tokens))
        fence(self.device)
        self.last_logits = [] if self.prefix_only else [output.logits[0, -1].detach().cpu().clone()]
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
            logits = output.logits[0, -1].detach().cpu().clone()
            self.last_logits.append(logits)
            token = int(logits.argmax())
            generated.append(token)
        fence(self.device)
        return generated


class TorchCacheTransport:
    """Explicit tensor-copy transport; never selected as a P2P fallback."""
    def __init__(self, *, source, destination, spec):
        self.source, self.destination, self.spec = source, destination, spec

    def supports(self, source, destination, spec):
        return (source, destination, spec) == (self.source, self.destination, self.spec)

    @torch.inference_mode()
    def transfer(self, request_id, source, destination):
        if source.spec != destination.spec or not self.supports(source.endpoint, destination.endpoint, source.spec):
            raise ValueError('unconfigured tensor-copy route')
        src, dst = source.payload.manager, destination.payload.manager
        position = src.get_seq_len()
        if not 0 < position <= dst.max_seq_len:
            raise ValueError('invalid source sequence length')
        copied = 0
        for i in range(src.layer_start, src.layer_end):
            keys, values = src.get(i)
            dst.update(i, keys.to(dst.device), values.to(dst.device), position=0)
            copied += keys.numel() * keys.element_size() + values.numel() * values.element_size()
        dst.set_seq_len(position)
        fence(src.device)
        fence(dst.device)
        return dict(transport='torch-copy', bytes=copied, cache_position=position, completed=True)

    def finish(self, request_id):
        pass
