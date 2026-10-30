"""Opt-in streaming adapters for the validated four-model research producers.

Requires the existing serving_matrix modules and Qwen prefill module on PYTHONPATH.
Factories preserve each producer's whole-result path for same-engine A/B controls.
Use --engine factory --engine-factory streaming_factories:qwen (or gemma/koni).
"""
import torch


class Streaming:
    def __init__(self, engine, stream):
        self.engine, self.stream = engine, stream

    def prefill(self, tokens):
        return self.engine.prefill(tokens)

    @torch.inference_mode()
    def prefill_stream(self, tokens, emit):
        return self.stream(self.engine, tokens, emit)


def _ids(engine, tokens):
    ids = list(tokens)
    if not 0 < len(ids) <= engine.width or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('invalid tokens or prompt exceeds compiled width')
    return ids


def qwen_stream(e, tokens, emit):
    ids = _ids(e, tokens)
    n = len(ids)
    x = e.embedding[ids + [ids[-1]] * (e.width - n)].to(torch.bfloat16).to(e.device)
    for i, layer in enumerate(e.layers):
        x, kv = e.module(x, *e.fixed, *layer, e.rotation, e.mask)
        k = kv[0].cpu()[..., :n].transpose(-1, -2).unsqueeze(0)
        v = kv[1].cpu()[..., :n].transpose(-1, -2).unsqueeze(0)
        emit(i, 0, k, v)
    e.sync(e.chip)
    z = x.cpu()[n - 1].float()
    z = z * torch.rsqrt(z.square().mean() + e.s['rms_norm_eps']) * e.norm
    return z @ e.head.T


def gemma_stream(e, tokens, emit):
    ids = _ids(e, tokens)
    from gemma_parts import initial_hidden, final_logits
    n = len(ids)
    if any(t >= e.config['vocab_size'] for t in ids):
        raise ValueError('invalid token ID')
    pad = e.config.get('pad_token_id')
    x = initial_hidden(e.weights['model.embed_tokens.weight'], ids + [0 if pad is None else pad] * (e.width - n)).to(e.device)
    for i, layer in enumerate(e.params):
        x, k, v = e.compiled(x, *layer, *e.positions[e.local[i]])
        emit(i, 0, k.cpu()[:n].permute(1, 0, 2).unsqueeze(0),
             v.cpu()[:n].permute(1, 0, 2).unsqueeze(0))
    e.sync(e.chip)
    return final_logits(x.cpu()[n - 1], e.weights['model.norm.weight'],
                        e.weights.get('lm_head.weight', e.weights['model.embed_tokens.weight']), e.config['rms_norm_eps'])


def koni_stream(e, tokens, emit):
    ids = _ids(e, tokens)
    n = len(ids)
    if any(t >= e.config['vocab_size'] for t in ids):
        raise ValueError('invalid token ID')
    stage = e.engine.begin_device_prefill(ids)
    layer = 0
    while True:
        done = stage.advance()
        k = stage.Ks[-1].cpu()[..., :n].transpose(-1, -2)
        v = stage.Vs[-1].cpu()[:, :, :n]
        emit(layer, 0, k, v)
        layer += k.shape[0]
        if done:
            break
    e.sync(e.chip)
    z = stage.x.cpu()[n - 1].float()
    z = z * torch.rsqrt(z.square().mean() + e.config['rms_norm_eps']) * e.norm.float()
    return z @ e.head.float().T


def qwen(config, weights, width, chip=0):
    from long_width_variant import make_prefill
    from rngd_pd.engines.fsdk import sync
    engine = make_prefill(config, weights, width, chip)
    engine.sync = sync
    return Streaming(engine, qwen_stream)


def gemma(config, weights, width, chip=0):
    from gemma_prefill import Prefill
    return Streaming(Prefill(config, weights, width, chip), gemma_stream)


def koni(config, weights, width, chip=0):
    from koni_prefill import Prefill
    return Streaming(Prefill(config, weights, width, chip), koni_stream)
