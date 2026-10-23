"""Owned GPU-prefix/native-suffix cache assembly for an opt-in experiment."""
import torch


@torch.inference_mode()
def load_split_prefix(graph, lane, prefix, suffix, prefix_length, last_token, count):
    if type(lane) is not int or not 0 <= lane < graph.batch_size or graph.remaining[lane]:
        raise ValueError('lane unavailable')
    if type(last_token) is not int or not 0 <= last_token < graph.model.config.vocab_size:
        raise ValueError('invalid last token')
    if type(count) is not int: raise ValueError('invalid output count')
    if type(prefix_length) is not int or not 1 <= prefix_length < graph.capacity:
        raise ValueError('invalid prefix length')
    if len(prefix) != 36 or any(len(pair) != 2 for pair in prefix):
        raise ValueError('expected36 GPU prefix K/V pairs')
    for pair in prefix:
        for value in pair:
            if (value.shape != (prefix_length,8,128) or value.dtype != torch.bfloat16
                    or value.device != graph.device):
                raise ValueError('invalid GPU prefix tensor')
    if (suffix.ndim != 4 or suffix.shape[0] != 72 or suffix.shape[2:] != (8,128)
            or suffix.dtype != torch.bfloat16 or suffix.device.type != 'cpu'
            or not suffix.is_contiguous() or not suffix.is_pinned('cuda')):
        raise ValueError('invalid owned pinned suffix')
    n=prefix_length+suffix.shape[1]
    if not 1 <= suffix.shape[1] or n >= graph.capacity or not 1 <= count <= graph.capacity-n:
        raise ValueError('combined prefix/output exceeds capacity')
    packed=suffix.to(graph.device,non_blocking=True)
    layers=[(torch.cat((key,packed[2*i]),dim=0),torch.cat((value,packed[2*i+1]),dim=0))
            for i,(key,value) in enumerate(prefix)]
    # This fences all copies before releasing either producer's buffers.
    graph.load_device_lane(lane,layers,last_token,count)
