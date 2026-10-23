"""Match full producer KV to consumer retention without losing absolute position."""


def retained_export(exported, layer):
    n = exported.shape[-2]
    if int(layer.get_seq_length()) != n:
        raise ValueError("consumer absolute cache position differs from export")
    kept = min(n, layer.sliding_window - 1) if layer.is_sliding else n
    if tuple(layer.keys.shape[1:]) != (exported.shape[0], kept, exported.shape[2]):
        raise ValueError("consumer retained cache geometry differs")
    return exported[:, n - kept :]
