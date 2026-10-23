"""Extract real prompt state from a right-padded causal prefill bucket."""


def valid_prefix(keys, values, hidden, length):
    if not 0 < length <= hidden.shape[0]:
        raise ValueError("prompt length must fit the compiled width")
    if keys.shape != values.shape or keys.shape[2] != hidden.shape[0]:
        raise ValueError("KV and hidden widths must agree")
    return (
        keys[:, :, :length].contiguous(),
        values[:, :, :length].contiguous(),
        hidden[length - 1],
    )
