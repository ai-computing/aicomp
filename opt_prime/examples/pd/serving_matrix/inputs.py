"""Exact-size token fixtures shared by all backend runners."""


def exact_prompt(prefix, filler, suffix, length):
    n = length - len(prefix) - len(suffix)
    if n < 0 or not filler:
        raise ValueError("capacity must hold prefix/question; filler must be nonempty")
    return (
        list(prefix)
        + (list(filler) * ((n + len(filler) - 1) // len(filler)))[:n]
        + list(suffix)
    )


def through_eos(tokens, eos_ids):
    for i, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[: i + 1]
    return tokens[:]
