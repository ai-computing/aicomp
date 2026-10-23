"""Small shared semantic fixtures and actual EOS stopping (not a quality suite)."""

QUESTIONS = [
    ("capital", "What is the capital of France? Reply with only the city name."),
    ("arithmetic", "Compute 17 + 25. Reply with only the integer."),
    ("korean", "대한민국의 수도는 어디인가요? 도시 이름만 답하세요."),
    (
        "retrieval",
        "The secret label is MAPLE. What is the secret label? Reply with only the label.",
    ),
]


def generate_until_stop(first, step, eos_ids, limit):
    if limit <= 0:
        raise ValueError("limit must be positive")
    tokens = [first]
    while tokens[-1] not in eos_ids and len(tokens) < limit:
        tokens.append(step(tokens[-1]))
    return tokens, "eos" if tokens[-1] in eos_ids else "length"


def fixtures(tokenizer):
    filler = "Background note: the blue box contains books and the red box contains apples.\n"
    for repeats in [0, 100, 400]:
        for name, question in QUESTIONS:
            text = filler * repeats + "\n" + question
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            yield {"case": f"{name}-{repeats}", "ids": ids, "question": question}


def eos_ids(configured, tokenizer_eos):
    values = list(configured) if isinstance(configured, list) else [configured]
    return sorted({x for x in [*values, tokenizer_eos] if x is not None})
