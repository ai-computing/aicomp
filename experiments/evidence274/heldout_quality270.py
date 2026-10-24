"""Preregistered synthetic structured-answer checks; not a general quality benchmark."""
import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path


def score(text, expected):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        value = None
    valid = isinstance(value, dict) and isinstance(value.get('answer'), str)
    return dict(answer_correct=bool(valid and value['answer'] == expected),
                format_correct=bool(valid and set(value) == {'answer'}))


def compare(baseline, candidate):
    left = {x['case']: x for x in baseline}
    right = {x['case']: x for x in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError('duplicate or missing paired cases')
    result = dict(cases=len(left), correct_to_wrong=0, wrong_to_correct=0, token_equal=0)
    for key, a in left.items():
        b = right[key]
        if a['input_sha256'] != b['input_sha256']:
            raise ValueError('paired input IDs differ')
        result['correct_to_wrong'] += bool(a['answer_correct'] and not b['answer_correct'])
        result['wrong_to_correct'] += bool(not a['answer_correct'] and b['answer_correct'])
        result['token_equal'] += a['tokens'] == b['tokens']
    return result


def fixtures(tokenizer):
    rng = random.Random(26320261024)
    tasks = []
    for variant in range(4):
        a, b = rng.randrange(21, 80), rng.randrange(11, 60)
        tasks.append(('arithmetic', variant, '', f'Compute {a} + {b}.', str(a + b)))
        secret = ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
        tasks.append(('passkey', variant, f'The access key for archive {variant} is {secret}.',
                      f'What is the access key for archive {variant}?', secret))
        city = ['춘천', '목포', '전주', '속초'][variant]
        tasks.append(('korean_table', variant,
                      f'배송표: 주문 가{variant}=대전; 주문 나{variant}={city}; 주문 다{variant}=부산.',
                      f'배송표에서 주문 나{variant}의 도착 도시를 찾으세요.', city))
    filler = 'Background: shelves hold ordinary books; this note contains no task answer.\n'
    result = []
    for target in [0, 2048, 8192]:
        for task, variant, fact, question, expected in tasks:
            def render(repeats):
                prompt = (filler * (repeats // 2) + fact + '\n' + filler * (repeats - repeats // 2)
                          + '\n' + question + '\nReturn only a JSON object with one key "answer" and a string value. No explanation.')
                ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                return prompt, [int(x) for x in ids]
            low, high = 0, target
            if target and len(render(0)[1]) > target:
                raise ValueError('base prompt exceeds context target')
            while low < high:
                mid = (low + high + 1) // 2
                if len(render(mid)[1]) <= target:
                    low = mid
                else:
                    high = mid - 1
            prompt, ids = render(low)
            digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            result.append(dict(case=f'{task}-{variant}-{target}', task=task, target_tokens=target,
                expected=expected, prompt=prompt, ids=ids, input_sha256=digest))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('fixtures')
    export.add_argument('--model', required=True)
    export.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--fixtures', type=Path, required=True)
    run.add_argument('--endpoint', required=True)
    run.add_argument('--route', required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fixtures':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        data = dict(model=args.model, seed=26320261024, max_tokens=128,
                    thinking=False, cases=fixtures(tokenizer))
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        return
    data = json.loads(args.fixtures.read_text())
    # Exclusive creation prevents silently replacing a prior run after seeing answers.
    with args.output.open('x') as output:
        for case in data['cases']:
            body = dict(model=data['model'], prompt=case['ids'], temperature=0,
                        max_tokens=data['max_tokens'], ignore_eos=False, stream=False)
            request = urllib.request.Request(args.endpoint.rstrip('/') + '/v1/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
                    'x-request-id': 'quality263-' + case['case']})
            start = time.perf_counter()
            row = dict(case=case['case'], route=args.route, input_sha256=case['input_sha256'],
                       input_tokens=len(case['ids']), expected=case['expected'])
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.load(response)
                choice = payload['choices'][0]
                row.update(text=choice['text'], tokens=choice['token_ids'],
                           finish_reason=choice['finish_reason'], response=payload,
                           **score(choice['text'], case['expected']))
            except Exception as error:
                row.update(error=repr(error), answer_correct=False, format_correct=False)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                raise
            row['elapsed_seconds'] = time.perf_counter() - start
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(case['case'], row['answer_correct'], row['format_correct'], flush=True)


if __name__ == '__main__':
    main()
