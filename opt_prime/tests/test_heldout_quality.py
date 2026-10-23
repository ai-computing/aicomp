"""Prevent answer substrings, duplicate keys, or missing pairs from inflating scores."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).parents[1] / 'examples/pd/http_matrix/heldout_quality.py'

def module():
    assert path.exists(), 'held-out evaluator is not implemented'
    spec = importlib.util.spec_from_file_location('heldout_quality', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

@pytest.mark.parametrize('text,correct,formatted', [
    ('{"answer":"73"}', True, True),
    ('{"answer":"73","note":"ok"}', True, False),
    ('{"answer":73}', False, False),
    ('The answer is 73', False, False),
    ('{"answer":"173"}', False, True),
    ('{"answer":"wrong","answer":"73"}', False, False),
    ('{"answer":"73"} trailing', False, False),
])
def test_score(text, correct, formatted):
    got = module().score(text, '73')
    assert (got['answer_correct'], got['format_correct']) == (correct, formatted)

def test_paired_regressions_are_not_hidden_by_equal_aggregate():
    m = module()
    base = [{'case': 'a', 'input_sha256': 'x', 'answer_correct': True, 'tokens': [1]},
            {'case': 'b', 'input_sha256': 'y', 'answer_correct': False, 'tokens': [2]}]
    other = [{'case': 'a', 'input_sha256': 'x', 'answer_correct': False, 'tokens': [3]},
             {'case': 'b', 'input_sha256': 'y', 'answer_correct': True, 'tokens': [2]}]
    got = m.compare(base, other)
    assert got == {'cases': 2, 'correct_to_wrong': 1, 'wrong_to_correct': 1, 'token_equal': 1}
    with pytest.raises(ValueError):
        m.compare(base, other[:1])
    other[0]['input_sha256'] = 'changed'
    with pytest.raises(ValueError):
        m.compare(base, other)

def test_context_builder_keeps_retrieval_fact_inside_budget():
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(messages[0]['content'].encode())
    m = module()
    cases = m.fixtures(CharacterTokenizer())
    assert len(cases) == len({x['case'] for x in cases}) == 36
    for case in cases:
        assert len(case['ids']) <= (case['target_tokens'] or 2048)
        if case['task'] != 'arithmetic':
            assert case['expected'] in case['prompt']
        if case['target_tokens']:
            assert len(case['ids']) > case['target_tokens'] - 150

def test_http_runner_supplies_required_request_id_and_frozen_tokens(tmp_path, monkeypatch):
    import json
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.headers.get('x-request-id'), body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'text': '{"answer":"73"}',
                'token_ids': [9], 'finish_reason': 'stop'}]}).encode())
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fixture = tmp_path / 'input.json'
    output = tmp_path / 'output.jsonl'
    fixture.write_text(json.dumps({'model':'model', 'max_tokens':128,
        'cases':[{'case':'a','ids':[4,5], 'input_sha256':'hash','expected':'73'}]}))
    monkeypatch.setattr(sys, 'argv', ['heldout_quality', 'run', '--fixtures',str(fixture),
        '--endpoint',f'http://127.0.0.1:{server.server_port}', '--route','test', '--output',str(output)])
    try:
        module().main()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert seen[0][0], 'stage HTTP API requires a nonempty request ID'
    assert seen[0][1]['prompt'] == [4,5]
    assert seen[0][1]['ignore_eos'] is False
    assert json.loads(output.read_text())['answer_correct'] is True
    with pytest.raises(FileExistsError):
        module().main()
