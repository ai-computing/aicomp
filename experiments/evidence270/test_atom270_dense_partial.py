"""CPU coverage of dense B4 ownership after an actual partial cache write.

Uses the existing fake SDK and real AtomBatcher; not NPU execution or model quality.
"""
import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.runtime import RequestCancelled
from opt_prime.accel.platforms.atom.batching import AtomBatcher
from test_atom_batching import FakeBlocks, _llama, _prompts, fake_rebel
from test_atom_stream_progress import Producer


@pytest.mark.parametrize('fault', ['cancel', 'error'])
def test_three_survivors_and_reuse_after_fourth_lane_partial_write(fake_rebel, fault):
    prompts = _prompts(5, 20, 35)

    def run(outcome):
        model = _llama()
        fake = FakeBlocks(model, batch=4, blocks=5)
        fake_rebel['model'] = fake
        producer = Producer(model)
        batch = AtomBatcher(fake, prefix_cache=False, prefill_worker=producer, local_prefill=False)
        alive = [batch.submit(PDRequest(str(i), prompts[i], 24)) for i in range(3)]
        victim = batch.submit(PDRequest('victim', prompts[3], 3))
        batch.step()
        assert batch.stats['progress_spans'] == 4
        r = batch.by_id['victim']
        block = batch.blocks.lane_block[r.slot]
        assert r.slot == 3 and not getattr(r, 'admitted', False) and not victim.done()
        torch.testing.assert_close(fake.kv[0][block, :, :len(prompts[3])], producer.jobs[3][2].keys[0].float())
        for i in range(3):
            producer.finish(i)
        batch.step()
        assert len(batch.decoding) == 3 and all(0 < len(r.tokens) < 24 for r in batch.decoding)
        if outcome == 'control':
            producer.finish(3)
        else:
            if outcome == 'cancel':
                assert batch.cancel('victim')
            else:
                producer.jobs[3][0].set_exception(RuntimeError('producer failed after partial import'))
            batch.step()
            with pytest.raises(RequestCancelled if outcome == 'cancel' else RuntimeError):
                victim.result()
            replacement = batch.submit(PDRequest('replacement', prompts[4], 3))
            batch.step()
            new = batch.by_id['replacement']
            assert new.slot == 3 and not getattr(new, 'admitted', False)
            new_block = batch.blocks.lane_block[3]
            snapshot = [k[new_block].clone() for k in fake.kv]
            if outcome == 'cancel':
                producer.finish(3)  # Late old result must not touch the reused destination.
            batch.step()
            for k, before in zip(fake.kv, snapshot):
                torch.testing.assert_close(k[new_block], before)
            producer.finish(4)
        batch.run_until_idle()
        assert batch.idle and not batch.poisoned and sorted(batch.free) == [0, 1, 2, 3]
        assert not batch.by_id and not batch.blocks.index
        if outcome != 'control':
            assert len(replacement.result()) == 3
        return [f.result() for f in alive]

    assert run(fault) == run('control')
