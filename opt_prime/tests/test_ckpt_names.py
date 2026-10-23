#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.opti_pri checkpoint naming: TP shards get their own files, only DP rank 0 writes, and the directory
# name never contains "None". Ranks are laid out as the (pp, dp, tp) device mesh: rank = (pp*dp + dp_i)*tp + tp_i.
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime import opti_pri  # noqa: E402
from opt_prime.opti_pri import Optimus_p  # noqa: E402


def _rank(rank, ckpt_dir, tp_size=2, dp_size=2):
    o = Optimus_p.__new__(Optimus_p)                    # no process group: just the state the save/load paths read
    o.tpl = SimpleNamespace(rank=rank, stage=0, tp_size=tp_size, dp_size=dp_size)
    o.checkpoint, o.ckpt_dir, o.lora_config, o.device = True, str(ckpt_dir), "lora", torch.device("cpu")
    torch.manual_seed(rank)                             # every rank holds different weights
    submod = nn.ModuleDict({"lora_A": nn.Linear(2, 2)})
    o.run_info = SimpleNamespace(submod=submod)
    o.optimizer = torch.optim.SGD(submod.parameters(), lr=0.1)
    return o


def _weight(o):
    return o.run_info.submod["lora_A"].weight.detach().clone()


def test_save_ckpt_one_file_per_tp_shard_from_dp_rank_0(tmp_path):
    ranks = [_rank(r, tmp_path) for r in range(4)]      # tp_i = r % 2, dp_i = r // 2
    for o in ranks:
        o.save_ckpt(1, 0)
    assert sorted(os.listdir(tmp_path)) == ["ckpt_step_1_epoch_0_tp0.pt", "ckpt_step_1_epoch_0_tp1.pt"]
    expected = _weight(ranks[1])                        # TP rank 1 of DP rank 0
    reload = _rank(3, tmp_path)                         # TP rank 1 of DP rank 1 reloads the same shard
    reload.load_ckpt(1, 0)
    assert torch.equal(_weight(reload), expected)


def test_save_ckpt_pp_only_name_unchanged(tmp_path):
    _rank(0, tmp_path, tp_size=1, dp_size=1).save_ckpt(1, 0)   # existing PP-only checkpoints keep loading
    assert os.listdir(tmp_path) == ["ckpt_step_1_epoch_0.pt"]


def test_save_lora_ckpt_only_dp_rank_0_writes(tmp_path):
    ranks = [_rank(r, tmp_path) for r in range(4)]
    for o in ranks:
        o.save_lora_ckpt(1, 0, lora_dir=str(tmp_path / "lora"))
    files = sorted(os.listdir(str(tmp_path / "lora") + "_stage_0"))
    assert files == ["lora_step_1_epoch_0_tp0.pt", "lora_step_1_epoch_0_tp1.pt"]
    expected = _weight(ranks[1])
    reload = _rank(3, tmp_path)
    reload.load_lora_ckpt(1, 0, lora_dir=str(tmp_path / "lora"))
    assert torch.equal(_weight(reload), expected)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("postfix, name", [(None, "checkpoint_0"), ("run1", "checkpoint_run1_0")])
def test_ckpt_dir_name(tmp_path, monkeypatch, postfix, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opti_pri, "Comm", lambda **kw: SimpleNamespace(rank=0, world_size=1, local_rank=0))

    def stop(*a):                                       # __init__ names the directory just before picking the GPU
        raise _Stop
    monkeypatch.setattr(torch.cuda, "set_device", stop)
    with pytest.raises(_Stop):
        Optimus_p(nn.Linear(2, 2), 1, checkpoint=True, ckpt_dir_postfix=postfix)
    assert os.listdir(tmp_path) == [name]
