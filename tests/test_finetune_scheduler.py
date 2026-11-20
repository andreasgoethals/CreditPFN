import copy
import warnings

import pytest
import torch

from src.eval.finetune_scheduler import UpdateAlignedLambdaLR


def test_learning_rate_tracks_updates_across_overflow_and_restart():
    param = torch.nn.Parameter(torch.ones(1))
    def build():
        optimizer = torch.optim.AdamW([param], lr=0.2)
        scheduler = UpdateAlignedLambdaLR(optimizer, lr_lambda=lambda i: 1 / (1 + i))
        return optimizer, scheduler

    opt, scheduler = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scheduler.step()  # overflow before first step
        assert scheduler.last_epoch == 0
        param.grad = torch.ones_like(param)
        opt.step()
        scheduler.step()
        scheduler.step()  # overflow after the first real update
        assert scheduler.get_last_lr() == [0.1]
        opt_state, lr_state = copy.deepcopy(opt.state_dict()), copy.deepcopy(scheduler.state_dict())
        assert "_observed_updates" not in lr_state
        opt, scheduler = build()
        opt.load_state_dict(opt_state)
        scheduler.load_state_dict(lr_state)
        scheduler.step()  # first attempt after resumption also overflows
        assert scheduler.last_epoch == 1
        opt.step()
        scheduler.step()
        assert scheduler.last_epoch == 2
        assert scheduler.get_last_lr()[0] == pytest.approx(0.2 / 3)
