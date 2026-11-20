"""Keep TabPFN's warmup/decay schedule aligned with completed optimizer updates."""
from torch.optim.lr_scheduler import LambdaLR


def _count_update(optimizer, args, kwargs):
    optimizer._creditpfn_update_count += 1


class UpdateAlignedLambdaLR(LambdaLR):
    """Use the upstream schedule, consuming a step only when AdamW actually ran.

    The pinned fine-tuner calls the scheduler even when GradScaler detects an overflow
    and skips its AdamW call. PyTorch's post-step hook observes successful calls.
    """

    def __init__(self, optimizer, *args, **kwargs):
        if not hasattr(optimizer, "_creditpfn_update_count"):
            optimizer._creditpfn_update_count = 0
            optimizer.register_step_post_hook(_count_update)
        self._observed_updates = None
        super().__init__(optimizer, *args, **kwargs)

    def step(self, epoch=None):
        updates = self.optimizer._creditpfn_update_count
        if updates != self._observed_updates:
            self._observed_updates = updates
            super().step(epoch)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # The fresh optimizer has a fresh hook counter on process restart. The
        # upstream scheduler state already contains the restored position.
        self._observed_updates = self.optimizer._creditpfn_update_count

    def state_dict(self):
        state = super().state_dict()
        state.pop("_observed_updates", None)
        return state
