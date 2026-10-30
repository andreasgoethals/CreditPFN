"""Non-reentrant activation recomputation without changing the objective."""
from __future__ import annotations

from torch.utils.checkpoint import checkpoint


def forward(model, *args, enabled: bool = False, **kwargs):
    """Recompute one native member forward, preserving its autograd graph.

    The pinned TabPFN v2 layer option uses reentrant checkpointing; that can
    change BF16 gradient aggregation. PyTorch's non-reentrant implementation
    preserves the original graph and RNG/autocast context. It discards saved
    activations, not weights, rows, views or precision. Native-model GPU checks
    compare outputs, gradients and updates and observe actual recomputation.
    """
    if not isinstance(enabled, bool):
        raise ValueError("train.activation_checkpointing must be a boolean")
    if not enabled:
        return model(*args, **kwargs)
    return checkpoint(model, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
