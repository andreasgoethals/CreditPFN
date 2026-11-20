"""Numerical controls for TabPFN v2 in Experiment 0."""
from contextlib import ExitStack, contextmanager
from functools import partial
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint


def _column_chunks(forward, x, single_eval_pos=None, *, cached_kv=None, return_kv=False):
    """Each column attends to its own full set of rows; limit concurrent columns."""
    if cached_kv is not None or return_kv:
        raise ValueError("The FP32 gradient reference requires the ordinary training forward")
    def attend(columns):
        return forward(columns, single_eval_pos=single_eval_pos)[0]
    outputs = [checkpoint(attend, columns, use_reentrant=False) for columns in x.split(2)]
    return torch.cat(outputs, dim=0), None


@contextmanager
def bounded_fp32_reference(model):
    """Recompute each block and its independent column-attention groups on backward.

    The previous whole-model checkpoint still needed a full member's backward graph.
    Both ensemble views and every context/query row are retained here. Neither module
    replacement registers new parameters or changes a checkpoint.
    """
    from tabpfn.architectures.tabpfn_v2 import TabPFNV2
    if not isinstance(model, TabPFNV2):
        raise ValueError("The blockwise reference is specific to TabPFN v2")
    with ExitStack() as stack:
        for block in model.blocks:
            attention = block.per_column_attention_between_cells
            stack.enter_context(patch.object(attention, "forward", partial(_column_chunks, attention.forward)))
            stack.enter_context(patch.object(block, "forward", partial(checkpoint, block.forward, use_reentrant=False)))
        yield
