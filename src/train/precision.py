"""Numerical policy used by continued pretraining and the identical Experiment 0 step."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def strict_fp32():
    """FP32 matmuls without TF32 for the duration of a monitor, as in the five-fold benchmark
    (a separate process that never enables TF32). Training keeps TF32 for speed."""
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


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
        stack.enter_context(strict_fp32())
        stack.enter_context(torch.amp.autocast(next(model.parameters()).device.type, enabled=False))
        for block in model.blocks:
            attention = block.per_column_attention_between_cells
            stack.enter_context(patch.object(attention, "forward", partial(_column_chunks, attention.forward)))
            stack.enter_context(patch.object(block, "forward", partial(checkpoint, block.forward, use_reentrant=False)))
        yield


@dataclass(frozen=True)
class TrainingPrecision:
    label: str
    amp_dtype: torch.dtype | None
    bounded_backbone: torch.nn.Module | None = None

    def scope(self):
        """Cover forward and backward, including checkpoint recomputation."""
        return bounded_fp32_reference(self.bounded_backbone) if self.bounded_backbone is not None else nullcontext()

    def scaler(self, device_type: str):
        return torch.amp.GradScaler(device_type, enabled=self.amp_dtype == torch.float16)


def training_precision(model, *, base, family, track, amp_dtype) -> TrainingPrecision:
    """Resolve the effective AMP mode and the bounded strict-FP32 exception together."""
    labels = {None: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    if amp_dtype not in labels:
        raise ValueError(f"Unexpected training autocast dtype {amp_dtype}")
    native = TrainingPrecision(labels[amp_dtype], amp_dtype)
    if family != "tabpfn" or track != "pd":
        return native

    from tabpfn.architectures.tabpfn_v2 import TabPFNV2

    from src.train.model import _infer_version

    if not isinstance(model, torch.nn.Module):
        raise ValueError("Missing loaded TabPFN backbone for the training precision policy")
    # The training loader infers the architecture from this same version. PEFT may
    # wrap its backbone; apply block checkpointing to the unique v2 instance inside.
    expect_v2 = _infer_version(Path(base)) == "v2"
    backbones = [module for module in model.modules() if isinstance(module, TabPFNV2)]
    if len(backbones) != int(expect_v2):
        raise ValueError(f"The loaded architecture differs from the TabPFN version in {Path(base).name}")
    if not expect_v2:
        return native
    backbone = backbones[0]
    if not len(backbone.blocks) or any(not callable(getattr(block, "per_column_attention_between_cells", None))
                                      for block in backbone.blocks):
        raise ValueError("The loaded v2 model lacks the validated column-attention layout")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("The bounded FP32 training policy requires FP32 model parameters")
    return TrainingPrecision("fp32_bounded", None, backbone)
