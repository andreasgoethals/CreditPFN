"""Resolve the continued-pretraining precision policy for the loaded backbone."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch


def _float_input(module, args):
    return (args[0].float(),)


@contextmanager
def _fp32_normalizations(layers):
    with ExitStack() as stack:
        for layer in layers:
            stack.enter_context(layer.register_forward_pre_hook(_float_input))
        yield


@dataclass(frozen=True)
class TrainingPrecision:
    label: str
    normalizations: tuple = ()

    def scope(self):
        """Cover the training forward and any backward checkpoint recomputations."""
        return _fp32_normalizations(self.normalizations) if self.normalizations else nullcontext()


def training_precision(model, *, base, family, track, amp_dtype) -> TrainingPrecision:
    """Keep v2 PD normalization and residuals in FP32 while projections use FP16."""
    if amp_dtype is None:
        return TrainingPrecision("fp32")
    if amp_dtype == torch.bfloat16:
        return TrainingPrecision("bf16")
    if amp_dtype != torch.float16:
        raise ValueError(f"Unexpected training autocast dtype {amp_dtype}")
    if family != "tabpfn" or track != "pd":
        return TrainingPrecision("fp16")

    from tabpfn.architectures.tabpfn_v2 import LowerPrecisionLayerNorm, TabPFNV2

    from src.train.model import _infer_version

    # A loaded model can be wrapped for adaptation. Check its actual architecture
    # against the checkpoint version before opting into (or out of) the v2 policy.
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Missing loaded TabPFN backbone for the FP16 training policy")
    backbones = [module for module in model.modules() if isinstance(module, TabPFNV2)]
    expect_v2 = _infer_version(Path(base)) == "v2"
    if len(backbones) != int(expect_v2):
        raise ValueError(f"The loaded architecture differs from the TabPFN version in {Path(base).name}")
    if not expect_v2:
        return TrainingPrecision("fp16")

    backbone = backbones[0]
    layers = tuple(layer for layer in backbone.modules() if isinstance(layer, LowerPrecisionLayerNorm))
    expected = {getattr(block, name, None) for block in backbone.blocks
                for name in ("layernorm_mha1", "layernorm_mha2", "layernorm_mlp")}
    if not layers or len(layers) != 3 * len(backbone.blocks) or set(layers) != expected:
        raise ValueError("The TabPFN v2 normalization layout differs from the validated architecture")
    return TrainingPrecision("fp16_norm32", layers)
