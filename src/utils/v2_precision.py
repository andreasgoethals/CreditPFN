"""Historical mixed-precision comparator for the TabPFN v2 PD pilot."""
from contextlib import ExitStack, contextmanager


def _float_input(module, args):
    return (args[0].float(),)


@contextmanager
def _fp32_normalizations(layers):
    with ExitStack() as stack:
        for layer in layers:
            stack.enter_context(layer.register_forward_pre_hook(_float_input))
        yield


def fp32_normalizations(model):
    """Match the measured E0 FP16 candidate, which casts each v2 norm input."""
    from tabpfn.architectures.tabpfn_v2 import LowerPrecisionLayerNorm
    layers = [layer for layer in model.modules() if isinstance(layer, LowerPrecisionLayerNorm)]
    if not layers:
        raise ValueError("No v2 normalization layers found for the historical precision comparison")
    return _fp32_normalizations(layers)
