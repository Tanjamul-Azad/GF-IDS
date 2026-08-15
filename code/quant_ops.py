"""
Eight bit quantization primitives for the INT8 baseline.

The paper argues that constraining hidden layer weights to one bit
lowers the cost of federated training. The obvious question a reader
will ask is why one bit rather than eight, since int8 quantization is
mature, widely deployed, and loses almost no accuracy. This module
provides that comparison so the question can be answered with numbers
instead of assertion.

The design deliberately mirrors binary_ops.py so that the two differ
only in bit width and nothing else:

  binary_ops.BinaryLinear   weights forced onto {-1, +1},   1 bit each
  quant_ops.QuantLinear     weights forced onto a 256 level grid,
                            8 bits each plus one float32 scale
                            per tensor

Both keep a real valued latent weight that the optimizer updates,
both round it in the forward pass, and both pass the gradient through
that rounding with a straight through estimator. Any difference in
the results therefore comes from the bit width rather than from a
difference in how the two were trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quant_scale(w, bits=8):
    """Symmetric per-tensor scale factor.

    The largest magnitude in the tensor is mapped to the largest
    representable level, so no weight is clipped.
    """
    qmax = 2 ** (bits - 1) - 1
    return w.detach().abs().max().clamp(min=1e-8) / qmax


def fake_quant(w, bits=8):
    """Round to the int8 grid in the forward pass, pass gradient through.

    Returns a tensor whose values sit exactly on the quantization grid
    but whose gradient flows to `w` unchanged. This is the same
    straight through trick used for binarization, with a finer grid.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = quant_scale(w, bits)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (q - w).detach()


class QuantLinear(nn.Linear):
    """Dense layer whose weights are constrained to an 8 bit grid."""

    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits

    def forward(self, x):
        return F.linear(x, fake_quant(self.weight, self.bits), self.bias)

    @torch.no_grad()
    def requantize_(self):
        """Snap the stored weights onto the grid.

        Used after federated averaging, so that what is distributed to
        the clients is a genuinely quantized model rather than a real
        valued one. This is the int8 counterpart of re-binarization.
        """
        self.weight.copy_(fake_quant(self.weight, self.bits))


def quant_weight_keys(model, bits=8):
    """State-dict keys of QuantLinear weight matrices, with bit width.

    Returned as a mapping so payload accounting can price each tensor
    at its own precision.
    """
    return {f"{name}.weight": getattr(module, "bits", bits)
            for name, module in model.named_modules()
            if isinstance(module, QuantLinear)}


def requantize_all(model):
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.requantize_()
