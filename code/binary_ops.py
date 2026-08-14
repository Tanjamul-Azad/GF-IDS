"""
Binarization primitives for GF-IDS.

Implements the three mechanisms the method section relies on:

  SignSTE            sign() in the forward pass, straight-through
                     estimator with a hardtanh-shaped gradient window
                     in the backward pass
  BinaryLinear       dense layer whose effective weights are +/-1
  BinaryActivation   binarizes the activation tensor between layers

Straight-Through Estimator
--------------------------
sign() is piecewise constant, so its true derivative is zero almost
everywhere and gradients cannot reach the underlying real-valued
weights. The STE substitutes a surrogate gradient

    d sign(x) / dx  ~=  1  if |x| <= 1
                        0  otherwise

so gradients pass through unchanged inside [-1, 1] and are blocked
outside it. The real-valued weights are what the optimizer updates;
the binary weights are re-derived from them on every forward pass.
This is the dual-weight mechanism that makes BNN training possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """sign() forward, clipped straight-through gradient backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        s = torch.sign(x)
        # sign(0) = 0 would leave dead weights; map them to +1.
        return torch.where(s == 0, torch.ones_like(s), s)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Gradient window: pass through where |x| <= 1, block outside.
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)


def binarize(x):
    return SignSTE.apply(x)


class BinaryLinear(nn.Linear):
    """Dense layer with weights constrained to {-1, +1}.

    The module keeps a real-valued `weight` tensor, which is what the
    optimizer updates and what gets aggregated by the federated server.
    The binary weights used in the forward pass are derived from it.
    """

    def forward(self, x):
        w_b = binarize(self.weight)
        return F.linear(x, w_b, self.bias)

    @torch.no_grad()
    def clip_weights(self):
        """Clamp latent weights to [-1, 1].

        Without this the real-valued weights drift outside the STE
        gradient window, the surrogate gradient becomes zero, and the
        layer silently stops learning.
        """
        self.weight.clamp_(-1, 1)


class BinaryActivation(nn.Module):
    """Binarizes activations to {-1, +1} with the same STE.

    With both weights and activations binary, the layer's matrix
    product reduces to XNOR followed by popcount on hardware that
    supports it.
    """

    def forward(self, x):
        return binarize(x)


def clip_all_binary_weights(model):
    """Call after each optimizer step to keep latent weights in range."""
    for module in model.modules():
        if isinstance(module, BinaryLinear):
            module.clip_weights()


def binary_weight_keys(model, prefix=""):
    """State-dict keys of the 2-D BinaryLinear weight matrices.

    Used both for post-aggregation re-binarization and for payload
    accounting. Deliberately excludes biases and BatchNorm affine
    parameters, which stay in full precision.
    """
    keys = []
    for name, module in model.named_modules():
        if isinstance(module, BinaryLinear):
            keys.append(f"{prefix}{name}.weight" if prefix
                        else f"{name}.weight")
    return set(keys)
