"""
Score-based binary subnetwork primitives -- a reproduction of the core
mechanism in BiPruneFL (S. Lee, H. Jang, "BiPruneFL: Computation and
Communication Efficient Federated Learning With Binary Quantization
and Pruning," IEEE Access, vol. 13, pp. 42441-42456, 2025, DOI
10.1109/ACCESS.2025.3547627), which itself builds on Biprop / edge-popup
(V. Ramanujan et al., "What's Hidden in a Randomly Weighted Neural
Network?", CVPR 2020; J. Diffenderfer, B. Kailkhura, "Multi-Prize
Lottery Ticket Hypothesis," ICLR 2021).

This is a REPRODUCTION, not the original authors' code -- registered
in MODEL_REGISTRY as "BiPruneFL-Repro" everywhere in this project's
outputs (tables, figures, logs), so every result is explicit about
that. Two things in the original paper could not be confirmed against
the full text (IEEE Xplore paywalled; only the abstract and secondary
summaries were reachable) and are handled here with a documented,
standard substitute rather than a silent guess:

  1. The exact per-layer amplitude-gain formula. This uses the
     XNOR-Net convention (Rastegari et al., ECCV 2016): the mean
     absolute value of the frozen weight among the currently unmasked
     (top-k) positions, recomputed every forward pass as the mask
     changes with training.
  2. The exact federated aggregation / server role. Available summaries
     describe the server performing "SGD updates, mask computation,
     and amplitude gain calculation" -- possibly a different protocol
     from plain FedAvg. This reproduction runs under the SAME FedAvg
     protocol as every other model in this project (same T, seed,
     client partition), which is the only way to keep the comparison
     controlled -- the same choice Liu et al. 2023 (IoT-J) made for
     their own "derived from [cited paper]" baselines when the
     original protocol wasn't reproducible verbatim.

The core mechanism IS reproduced faithfully:
  - Each layer's weight is drawn once at construction and then FROZEN
    -- never updated by the optimizer, registered as a buffer rather
    than a Parameter. Only a per-weight importance SCORE is trained.
  - The forward pass keeps only the top-k% of weights by |score| (a
    hard, non-differentiable selection) and zeroes the rest. This is
    GetSubnet, using the same straight-through-estimator idea already
    used elsewhere in this codebase (see binary_ops.SignSTE): forward
    is a hard threshold, backward passes the incoming gradient straight
    through to every score unchanged, since top-k selection has no
    real gradient of its own.
  - The surviving weights are binarized to the frozen weight's sign
    and scaled by the layer's amplitude gain.

Because the underlying weight values never change round to round, they
do not need to cross the network after the first round in principle;
evaluate.py's payload accounting treats them that way explicitly (see
edgepop_frozen_keys()). The trainable score tensor is priced the same
way this project already prices any other real-valued FedAvg
parameter -- this mirrors the same "uplink is not what you'd expect"
honesty already established for BNN-MATCHED's own latent weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GetSubnet(torch.autograd.Function):
    """Hard top-k mask by |score|, forward pass; straight-through
    gradient, backward pass. Reference algorithm from Ramanujan et al.
    2020 ("edge-popup"), reproduced here since the original is not
    published as an importable library.
    """

    @staticmethod
    def forward(ctx, scores, keep_fraction):
        out = scores.clone()
        flat = out.flatten()
        _, idx = scores.flatten().abs().sort()
        j = int((1 - keep_fraction) * scores.numel())
        flat[idx[:j]] = 0
        flat[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: every score receives the incoming gradient
        # unchanged, since top-k selection has no true derivative.
        return grad_output, None


class EdgePopupLinear(nn.Linear):
    """Dense layer whose weight is frozen at a random initialization;
    only a per-weight importance score is trained. keep_fraction sets
    the target sparsity (e.g. 0.5 keeps the top 50% of weights by
    |score|, prunes the rest to zero).
    """

    def __init__(self, in_features, out_features, bias=True,
                keep_fraction=0.5):
        super().__init__(in_features, out_features, bias=bias)
        self.keep_fraction = keep_fraction

        # Freeze the weight at nn.Linear's own random initialization.
        # Kept as real-valued (not yet sign()'d) so the magnitude is
        # still available for the amplitude-gain calculation below --
        # sign() would discard it and silently make the gain always 1.
        frozen = self.weight.data.clone()
        del self.weight
        self.register_buffer("frozen_weight", frozen)

        self.score = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.score, a=5 ** 0.5)

    def forward(self, x):
        mask = GetSubnet.apply(self.score, self.keep_fraction)
        sign = torch.sign(self.frozen_weight)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        with torch.no_grad():
            kept_mag = self.frozen_weight.abs()[mask.bool()]
            gain = (kept_mag.mean() if kept_mag.numel() > 0
                   else torch.tensor(1.0, device=self.frozen_weight.device))
        w_eff = sign * mask * gain
        return F.linear(x, w_eff, self.bias)

    @torch.no_grad()
    def achieved_sparsity(self):
        mask = GetSubnet.apply(self.score, self.keep_fraction)
        return 1.0 - mask.mean().item()


def edgepop_score_keys(model):
    """State-dict keys of the trainable score tensors -- the only
    thing FedAvg actually needs to average every round for this model.
    """
    return {f"{name}.score" for name, module in model.named_modules()
            if isinstance(module, EdgePopupLinear)}


def edgepop_frozen_keys(model):
    """State-dict keys of the frozen weight buffers -- fixed at
    construction, bit-identical across every client and every round
    (load_state_dict propagates the global model's value to every
    freshly-constructed local model before training starts, and the
    value itself is never touched by any optimizer), so payload
    accounting treats them as free to transmit after the first round.
    """
    return {f"{name}.frozen_weight" for name, module in model.named_modules()
            if isinstance(module, EdgePopupLinear)}


def count_edgepop_linear(module, x, y):
    """thop handler for EdgePopupLinear.

    Same blind spot as BinaryLinear/QuantLinear: subclasses nn.Linear,
    thop dispatches on exact type, so without this handler it would
    silently contribute zero operations.

    Reports the sparsity-scaled operation count (in_features *
    out_features * keep_fraction) as the "useful" binary op count.
    Note this is the same kind of theoretical count already flagged
    elsewhere in this project as not automatically reflected in
    measured wall-clock latency: a plain dense F.linear call still
    computes the zeroed positions too, since PyTorch has no
    sparse-aware kernel here. The Limitations section already makes
    this same point for BNN's bit-packed operations.
    """
    module.total_ops += torch.DoubleTensor(
        [module.in_features * module.out_features * module.keep_fraction])
