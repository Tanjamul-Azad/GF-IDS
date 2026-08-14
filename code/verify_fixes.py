"""
Self-contained verification of the GF-IDS binarization mechanisms.

Runs on synthetic data, needs no dataset and no GPU. Checks that each
mechanism the method section describes is actually doing what it
claims, and prints a PASS/FAIL line for each.

    python code/verify_fixes.py

Checks
  1. STE lets gradients reach the latent weights of binary layers
  2. Effective forward weights really are +/-1
  3. Activations between hidden layers really are +/-1
  4. Post-aggregation re-binarization yields a binary global model
  5. Parameter counts match the architecture
  6. Payload accounting is consistent and packing gives the expected saving
  7. thop counts the binary layers once a handler is registered
"""

import numpy as np
import torch
import torch.nn as nn

from binary_ops import (BinaryActivation, BinaryLinear, binary_weight_keys,
                        clip_all_binary_weights)
from models import BNNModel, MLPModel

INPUT_DIM = 31
NUM_CLASSES = 34
torch.manual_seed(0)
np.random.seed(0)

results = []


def check(name, passed, detail=""):
    results.append((name, passed))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")


print("=" * 66)
print("  GF-IDS binarization verification (synthetic data, CPU)")
print("=" * 66)

# ── 1. Gradients reach binary-layer latent weights ───────────
print("\n1. Straight-through estimator")
model = BNNModel(INPUT_DIM, NUM_CLASSES)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()

before = model.hidden1[0].weight.detach().clone()
X = torch.randn(64, INPUT_DIM)
y = torch.randint(0, NUM_CLASSES, (64,))

opt.zero_grad()
crit(model(X), y).backward()

grad = model.hidden1[0].weight.grad
grad_nonzero = grad is not None and grad.abs().sum().item() > 0
opt.step()
clip_all_binary_weights(model)
after = model.hidden1[0].weight.detach()
weights_changed = not torch.equal(before, after)

check("gradient reaches hidden1 latent weights", grad_nonzero,
      f"grad L1 norm = {grad.abs().sum().item():.4f}")
check("latent weights update after optimizer step", weights_changed,
      f"mean |delta| = {(after - before).abs().mean().item():.6f}")
check("latent weights stay clipped to [-1, 1]",
      bool(after.abs().max().item() <= 1.0 + 1e-6),
      f"max |w| = {after.abs().max().item():.4f}")

# Contrast: the plain sign() formulation without STE.
print("\n   Contrast - plain torch.sign() without STE:")


class NaiveBinaryLinear(nn.Linear):
    def forward(self, x):
        w_b = torch.sign(self.weight)
        w_b[w_b == 0] = 1
        return nn.functional.linear(x, w_b, self.bias)


naive = NaiveBinaryLinear(16, 8)
out = naive(torch.randn(4, 16)).sum()
out.backward()
naive_grad = naive.weight.grad.abs().sum().item()
check("plain sign() passes zero gradient (expected)", naive_grad == 0.0,
      f"grad L1 norm = {naive_grad}  -> latent weights would never move")

# ── 2. Effective weights are binary ──────────────────────────
print("\n2. Weight binarization")
with torch.no_grad():
    from binary_ops import binarize
    w_eff = binarize(model.hidden1[0].weight)
unique_w = torch.unique(w_eff).tolist()
check("effective forward weights are exactly {-1, +1}",
      set(unique_w) <= {-1.0, 1.0} and len(unique_w) > 0,
      f"unique values = {unique_w}")

# ── 3. Activations are binary ────────────────────────────────
print("\n3. Activation binarization")
model.eval()
captured = {}


def hook(_m, _i, out):
    captured["act"] = out.detach()


h = model.hidden1[2].register_forward_hook(hook)
with torch.no_grad():
    model(torch.randn(32, INPUT_DIM))
h.remove()
unique_a = torch.unique(captured["act"]).tolist()
check("hidden-layer activations are exactly {-1, +1}",
      set(unique_a) <= {-1.0, 1.0} and len(unique_a) > 0,
      f"unique values = {unique_a}")

# ── 4. Post-aggregation re-binarization ──────────────────────
print("\n4. FedAvg + re-binarization")
import sys
sys.path.insert(0, ".")
from federated_train import fedavg

clients = [BNNModel(INPUT_DIM, NUM_CLASSES) for _ in range(5)]
states = [c.state_dict() for c in clients]
sizes = [857184] * 5
rebin_keys = binary_weight_keys(clients[0])

plain = fedavg(states, sizes, rebinarize_keys=None)
rebin = fedavg(states, sizes, rebinarize_keys=rebin_keys)

key = "hidden1.0.weight"
plain_binary = set(torch.unique(plain[key]).tolist()) <= {-1.0, 1.0}
rebin_binary = set(torch.unique(rebin[key]).tolist()) <= {-1.0, 1.0}

check("plain FedAvg leaves weights real-valued (expected)",
      not plain_binary,
      f"{key}: {torch.unique(plain[key]).numel()} distinct values, "
      f"range [{plain[key].min():.4f}, {plain[key].max():.4f}]")
check("re-binarized FedAvg gives a binary global model", rebin_binary,
      f"{key}: unique = {torch.unique(rebin[key]).tolist()}")
bn_keys = [k for k in rebin if "hidden" in k and k.endswith(".weight")
           and rebin[k].dim() == 1]
check("BatchNorm weights excluded from re-binarization",
      all(k not in rebin_keys for k in bn_keys) and len(bn_keys) == 3,
      f"BatchNorm keys {bn_keys} are not in the re-binarization set "
      f"{sorted(rebin_keys)}")

# ── 5. Parameter counts ──────────────────────────────────────
print("\n5. Parameter counts")
bnn_params = sum(p.numel() for p in BNNModel(INPUT_DIM, NUM_CLASSES).parameters())
mlp_params = sum(p.numel() for p in MLPModel(INPUT_DIM, NUM_CLASSES).parameters())

expected_bnn = (31 * 128 + 128 * 128 + 128 * 64 + 64 * 32 + 32 * 34) \
    + (128 + 128 + 64 + 32 + 34) + 2 * (128 + 64 + 32)
expected_mlp = (31 * 128 + 128 * 64 + 64 * 32 + 32 * 34) \
    + (128 + 64 + 32 + 34) + 2 * (128 + 64)

check("BNN parameter count matches architecture",
      bnn_params == expected_bnn == 32514,
      f"counted {bnn_params:,}, derived {expected_bnn:,}")
check("MLP parameter count matches architecture",
      mlp_params == expected_mlp == 15938,
      f"counted {mlp_params:,}, derived {expected_mlp:,}")

# ── 6. Payload accounting ────────────────────────────────────
print("\n6. Payload accounting")
from evaluate import (float32_payload_bits, packed_payload_bits,
                      payload_bits)

bnn = BNNModel(INPUT_DIM, NUM_CLASSES)
mlp = MLPModel(INPUT_DIM, NUM_CLASSES)

bnn_packed = packed_payload_bits(bnn) / 8 / 1024
bnn_analytic = payload_bits(bnn) / 8 / 1024
bnn_float32 = float32_payload_bits(bnn) / 8 / 1024
mlp_float32 = float32_payload_bits(mlp) / 8 / 1024

check("packed and analytic payloads agree (within padding)",
      abs(bnn_packed - bnn_analytic) < 0.05,
      f"packed = {bnn_packed:.2f} KB, analytic = {bnn_analytic:.2f} KB")
check("packing actually shrinks the BNN payload",
      bnn_packed < bnn_float32,
      f"float32 = {bnn_float32:.2f} KB -> packed = {bnn_packed:.2f} KB "
      f"({(1 - bnn_packed / bnn_float32) * 100:.1f}% smaller)")
print(f"\n         BNN packed   : {bnn_packed:8.2f} KB")
print(f"         BNN float32  : {bnn_float32:8.2f} KB")
print(f"         MLP float32  : {mlp_float32:8.2f} KB")
print(f"         packed BNN vs float32 MLP: "
      f"{(1 - bnn_packed / mlp_float32) * 100:+.1f}%")
print(f"         float32 BNN vs float32 MLP: "
      f"{(1 - bnn_float32 / mlp_float32) * 100:+.1f}%")

# ── 7. thop sees the binary layers ───────────────────────────
print("\n7. Operation counting")
try:
    from thop import profile
    from evaluate import count_binary_linear

    dummy = torch.randn(1, INPUT_DIM)
    without, _ = profile(BNNModel(INPUT_DIM, NUM_CLASSES),
                         inputs=(dummy,), verbose=False)
    with_handler, _ = profile(
        BNNModel(INPUT_DIM, NUM_CLASSES), inputs=(dummy,), verbose=False,
        custom_ops={BinaryLinear: count_binary_linear})
    mlp_ops, _ = profile(MLPModel(INPUT_DIM, NUM_CLASSES),
                         inputs=(dummy,), verbose=False)

    binary_ops_count = 128 * 128 + 128 * 64 + 64 * 32
    check("thop skips BinaryLinear without a handler (expected)",
          without < with_handler,
          f"without handler = {without:,.0f} ops "
          f"(float layers only)")
    check("thop counts binary layers with the handler registered",
          abs((with_handler - without) - binary_ops_count) < 1,
          f"with handler = {with_handler:,.0f} ops, "
          f"difference = {with_handler - without:,.0f} "
          f"(expected {binary_ops_count:,})")
    print(f"\n         BNN float ops (MACs) : {without:12,.0f}")
    print(f"         BNN binary ops (BOPs): {binary_ops_count:12,.0f}")
    print(f"         BNN total ops        : {with_handler:12,.0f}")
    print(f"         MLP total ops (MACs) : {mlp_ops:12,.0f}")
except ImportError:
    print("  [SKIP] thop not installed")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 66)
passed = sum(1 for _, p in results if p)
print(f"  {passed}/{len(results)} checks passed")
print("=" * 66)
sys.exit(0 if passed == len(results) else 1)
