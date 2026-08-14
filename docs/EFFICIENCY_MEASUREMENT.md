# Efficiency Measurement Notes

How the communication and computation numbers are produced, and what
`code/verify_fixes.py` confirms about them.

These quantities depend only on the architecture, not on the data, so
they are exact and reproducible without running the full federated
training. Detection metrics (accuracy, F1, MCC) do require the trained
models.

Reproduce with:

```bash
python code/verify_fixes.py
```

---

## Communication payload

Three figures, all measured on the full `state_dict`, since FedAvg
transports the entire state dict each round.

| Model | Encoding | Payload per round |
|---|---|---|
| BNN | binary layers bit-packed | **28.03 KB** |
| BNN | uncompressed float32 | 128.78 KB |
| MLP | uncompressed float32 | 63.77 KB |

- Bit-packing the three `BinaryLinear` weight matrices shrinks the BNN
  payload by **78.2%** relative to sending it uncompressed.
- Packed BNN against the float32 MLP baseline: **56.0% smaller**.
- Without packing, the BNN is **101.9% larger** than the MLP, because
  it has roughly twice the parameters (32,514 vs 15,938). The saving
  comes entirely from the encoding, not from the architecture, so the
  packing step is what has to be implemented and measured for the
  communication claim to hold.

`packed_payload_bits()` performs the packing with `numpy.packbits`
and reports the resulting byte count. `payload_bits()` computes the
same quantity analytically; the two agree exactly.

### BatchNorm running statistics

Payload accounting iterates `state_dict()` rather than
`named_parameters()`. BatchNorm `running_mean` and `running_var` are
buffers, not parameters, but FedAvg still transmits them every round.
Counting only `named_parameters()` omits 1.75 KB per round for the BNN
and 1.5 KB for the MLP. Every model here uses BatchNorm, so the
omission is systematic, but the absolute payloads are understated
unless buffers are included.

## Operation counts

Reported as two separate quantities, because binary layers execute as
XNOR plus popcount rather than multiply-accumulate. Folding them into
one number would misstate both the cost and the saving.

| Model | Float ops (MACs) | Binary ops (BOPs) | Total |
|---|---|---|---|
| BNN | **5,952** | 26,624 | 32,576 |
| MLP | 16,064 | 0 | 16,064 |

The defensible statement is that **BNN-FL performs 63% fewer
floating-point operations than MLP-FL** (5,952 vs 16,064), moving the
bulk of its arithmetic into 26,624 bitwise operations that are far
cheaper per operation on hardware with native support.

It is not correct to say the BNN performs fewer operations overall —
it performs about twice as many (32,576 vs 16,064). The advantage is
in the *kind* of operation, not the count.

### thop and subclassed modules

`thop` dispatches on exact module type. `BinaryLinear` subclasses
`nn.Linear`, so the built-in rule does not match it and the binary
layers contribute **zero** unless a handler is registered. Verified:

- without a handler: 5,952 ops (float layers only)
- with `custom_ops={BinaryLinear: count_binary_linear}`: 32,576 ops
- difference: 26,624, exactly the binary layer weight count
  (128×128 + 128×64 + 64×32)

`evaluate.py` registers the handler. Any FLOP figure produced without
it silently excludes the binarized layers.

## Energy proxy

Derived from the operation count using a nominal 0.5 pJ per operation.
Unit reminder: 1 nJ = 1000 pJ.

Because the proxy is a linear function of the operation count, it
inherits whatever that count includes. Applying a single per-operation
coefficient to a model that is mostly bitwise while comparing it
against a model that is entirely floating-point overstates the binary
model's cost, since a bitwise operation is considerably cheaper than a
float multiply-accumulate. Either apply separate coefficients to FLOPs
and BOPs, or state that the proxy is an upper bound for the BNN.

## Parameter counts

Verified against the architecture:

| Model | Weights | Biases | BatchNorm affine | Total |
|---|---|---|---|---|
| BNN | 31,680 | 386 | 448 | **32,514** |
| MLP | 15,296 | 258 | 384 | **15,938** |

The BNN carries roughly twice the parameters of the MLP because it has
an additional 128 → 128 binary layer between the Float32 input
projection and the 128 → 64 layer. The two models therefore do not
have identical hidden widths, which is worth stating explicitly
wherever the comparison is described as like-for-like.

## Binarization mechanisms

`verify_fixes.py` confirms each mechanism on synthetic data:

| Property | Result |
|---|---|
| Gradient reaches binary-layer latent weights | non-zero (L1 = 4.21) |
| Latent weights move after an optimizer step | mean \|Δ\| = 0.0100 |
| Latent weights stay within [-1, 1] | max \|w\| = 0.098 |
| Effective forward weights | exactly {-1, +1} |
| Hidden-layer activations | exactly {-1, +1} |
| FedAvg without re-binarization | 16,383 distinct values |
| FedAvg with re-binarization | exactly {-1, +1} |
| BatchNorm excluded from re-binarization | confirmed |

The straight-through estimator matters because `sign()` has zero
derivative almost everywhere. A plain `torch.sign(self.weight)`
formulation was measured to pass a gradient of exactly **0.0** to the
latent weights, so those layers would remain at their initialization
for the entire run. The STE substitutes a surrogate gradient inside
[-1, 1], and the post-step clamp keeps the latent weights inside that
window so the surrogate stays active.
