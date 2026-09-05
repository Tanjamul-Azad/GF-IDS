# Progress

Single source of truth for where the revision stands. Updated every time
something changes; read this file first. Full round-by-round numbers and
the reasoning behind each decision are in
[`RUN_RESULTS.md`](RUN_RESULTS.md).

**Last updated: 2026-09-06.**

## Status

5 of 6 planned federated runs are complete, all at the same configuration
(`T=45` rounds, `seed=42`) so they are directly comparable. The sixth,
`BNN-INT8IO`, is in progress.

```
python federated_train.py --model BNN-INT8IO --seed 42 --rounds 45 --resume
```

## Why T=45 and seed 42

The original configuration used `T=20` rounds with no fixed seed. Once the
straight-through estimator was corrected (see "What changed" below), BNN
took 36 rounds to reach a stable plateau instead of the earlier apparent
peak at round 8, so `T=20` was cutting every architecture off before it
had converged, not just BNN. `T=45` was set from that diagnostic run and
is now applied identically to all six models. The client partition and
weight initialization are seeded so every run is reproducible and every
model trains on the same data split.

## Results so far (best accuracy, T=45, seed 42)

| Model | Best accuracy | Round | Final accuracy |
|---|---:|---:|---:|
| BNN | 97.68% | 44 | 97.60% |
| MLP | 95.86% | 23 | 63.53% |
| LSTM | 95.47% | 13 | 86.41% |
| CNN | 95.14% | 37 | 74.79% |
| MLP-INT8 | 87.85% | 29 | 71.87% |
| BNN-INT8IO | pending | — | — |

Final-round accuracy varies too much between runs to use as a headline
number (see `RUN_RESULTS.md` for why); best accuracy and a last-5-round
mean/std are the reported metrics going forward.

## Efficiency (architecture-derived, independent of training)

| Model | Payload / round | Float ops | Binary ops |
|---|---:|---:|---:|
| MLP (float32) | 63.77 KB | 16,064 | — |
| MLP-INT8 | 18.98 KB | 16,064 | — |
| BNN (current design) | 28.03 KB | 5,952 | 26,624 |
| BNN-INT8IO | 13.23 KB | 5,952 | 26,624 |

## What changed in the code, and why

Full detail in the repository's commit history on `fix/ste-rebinarization`
and in `docs/EFFICIENCY_MEASUREMENT.md`. Summary:

- **Straight-through estimator**: the binary layers previously received
  zero gradient and never trained past initialization. `binary_ops.py`
  now implements a proper STE so they learn.
- **Post-aggregation re-binarization**: the global model is now snapped
  back onto `{-1, +1}` after FedAvg, not left real-valued.
- **Payload measurement**: bytes per round are now measured by actually
  bit-packing the binary layers (`numpy.packbits`) rather than estimated
  from parameter names.
- **FLOPs**: a custom `thop` handler now counts the binary layers, which
  were previously invisible to the profiler; floating-point and binary
  operations are reported separately rather than combined.
- **INT8 baselines added**: `MLP-INT8` (every dense layer quantized) and
  `BNN-INT8IO` (only the two Float32 layers quantized, hidden layers stay
  binary) to test the design against 8-bit quantization, not only
  Float32.
- **Reproducibility**: client partitioning and weight initialization are
  now seeded; every round is checkpointed so an interrupted run resumes
  instead of restarting.

## Open questions / next steps

1. Finish `BNN-INT8IO`, then compare its accuracy against `MLP-INT8`'s
   87.85% best — see `RUN_RESULTS.md` for what each outcome would mean.
2. `MLP`'s 95.86% best (this run) vs. 97.78% (an earlier, unseeded run)
   is a large enough gap to warrant multiple seeds before treating either
   number as final.
3. Round-to-round accuracy oscillates significantly for every model
   after its peak (FedAvg + Adam interaction); worth a learning-rate
   schedule experiment.
4. Optional, larger-scope items under discussion: a SignSGD aggregation
   ablation, class-weighted loss for the low-support classes, and a
   second dataset.
