# Progress

Single source of truth for where the revision stands. Updated every time
something changes; read this file first. Full round-by-round numbers and
the reasoning behind each decision are in
[`RUN_RESULTS.md`](RUN_RESULTS.md).

**Last updated: 2026-09-16.**

## Status

All 8 model variants are complete, run locally on an RTX 4060 (not
Colab — an earlier hardware/library difference moved accuracy by up
to ~6 points, see Limitations in the manuscript), all at `T=45`
rounds, `seed=42`. The six-model headline comparison plus two
ablations (`BNN-FULL`, fully binarized; the original unmatched `BNN`)
are all written into the manuscript, including a new ablation section,
a rewritten int8 comparison, and honest downlink/uplink/round-trip
communication accounting throughout. The manuscript currently compiles
cleanly (0 errors, 0 warnings, 16 pages, IEEEtran/IoT-J template).

## The headline result

`BNN-MATCHED` — hidden layers sized to hold the same number of weights
as MLP-FL, removing an earlier, undocumented extra 128x128 layer that
had doubled the original BNN's weight count — is now the model called
"BNN-FL (Proposed)" throughout the manuscript.

| Model | Best accuracy | Round | MCC | FPR | Params |
|---|---:|---:|---:|---:|---:|
| **BNN-FL (proposed / BNN-MATCHED)** | **97.75%** | 38 | **0.9754** | **0.07%** | **15,746** |
| LSTM-FL | 96.22% | 11 | 0.9588 | 0.12% | 53,634 |
| MLP-FL | 92.08% | 28 | 0.9143 | 0.25% | 15,938 |
| CNN-FL | 89.18% | 30 | 0.8830 | 0.35% | 84,962 |
| BNN-INT8IO | 85.08% | 42 | 0.8372 | 0.48% | 32,514 |
| MLP-INT8 | 84.59% | 32 | 0.8326 | 0.49% | 15,938 |

BNN-MATCHED beats the original (unmatched) BNN on every metric while
using half the parameters — the extra capacity in the original design
was not just uncontrolled, it was actively unhelpful.

## Ablation: why BNN-MATCHED, not the original BNN

| Variant | Params | Accuracy | MCC | FPR |
|---|---:|---:|---:|---:|
| BNN-FL (proposed / matched) | 15,746 | 97.75% | 0.9754 | 0.07% |
| BNN-FL (extra layer, original) | 32,514 | 97.30% | 0.9704 | 0.08% |
| BNN-FL (fully binarized, input+output too) | 32,514 | 71.08% | 0.6959 | 0.93% |

Fully binarizing the input/output layers collapses accuracy and holds
flat at ~71% for the last 21 of 45 rounds — a measured confirmation,
not just a citation, that keeping those two layers in Float32 is load
bearing.

## Communication: downlink and uplink are not the same number

A real gap was found and fixed this pass: `PackedPayload(KB)` is the
**downlink** (the re-binarized global model the server sends out).
Clients upload real-valued latent weights every round regardless of
hidden-layer precision, because FedAvg must average in full precision
— averaging ±1 values directly produces meaningless fractions. So the
**uplink** is the model's full Float32 size, and for the original BNN
that made its round trip *larger* than MLP-FL's, not smaller.

| Model | Downlink (KB) | Uplink (KB) | Round trip (KB) |
|---|---:|---:|---:|
| BNN-FL (proposed) | 23.52 | 62.27 | **85.79** |
| BNN-INT8IO | 13.23 | 128.78 | 142.01 |
| MLP-FL | 63.77 | 63.77 | 127.54 |
| MLP-INT8 | 18.98 | 63.77 | 82.75 |
| CNN-FL | 333.40 | 333.40 | 666.80 |
| LSTM-FL | 209.51 | 209.51 | 419.02 |

BNN-MATCHED's uplink is close to MLP-FL's because the two models now
hold nearly the same number of weights — the fix for the capacity
confound also fixed the uplink accounting, without any transport
trick. `BNN-INT8IO` still carries the *old* unmatched architecture
(never retrained), so its round trip is worse than MLP-INT8's despite
a smaller downlink — flagged honestly in the manuscript as an open
gap, not smoothed over.

## Cost to reach a working model (round-trip, K=5 clients)

| Model | To 85% | To 90% |
|---|---:|---:|
| BNN-FL (proposed) | round 11, 4.61 MB | round 13, **5.45 MB** |
| MLP-FL | round 7, **4.36 MB** | round 28, 17.44 MB |
| LSTM-FL | round 5, 10.23 MB | round 5, 10.23 MB |
| CNN-FL | round 14, 45.58 MB | not reached |

Honest nuance kept in the manuscript: MLP-FL is *cheaper* at the 85%
threshold (fewer rounds outweighs its higher per-round cost), but
BNN-FL is far cheaper at 90%. The paper does not claim BNN-FL wins on
every threshold — it claims BNN-FL wins on reaching a *high-accuracy*
model, which is the more demanding and more relevant claim.

## What changed in the code, and why

Full detail in the repository's commit history on `fix/ste-rebinarization`.
Summary of everything since the last update, in order:

- **Straight-through estimator, re-binarization, payload measurement,
  FLOPs/BOPs, int8 baselines, reproducibility** — see prior commits,
  unchanged since 2026-09-09.
- **Checkpoint bug**: `evaluate.py`/`figures.py` looked for
  `{model}_best.pt` but `federated_train.py` names checkpoints
  `{model}_seed{seed}_best.pt` via `run_tag()` — every model was
  silently skipped until caught. Fixed, both scripts now take `--seed`.
- **QuantLinear thop blind spot**: same issue `BinaryLinear` originally
  had — subclasses `nn.Linear`, thop skipped it, so `MLP-INT8` showed
  near-zero FLOPs. Fixed; int8 ops now reported as a separate
  `IOPs(M)` column.
- **`BNNFullModel` and `BNNMatchedModel` added** to `models.py` — see
  above.
- **`figures.py` rewritten**: `ORDER` now lists `BNN-MATCHED` as
  primary; new `fig_ablation()` compares the three BNN variants; fixed
  two rename-era bugs (`fig_pareto`'s `OFFSET`/marker-size checks were
  still keyed on the old `"BNN"` string) and every "Uplink payload"
  axis label, which was actually plotting downlink data.

## Open questions / next steps

1. Train a matched-capacity `BNN-INT8IO` so the int8 comparison in the
   manuscript has the same fair footing the main comparison now has —
   flagged as future work in Limitations, not yet done.
2. Multiple seeds for confidence intervals — a single-seed
   Colab-vs-local hardware check already showed CNN-FL moving ~6
   points and MLP-FL ~4 points with everything else held fixed, so the
   baseline ordering (LSTM-FL vs MLP-FL vs CNN-FL specifically) should
   not be treated as settled.
3. Non-IID client partitioning, physical IoT hardware validation, and
   a real hardware energy measurement (Raspberry Pi 4 available) all
   remain on the IoT-J readiness plan — see
   `JOURNAL_TARGET_PLAN.md` (local only, not in this repo).
4. Fig. 1's designed graphic (`final dig.pdf`) still says
   "Rounds (T): e.g., 20" — cosmetic, needs fixing by hand in whatever
   tool made it; not fixable from LaTeX since no source file is
   available in this environment.
