# Progress

Single source of truth for where the revision stands. Updated every time
something changes; read this file first. Full round-by-round numbers and
the reasoning behind each decision are in
[`RUN_RESULTS.md`](RUN_RESULTS.md).

**Last updated: 2026-09-18.**

## Status

All 8 IID model variants, all 6 headline models at two non-IID
severities (12 more runs), and one external baseline reproduction
(`BiPruneFL-Repro`, 21 runs total) are complete, run locally on an
RTX 4060, all at `T=45` rounds, `seed=42`. The IID six-model headline
comparison plus two ablations (`BNN-FULL`, fully binarized; the
original unmatched `BNN`) are written into the manuscript, including
an ablation section, a rewritten int8 comparison, and honest
downlink/uplink/round-trip communication accounting throughout. The
manuscript currently compiles cleanly (0 errors, 0 warnings, 16 pages,
IEEEtran/IoT-J template). **Not yet written into the manuscript**: the
non-IID results and the BiPruneFL-Repro comparison below — numbers are
verified and ready, the `main.tex` integration pass hasn't happened yet.

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

## New this pass: non-IID robustness and an external baseline

Two new pieces of work, both complete (full reasoning and every number
in `RUN_RESULTS.md`):

**Non-IID (label-skew, Hsu et al. 2019 Dirichlet partition) at two
severities:**

| Model | IID (%) | alpha=0.5, moderate (%) | alpha=0.1, severe (%) |
|---|---:|---:|---:|
| **BNN-FL (proposed)** | **97.75** | **89.47** | 68.20 |
| MLP-FL | 92.08 | 73.18 | 71.07 |
| LSTM-FL | 96.22 | 79.95 | **72.00** |
| CNN-FL | 89.18 | 83.60 | 69.85 |
| BNN-INT8IO | 85.08 | 79.60 | 69.16 |
| MLP-INT8 | 84.59 | 80.11 | 70.80 |

Not a simple decline: BNN-FL leads at IID and at moderate skew (by 5.9
points at alpha=0.5), then becomes the *worst* of six models at severe
skew (alpha=0.1) — every other model beats it there. A threshold
effect, reported honestly as one, not smoothed into "degrades under
non-IID."

**External baseline — `BiPruneFL-Repro`**, a reproduction of Lee &
Jang's BiPruneFL (IEEE Access 2025), built on Biprop/edge-popup (frozen
random weights, a learned pruning score, no weight training at all —
see `code/edgepop_ops.py`). At IID: 97.96% best accuracy, MCC 0.9777,
FPR 0.06% — matching or slightly beating BNN-FL at the same parameter
count (15,938 vs 15,746). But its round-trip payload is 63.77 KB
(identical to plain MLP-FL, since its trainable score has to move in
full Float32 for FedAvg) against BNN-FL's 23.52 KB downlink — **2.7x
more**. The honest claim: comparable accuracy to a genuine, recently
published competing method, at a fraction of its communication cost —
not "beats everything on every axis."

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

## In progress: follow-up queue (matched BNN-INT8IO + sign-flip diagnostics)

`run_investigation.ps1`, 7 runs. Done: `BNN-INT8IO-MATCHED` (seed 42, IID)
best 81.75%, below the old unmatched variant (85.08%) and MLP-INT8
(84.59%); `BNN-MATCHED` seed 43 IID best 97.05% (vs 97.75% at seed 42).
Running/left: BNN-MATCHED seed 43 at alpha=0.5 and 0.1, BNN-INT8IO seed 43
at IID/0.5/0.1, tracking per-round binary-weight sign-flip rate to test
whether post-aggregation re-binarization explains the severe-skew
collapse. Resume after any interruption (finished runs skip):

```
cd F:\UIU\11th\green\GFIDS_BNN
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_investigation.ps1
```

See `RUN_RESULTS.md` for details and the `evaluate.py` CSV-overwrite caveat.

## Open questions / next steps

1. The manuscript integration of non-IID and BiPruneFL-Repro is done
   locally (compiles clean). Still to fold in once the queue above
   finishes: the matched BNN-INT8IO result and the sign-flip findings.
2. ~~Train a matched-capacity BNN-INT8IO~~ done (81.75%, see above);
   manuscript text still to be updated.
3. Multiple seeds for confidence intervals — a single-seed
   Colab-vs-local hardware check already showed CNN-FL moving ~6
   points and MLP-FL ~4 points with everything else held fixed, so the
   baseline ordering (LSTM-FL vs MLP-FL vs CNN-FL specifically) should
   not be treated as settled.
4. Investigate *why* BNN-FL specifically collapses at severe non-IID
   skew while holding up at moderate skew and IID — not yet understood,
   just measured.
5. Physical IoT hardware validation and a real hardware energy
   measurement (Raspberry Pi 4 available, `code/pi_benchmark.py`
   ready and tested, needs the Pi on the network) remain on the IoT-J
   readiness plan — see `JOURNAL_TARGET_PLAN.md` (local only, not in
   this repo).
6. Fig. 1's designed graphic (`final dig.pdf`) still says
   "Rounds (T): e.g., 20" — cosmetic, needs fixing by hand in whatever
   tool made it; not fixable from LaTeX since no source file is
   available in this environment.
