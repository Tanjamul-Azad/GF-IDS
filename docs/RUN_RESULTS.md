# Run Results Log

Every run against the fixed code (`fix/ste-rebinarization`), so the old
notebook numbers and the new ones never get confused with each other.

Config unless stated: CICIoT2023, 31 features, 34 classes, K=5 IID clients,
T=20 rounds, E=5 local epochs, batch 256, Adam, lr=0.0005, **seed 42**,
Colab T4.

---

## BNN, seed 42, T=20, unweighted loss

Run 2026-08-17. First run with a working straight-through estimator,
activation binarization, and post-aggregation re-binarization.

| Round | Old (broken STE) | New (working STE) |
|---:|---:|---:|
| 1  | 72.58 | 71.60 |
| 2  | 81.45 | 74.44 |
| 3  | 81.03 | 75.81 |
| 4  | 92.88 | 75.19 |
| 5  | 84.91 | 81.07 |
| 6  | 94.22 | 78.82 |
| 7  | 91.97 | 87.40 |
| 8  | **97.06** | 79.86 |
| 9  | 89.37 | 88.06 |
| 10 | 87.10 | 84.86 |
| 11 | 70.99 | 84.79 |
| 12 | 86.37 | 88.20 |
| 13 | 78.08 | 82.99 |
| 14 | 85.50 | 74.98 |
| 15 | 84.41 | 75.44 |
| 16 | 76.25 | 89.27 |
| 17 | 85.78 | 93.32 |
| 18 | 77.75 | 84.15 |
| 19 | 92.40 | 91.85 |
| 20 | 86.89 | **94.91** |

|  | Old | New |
|---|---:|---:|
| Peak accuracy | 97.06 (R8) | 94.91 (R20) |
| Final accuracy | 86.89 | 94.91 |
| Mean, first 5 rounds | 82.6 | 75.6 |
| Mean, last 5 rounds | 83.8 | 90.7 |

### What this means

**The STE fix worked, and the trajectory is the evidence.** The old model
started high and went nowhere: it reached 92.88 by round 4 and 97.06 by
round 8, then wandered between 70 and 92 for the remaining twelve rounds
with no trend. That is the signature of a network whose hidden layers are
frozen at their initialization, where only the Float32 input and output
layers plus BatchNorm are adapting. It fits quickly, then has nothing left
to learn.

The new model starts lower and climbs. Its first five rounds average 75.6
against the old 82.6, and its last five average 90.7 against the old 83.8.
That is what representation learning looks like: the binarized layers now
receive gradient and take time to organise, and the payoff arrives later.

**Three consequences for the paper:**

1. **Final accuracy is much better**, 94.91 against 86.89, a gain of 8.02
   points. The model a practitioner would actually deploy, the one at the
   end of training, is substantially stronger than before.

2. **Peak accuracy is lower**, 94.91 against 97.06. The old 97.06 was
   almost certainly a noise excursion rather than a converged state, since
   the model fell to 70.99 three rounds later and never returned near it.
   Reporting it as the headline was always fragile. The new peak coincides
   with the final round, so it is a real operating point.

3. **The model has not converged at T=20.** It is still improving at the
   last round. The "fastest convergence, peaks at round 8" claim is dead,
   and the cost-to-readiness table in Section V.E cannot be filled in from
   this run, because BNN has no interior peak to measure against.

### Open question this raises

T=20 was inherited from the original configuration, chosen when the BNN
could not learn in its hidden layers and therefore plateaued early. With a
working STE the budget looks too small. A longer diagnostic run is needed
to find where accuracy actually flattens. If it plateaus around 96 to 97,
the accuracy gap to MLP-FL closes and the paper gets stronger; if it
flattens at 95, that is still an 8 point improvement on the old final
number and worth reporting honestly.

Whatever T is chosen in the end has to be the same for every model, so
this decision has to be made before the other five runs, not after.

---

## BNN diagnostic, seed 42, T=45 (in progress)

Run started 2026-08-21, Colab T4, checkpointed to Drive every round via
`--resume`. Stopped by Colab's daily quota at round 32/45; resumable, no
data lost, continues from round 33 whenever the session is available again.

| Round | Acc | Round | Acc | Round | Acc |
|---:|---:|---:|---:|---:|---:|
| 21 | 88.46 | 25 | 92.08 | 29 | 83.09 |
| 22 | 93.88 | 26 | 90.85 | 30 | **95.59** |
| 23 | 89.05 | 27 | 94.59 | 31 | 95.13 |
| 24 | 85.81 | 28 | 90.16 | 32 | 91.75 |

Mean, rounds 28-32: 91.14 (against 90.7 for rounds 16-20 in the T=20 run).
New peak so far: **95.59% at round 30**, surpassing the old run's 94.91%.

Still noisy round to round, no clear flattening yet through round 32.
Genuinely undetermined whether this settles somewhere in the 92-96% band
or keeps drifting upward; need rounds 33-45 to tell. Continue with the
identical command:

```
python federated_train.py --model BNN --seed 42 --rounds 45 --resume
```

**Update, resumed 2026-08-22, rounds 33-35:** note this segment ran on
**CPU** (`Device: cpu` printed — no GPU runtime connected that session),
slower but not incorrect, results are still valid.

| Round | Acc |
|---:|---:|
| 33 | **97.36** |
| 34 | 97.06 |
| 35 | 95.67 |

A clear jump from round 32's 91.75%, and now within a point of MLP-FL's
best-ever peak (97.78%). First real sign the model may be settling near
MLP-FL's range rather than the 92-96% band guessed earlier.

**Rounds 36-45, completed 2026-08-22, GPU (cuda):**

| Round | Acc | Round | Acc |
|---:|---:|---:|---:|
| 36 | 97.30 | 41 | 92.56 |
| 37 | 97.41 | 42 | 97.50 |
| 38 | 97.45 | 43 | 92.17 |
| 39 | 96.17 | 44 | **97.68** |
| 40 | 97.44 | 45 | 97.60 |

**Best: round 44, 97.68%. Final: round 45, 97.60%.**

### This is a genuine plateau, not another noise excursion

Six of the last ten rounds sit at 97.3-97.7%. The two dips (round 41 at
92.56, round 43 at 92.17) recover immediately to the same band rather than
wandering off the way the old run did after its round-8 peak. The model
has found and is holding a high-accuracy region.

### What this changes in the paper

**Old broken-STE run:** peak 97.06% (round 8), final 86.89%.
**New working-STE run:** peak 97.68% (round 44), final 97.60%.

Against the **old, not-yet-rerun** MLP-FL numbers (peak 97.78% round 17,
final 97.46%), the fixed BNN-FL is now **0.10 points from MLP-FL's peak**
and **0.14 points above MLP-FL's final accuracy**. The near-1-point
accuracy penalty that framed every efficiency claim in Sections V and VI
has nearly disappeared. This has to be confirmed against a freshly
rerun MLP-FL at the same T and seed before it goes in the paper, but
directionally the story has changed from "trade a little accuracy for a
lot of efficiency" to "keep accuracy and still get the efficiency."

**One claim is now dead and cannot be revived:** "BNN-FL achieves the
fastest convergence, peaking at round 8." Its peak is now at round 44 of
45, essentially the end of training. Whatever replaces the convergence
narrative has to be honest that BNN-FL takes as long as or longer than
the baselines to reach its accuracy, not less.

**Unaffected:** every architecture-derived efficiency number (payload,
FLOPs/BOPs, energy proxy) is unchanged, since those come from the model
definition, not from this run's accuracy trajectory.

### Consequence for T

T=20 was too short for a properly trained BNN; the plateau only firms up
after round 36. **T=45 must now be used for every remaining model**
(MLP, CNN, LSTM, MLP-INT8, BNN-INT8IO), or the comparison is not on equal
footing. MLP-FL under the old T=20 run peaked early (round 17) and is
expected to plateau well before round 45, so this should not
disadvantage it, but it must be rerun to confirm rather than assumed.

---

## MLP, seed 42, T=45

Run 2026-08-22 (resumed partway through, GPU). Full round-by-round log has
severe oscillation, loss included for the first time this run since
`evaluate_loss` was just added:

Round 23 hit the run's best: **95.86%** (loss 0.147, the lowest loss in
the whole run — the two agree, which is reassuring). Several rounds have
loss in the double digits (round 25: 10.97, round 26: 13.77) alongside
mid-60s accuracy. **Final round (45): 63.53%**, one of the worst rounds
in the entire run.

### This is not a new bug

The old T=20, unseeded MLP run showed the same character of instability:
round 19 at 59.99% followed immediately by round 20 at 97.46%. MLP has
always oscillated this hard round to round; loss was never tracked
before, so how severe it gets (double-digit cross-entropy) was invisible
until now. This matches the paper's own Limitations text on client-side
Adam momentum interacting with server-side FedAvg averaging.

### Consequence: "final round accuracy" is not a trustworthy metric

Given oscillation this large, whichever round training happens to stop
on is close to arbitrary. BNN's T=45 run happened to end on a good
stretch (rounds 36-45 mostly 97%+), so its final accuracy looked
consistent with its best. MLP's happened to end on a bad round. Reporting
"BNN final 97.60% vs MLP final 63.53%" as if it reflects a real
difference between the models would be reporting noise as signal.

**Decision: stop treating "final round" as a headline number.** Report
Best accuracy (already tracked, and cross-validated by its low loss) as
the primary comparison, plus the mean and std of the last 5 rounds as a
secondary robustness figure, once every model has been rerun.

Learning-rate decay would likely reduce this oscillation directly (this
is exactly what the paper's Limitations section already names as future
work), but implementing and rerunning six models with it is out of scope
for finishing this revision. Worth flagging to the advisor as a concrete,
already-diagnosed improvement for a follow-up version.

---

## CNN, seed 42, T=45

Run 2026-08-23, GPU, across two sessions (round 33 completed in the first
session but its line was not captured before the cutoff; it is in the
checkpoint history, resume picked up correctly at 34).

**Best: round 37, 95.14% (loss 0.207). Final: round 45, 74.79%.**

### CNN is not the weak baseline the old run made it look like

| | Old run (T=20, unseeded) | New run (T=45, seed 42) |
|---|---:|---:|
| Best accuracy | 77.04% (R12) | **95.14% (R37)** |
| Final accuracy | 69.56% | 74.79% |

An 18-point jump in best accuracy. The old T=20 budget was cutting CNN
off long before it converged, exactly as it was cutting off BNN. Its
loss trace confirms this: loss was still descending into the 0.2 range
around rounds 27-37, well past where the old run stopped.

This matters for the paper beyond a single number. The submitted
manuscript leaned on CNN-FL being far behind (77.04%) to make BNN-FL's
efficiency look like a free win against a weak competitor. Under an
equal, adequate round budget that gap largely closes, and the honest
comparison is between models that all reach the mid-90s.

### Standings so far, all at T=45 / seed 42 (the fair comparison)

| Model | Best acc | Round | Final acc |
|---|---:|---:|---:|
| **BNN** | **97.68%** | 44 | 97.60% |
| MLP | 95.86% | 23 | 63.53% |
| CNN | 95.14% | 37 | 74.79% |

On this like-for-like basis BNN-FL currently has the **highest best
accuracy of the three**, which is a stronger position than the submitted
paper ever claimed (it conceded MLP-FL was ahead by 0.72 points). Two
cautions before anyone writes that down: this is a single seed, and the
old unseeded MLP run reached 97.78%, so MLP's 95.86% here may be partly
partition luck rather than a real drop. Multi-seed runs would settle it.

### Third confirmation that final-round accuracy is noise

BNN ended on 97.60%, MLP on 63.53%, CNN on 74.79% — while their bests
are 97.68%, 95.86% and 95.14%. Whichever round training stops on is
close to arbitrary for all three models, not just MLP. The decision to
report Best accuracy plus last-5-round mean ± std, rather than final
round, is now supported by three independent runs.

---

## LSTM, seed 42, T=45

Run 2026-09-03, GPU, across five sessions (resumed at rounds 5, 21, 37,
44 — `--resume` held up correctly every time, checkpoint round-counting
was exact across all five reconnects).

**Best: round 13, 95.47% (loss 0.142). Final: round 45, 86.41%.**

| | Old run (T=20, unseeded) | New run (T=45, seed 42) |
|---|---:|---:|
| Best accuracy | 89.36% (R5) | **95.47% (R13)** |
| Final accuracy | 74.32% | 86.41% |

Same pattern as CNN and BNN: T=20 was cutting LSTM off before it found
its real peak, this time by about 6 points. LSTM also confirms something
CNN and MLP already showed — after finding a strong peak early (round
13 of 45), the model spends the rest of the budget oscillating in the
72-94% band without ever regaining that peak. Whatever destabilises
these FedAvg + Adam runs past their best round affects every
architecture tried so far, not just BNN.

## Standings, all four full-precision-vs-binarized models done (T=45, seed 42)

| Model | Best acc | Round | Final acc |
|---|---:|---:|---:|
| **BNN** | **97.68%** | 44 | 97.60% |
| MLP | 95.86% | 23 | 63.53% |
| LSTM | 95.47% | 13 | 86.41% |
| CNN | 95.14% | 37 | 74.79% |

Three baselines now cluster tightly at 95.1-95.9%, with BNN clearly
ahead at 97.68%. Remaining: MLP-INT8, BNN-INT8IO. Same caveat as before
applies until multi-seed is run: this is one seed, and the old unseeded
MLP peak (97.78%) suggests today's 95.86% could be partly partition
luck rather than a real number to build the paper's framing on.

---

## MLP-INT8, seed 42, T=45

Run 2026-09-05, GPU, across four sessions (checkpoint at Drive-backed
`runs/`, resumed at rounds 7, 29, 32, 41 — all continuous, no gaps).

**Best: round 29, 87.85% (loss 1.274, notably not the lowest loss in the
run — see below). Final: round 45, 71.87%.**

### This is the weakest model tried at T=45, and that itself is informative

| Model | Best acc | Round |
|---|---:|---:|
| BNN | 97.68% | 44 |
| MLP (float32) | 95.86% | 23 |
| LSTM | 95.47% | 13 |
| CNN | 95.14% | 37 |
| **MLP-INT8** | **87.85%** | 29 |

MLP-INT8 trails the float32 MLP-FL by 8 points at best accuracy, and its
loss trace is the most unstable of any run so far: it spikes to 7.45 at
round 33, 5.72 at round 36, and **11.75 at round 43** — worse than
anything seen in the float32 baselines. Best accuracy (87.85%, round 29)
does not correspond to the lowest loss in the run, unlike every other
model's best round, which is itself a sign of how noisy this
particular run is.

### This complicates the "8-bit is nearly free" framing in Section V's
### new int8 subsection, in a way worth writing up honestly

The common assumption, and the one the paper's new int8 discussion
(Section V.C, "Comparison Against an 8-Bit Quantized Baseline") leaned
on, is that 8-bit quantization costs little to no accuracy. That holds
in centralized, non-federated settings, where int8 quantization is
extremely well studied. It does not obviously hold here: quantizing
*every* dense layer of MLP-FL with a straight-through estimator, inside
an already-unstable FedAvg+Adam federated loop, costs a real 8 points
of accuracy and visibly worsens the round-to-round instability already
documented for the float32 models.

This is a useful result, not just a disappointing one, because it
changes what BNN-INT8IO is actually being compared against and why it
might be expected to do better. BNN-INT8IO does **not** quantize every
layer to int8 the way MLP-INT8 does — it keeps the hidden layers binary
(already trained and stable, per the BNN run) and quantizes only the
two Float32 layers, input and output. If BNN-INT8IO's accuracy holds up
close to plain BNN-FL's 97.68% once that run finishes, the honest
takeaway is not "int8 is free," it's "quantizing only the layers that
were never binarized preserves accuracy better than quantizing
everything," which is a more specific and more defensible claim, and a
more interesting one for the paper. If BNN-INT8IO's accuracy also drops
substantially, that is equally worth reporting and would mean the
int8-IO variant should be presented as a communication-vs-accuracy
trade-off rather than a strict improvement over BNN-FL.

**Do not write the accuracy number into Section V.C's int8 subsection
until BNN-INT8IO's run is in hand** — the comparison only means
something once both sides are measured.

---

## BNN-INT8IO, seed 42, T=45 — the sixth and last model, all runs now complete

Run 2026-09-09, GPU, across three sessions (resumed at rounds 11, 35 —
both continuous).

**Best: round 39, 88.02% (loss 0.362, the lowest loss in the run — the
two agree, same as they did for MLP-INT8's best round). Final: round
45, 86.69%.**

### This overturns the hypothesis written into this file and into
### main.tex's Section V.C — say so plainly, do not quietly drop it

The earlier entry for MLP-INT8 reasoned that BNN-INT8IO should hold
close to plain BNN-FL's 97.68%, because it only quantizes the two
Float32 layers while leaving the already-stable binary hidden layers
untouched, unlike MLP-INT8 which quantizes everything. That reasoning
was wrong. BNN-INT8IO's best accuracy (88.02%) is nearly identical to
MLP-INT8's (87.85%), a difference of 0.17 points, and both sit roughly
10 points below plain BNN-FL.

The loss trace rules out the obvious alternative explanation. BNN-INT8IO
trains far more smoothly than MLP-INT8 did: its loss stays inside
0.36-0.65 for the entire run and never spikes into the double digits the
way MLP-INT8's did (round 43: 11.75). So this is not the same kind of
optimizer instability seen in MLP-INT8 and, less severely, in every
float32 model past its peak round. BNN-INT8IO converges cleanly to a
ceiling around 86-88% and stays there. The accuracy loss looks
structural, not a training-stability artifact: quantizing the input
layer to 8 bits costs real information before the network has extracted
any features from it, and 8 bits is evidently not enough to avoid that
cost here even though it preserves far more resolution than 1 bit would.

### What this changes about the design's central claim

Two comparisons now have to be kept separate, because they say different
things:

**BNN-INT8IO vs. MLP-INT8** (same precision tier — both use 8-bit
quantization somewhere): BNN-INT8IO wins. 13.23 KB against 18.98 KB per
round, 30.3% less, at accuracy that is a statistical tie (88.02% vs.
87.85%). This is a genuinely defensible claim: *if* a deployment is
going to accept int8-level accuracy, the hybrid design (binary hidden
layers, int8 input/output) reaches that accuracy tier more cheaply than
naively quantizing every layer.

**BNN-INT8IO vs. the original hybrid BNN-FL** (Float32 input/output):
BNN-INT8IO is not an improvement, it is a trade-off. It gives up roughly
10 points of accuracy (97.68% to 88.02%) to save 53% of the payload
(28.03 KB to 13.23 KB). Whether that trade is worth making depends on
the deployment's priorities, and the paper should present it as a
second option alongside the primary BNN-FL result, not as a strict
replacement for it.

This also retroactively justifies something the original design already
did for a different stated reason. Section III.C keeps the input and
output layers in Float32 specifically because binarizing them was
expected to cost too much accuracy. This result shows the same is true,
to a real if lesser degree, even at 8 bits — full precision on those two
layers is doing more work than it looked like it was doing.

### Standings — all six models, T=45, seed 42

| Model | Best acc | Round | Final acc |
|---|---:|---:|---:|
| **BNN** | **97.68%** | 44 | 97.60% |
| MLP | 95.86% | 23 | 63.53% |
| LSTM | 95.47% | 13 | 86.41% |
| CNN | 95.14% | 37 | 74.79% |
| BNN-INT8IO | 88.02% | 39 | 86.69% |
| MLP-INT8 | 87.85% | 29 | 71.87% |

### What goes in Section V.C now

Replace the `%TODO` placeholder with both numbers and both comparisons
above. The honest framing is: BNN-FL (Float32 I/O) is the primary
result at 97.68% best accuracy and 28.03 KB/round. BNN-INT8IO is offered
as a lower-communication variant (13.23 KB/round, 30.3% less than the
equivalent-accuracy MLP-INT8 baseline) for deployments that can accept
roughly 88% accuracy in exchange for less than half the payload. It is
not claimed to match BNN-FL's accuracy.

---

## LOCAL RERUN, RTX 4060, seed 42, T=45 — this is now the canonical set

Run 2026-09-10/11 on the local RTX 4060 (torch 2.6.0+cu124), not Colab.
Reason for the rerun: `federated_train.py` previously overwrote its
checkpoint every round, so only the last round's weights survived and
the security metrics (Macro F1 / MCC / FPR / confusion matrix) could
not be computed at the best-accuracy operating point the paper
reports. The code now writes `{tag}_best.pt` whenever a new best is
found, so this rerun produces both.

All six histories verified: 45 rounds each, contiguous, no gaps or
repeats. Every `_best.pt` accuracy matches its run's own best-round
history entry exactly, so the best-checkpoint mechanism is confirmed
working.

### Table V data — measured on the best-round checkpoint of each model

| Model | Acc (%) | Best round | Macro F1 | Precision | Recall | MCC | FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BNN** | **97.30** | 38 | 0.5960 | 0.6366 | 0.5932 | **0.9704** | **0.08** |
| LSTM | 96.22 | 11 | **0.6066** | **0.7064** | **0.6040** | 0.9588 | 0.12 |
| MLP | 92.08 | 28 | 0.5748 | 0.6265 | 0.5770 | 0.9143 | 0.25 |
| CNN | 89.18 | 30 | 0.5991 | 0.6974 | 0.5924 | 0.8830 | 0.35 |
| BNN-INT8IO | 85.08 | 42 | 0.5413 | 0.5845 | 0.5365 | 0.8372 | 0.48 |
| MLP-INT8 | 84.59 | 32 | 0.5432 | 0.5890 | 0.5492 | 0.8326 | 0.49 |

### Table VII data — efficiency, architecture-derived

| Model | Params | FLOPs (M) | BOPs (M) | IOPs (M) | Payload (KB) | Float32 payload (KB) | Inference (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 15,938 | 0.0161 | — | — | 63.77 | 63.77 | 41.8 |
| CNN | 84,962 | 0.4497 | — | — | 333.40 | 333.40 | 48.0 |
| LSTM | 53,634 | 1.5983 | — | — | 209.51 | 209.51 | 48.9 |
| MLP-INT8 | 15,938 | 0.0008 | — | 0.0153 | 18.98 | 63.77 | 57.1 |
| **BNN** | 32,514 | 0.0060 | 0.0266 | — | 28.03 | 128.78 | 51.4 |
| BNN-INT8IO | 32,514 | 0.0009 | 0.0266 | 0.0051 | 13.23 | 128.78 | 48.3 |

### A second thop blind spot, found and fixed during this run

`QuantLinear` subclasses `nn.Linear`, so `thop` skipped it exactly the
way it skipped `BinaryLinear` before the original audit. MLP-INT8 was
reporting 0.0008M operations against MLP's 0.0161M despite being the
same architecture at a different precision. A handler is now
registered for `QuantLinear` too, and 8-bit integer MACs are reported
as a separate **IOPs** column rather than folded into FLOPs or BOPs.
The check: MLP-INT8's 0.0008M float + 0.0153M int8 = 0.0161M, exactly
matching float32 MLP's total, which is what an identical architecture
at a different precision must produce. BNN-INT8IO likewise splits
BNN's 0.0060M float into 0.0009M float + 0.0051M int8, the two
input/output layers having moved to int8.

### The finding that matters most: same seed, different results

These numbers are **not** the same as the earlier Colab runs of the
identical configuration (T=45, seed 42, same code logic):

| Model | Colab run | Local rerun | Difference |
|---|---:|---:|---:|
| BNN | 97.68 (R44) | 97.30 (R38) | -0.38 |
| LSTM | 95.47 (R13) | 96.22 (R11) | +0.75 |
| MLP | 95.86 (R23) | 92.08 (R28) | **-3.78** |
| CNN | 95.14 (R37) | 89.18 (R30) | **-5.96** |
| BNN-INT8IO | 88.02 (R39) | 85.08 (R42) | -2.94 |
| MLP-INT8 | 87.85 (R29) | 84.59 (R32) | -3.26 |

Seeding `numpy` and `torch` does not make GPU training reproducible
across different hardware and library versions: the Colab runs used a
T4 with Colab's torch build, these used an RTX 4060 with torch
2.6.0+cu124, cuDNN picks different kernels, and the `DataLoader`
workers carry their own RNG state. `torch.backends.cudnn.deterministic`
was never set either.

**This is not a bug, and it is not something to hide — it is the
strongest argument yet for the multi-seed requirement.** Two runs that
differ only in hardware disagree by up to 5.96 points (CNN). Any
single-seed claim in this paper, including the ordering of the three
baselines, is therefore not trustworthy on its own, and a reviewer
would be right to say so. Reporting mean ± std over several seeds is
no longer a nice-to-have.

**What survives unchanged:** BNN-FL still has the highest best accuracy
of all six models, the highest MCC, and the lowest FPR, in both runs
independently. That is the paper's central claim and it held up across
a hardware change.

**What changes:** the three float32 baselines no longer cluster
tightly. In the Colab runs they sat at 95.1-95.9; here they spread
across 89.18-96.22, and LSTM rather than MLP is the closest competitor.
Any prose describing the baselines as "clustered" or naming MLP as the
runner-up has to be rewritten against this set.

**Decision needed:** the local set is the one to use going forward
(it is the only one with best-round checkpoints and full security
metrics), and the Colab numbers should be reported as what they are —
an incidental reproducibility observation worth a sentence in
Limitations, not deleted quietly.

---

## BNN-MATCHED and BNN-FULL ablations, seed 42, T=45 — the capacity confound is resolved

Both trained locally on the RTX 4060, both complete at 45/45 rounds,
contiguous histories, `_best.pt`/`_final.pt`/`_history.pkl` all present
and verified.

### Why these two runs exist

Two problems were found while writing the algorithm block for the
paper, not by running anything new — they came from reading the code
against the manuscript's own claims.

**The capacity confound.** `BNNModel` carries an extra 128x128
binarized layer that `MLPModel` does not have. That single layer holds
16,384 of BNN's 31,680 weights — 51.7% of the whole model. So every
accuracy comparison in this paper up to now (BNN 97.30% vs MLP 92.08%)
was between a model with roughly double the capacity and one with
half, not a clean precision-only comparison. The layer was never
documented in the original manuscript; it surfaced during the STE
audit and was only added to Table 2 afterwards with no justification
for why it exists. `BNN-MATCHED` removes it: weight matrices are
31x128, 128x64, 64x32, 32x34, identical in shape to MLPModel, giving
15,296 weight elements in both (15,746 total parameters including
biases/BatchNorm, against MLP's 15,938).

**The uplink is not binary.** Clients upload real-valued latent
weights, not binary ones — `fedavg` must average in full precision,
since averaging ±1 directly produces fractions that are neither valid
binary weights nor meaningful magnitudes (confirmed empirically: a
locally-trained hidden layer has ~3,942 distinct values before
aggregation, the global re-binarized model has exactly 2). So the
28.03 KB figure used everywhere so far is the **downlink** only. The
real per-client **uplink** is 128.78 KB (the full float32 state dict),
making the round trip 156.81 KB against MLP's 127.54 KB — BNN sends
**22.9% more**, not less, once both directions are counted. Every
figure/table/prose instance saying "uplink payload" for the 28.03 KB
number is wrong and needs correcting to "downlink" or "per-round
payload" (ambiguous, avoid).

An int8-uplink transport fix was considered and explicitly rejected:
it's a generic trick any model can use, so applying it only to BNN
would recreate the exact naming-bias bug the original STE audit found
(only BNN's layers were named in a way the old payload formula
recognized). BNN-MATCHED fixes the uplink problem as a side effect of
fixing the capacity problem, with no transport trick needed.

### Results

| Model | Best acc (%) | Round | MCC | FPR (%) | Params | Downlink (KB) | Uplink f32 (KB) | Round-trip (KB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BNN-MATCHED** | **97.75** | 38 | **0.9754** | **0.07** | **15,746** | 23.52 | 62.27 | **85.79** |
| BNN (current) | 97.30 | 38 | 0.9704 | 0.08 | 32,514 | 28.03 | 128.78 | 156.81 |
| LSTM | 96.22 | 11 | 0.9588 | 0.12 | 53,634 | 209.51 | 209.51 | 419.02 |
| MLP | 92.08 | 28 | 0.9143 | 0.25 | 15,938 | 63.77 | 63.77 | 127.54 |
| CNN | 89.18 | 30 | 0.8830 | 0.35 | 84,962 | 333.40 | 333.40 | 666.80 |
| BNN-INT8IO | 85.08 | 42 | 0.8372 | 0.48 | 32,514 | 13.23 | 128.78 | 142.01 |
| MLP-INT8 | 84.59 | 32 | 0.8326 | 0.49 | 15,938 | 18.98 | 63.77 | 82.75 |
| **BNN-FULL** | **71.08** | 31 | 0.6959 | 0.93 | 32,514 | 8.90 | 128.78 | 137.68 |

BNN-MATCHED full history: last5 mean 94.60%, std 4.88; final round
96.30%. Energy (Horowitz 45nm pricing, same method as the rest of the
paper): BNN-MATCHED 25.15 nJ, BNN-FULL 5.09 nJ (cheap because it is
almost entirely binary ops, which are priced at 0.03 pJ), against
BNN's 28.40 nJ and MLP's 74.06 nJ.

Cost-to-readiness (round-trip KB, K=5 clients):
- To 85% accuracy: BNN-MATCHED round 11 (4.61 MB), MLP round 7 (4.36
  MB), BNN round 9 (6.89 MB), LSTM round 5 (10.23 MB), CNN round 14
  (45.58 MB)
- To 90%: BNN-MATCHED round 13 (5.45 MB), BNN round 17 (13.02 MB), MLP
  round 28 (17.44 MB), LSTM round 5 (10.23 MB, already past 90 by
  round 5), CNN not reached

Note MLP is now marginally cheaper than BNN-MATCHED to reach 85% on a
*round-trip* basis (4.36 vs 4.61 MB) because MLP needs fewer rounds (7
vs 11), even though BNN-MATCHED's per-round cost is lower — but
BNN-MATCHED pulls ahead by 90% (5.45 vs 17.44 MB) because MLP needs a
lot more rounds to close that gap. This has to be stated honestly, not
smoothed over: on a strict round-trip basis, "cheapest to a working
model" now depends on which accuracy threshold counts as "working."
The downlink-only comparison (what the original 28.03 KB claims were
built on) still favors BNN-MATCHED unambiguously at every threshold.

### What this settles

BNN-FULL's collapse to 71.08% — and specifically its trajectory,
climbing normally through round ~10 then going completely flat at
70.99–71.08% for 21 straight rounds — is a clean confirmation that
keeping the input/output layers in Float32 is load-bearing, not just
citation-justified. Binarizing those two layers does not degrade
gracefully; it caps learning entirely. This is the strongest, most
direct evidence in the paper for the hybrid-precision design choice,
and it was previously asserted on citation alone.

BNN-MATCHED **beats plain BNN** on every headline metric — accuracy,
MCC, FPR — while using half the parameters and a smaller downlink.
This is the opposite of a disappointing ablation: it means the extra
128x128 layer was not just an uncontrolled confound, it was actively
unhelpful. The honest framing for the paper is that **BNN-MATCHED,
not the original BNN, should be the primary proposed model going
forward**, since it is simultaneously the most accurate, the cheapest,
and the one whose comparison to MLP is actually fair. The original BNN
architecture (with the extra layer) should be kept in the paper only
as a secondary variant showing that adding capacity does not help
once precision is already controlled for.

### DONE (2026-09-15/16) — all six items below completed in main.tex

1. [x] "Uplink" renamed to "downlink" everywhere it mislabeled the KB
   figures. Table VII was split into two tables: "Per-Inference
   Operations and Energy" (compute only) and "Per-Round Communication,
   Both Directions" (`tab:comm`, downlink/uplink/round-trip columns).
2. [x] BNN-MATCHED replaced the original BNN as "BNN-FL (Proposed)"
   throughout — abstract, contributions, Table V, Table VI, Table VII,
   Table IX (cost-to-readiness), per-class/ROC/misclassification
   prose, Limitations, Conclusion. The original BNN is demoted to
   "BNN-FL (extra layer)" in the new ablation section only.
3. [x] All figures regenerated with `ORDER = ["BNN-MATCHED", "MLP",
   "LSTM", "CNN", "BNN-INT8IO", "MLP-INT8"]` as the primary series and
   a new `ABLATION_ORDER`/`fig_ablation()` for BNN vs BNN-MATCHED vs
   BNN-FULL. Two rename-era bugs found and fixed while doing this:
   `fig_pareto`'s `OFFSET` dict and marker-size check were still keyed
   on the old `"BNN"` string (so BNN-MATCHED silently lost its
   hand-tuned label position), and every "Uplink payload" axis label
   was actually plotting downlink data.
4. [x] New subsection added: "Ablation Study: Isolating Capacity and
   Full Binarization" (`sec:ablation`), placed right after Security
   Performance Comparison. Table VI (ablation) + Fig. 4 report BNN vs
   BNN-MATCHED vs BNN-FULL side by side. Abstract/contributions/
   conclusion updated to BNN-MATCHED's 97.75%/0.9754/0.07%.
5. [x] Table II (architecture) rewritten to show the matched 4-layer
   design as primary, with a paragraph explaining the extra layer was
   removed and pointing to the ablation section. Fig. 1
   (`final dig.pdf`) did NOT need replacing — it shows a generic
   "Hidden Layer (Binary)" box, not layer-specific shapes, so it is
   still accurate. It does still say "Rounds (T): e.g., 20" (stale,
   cosmetic) — this is a designed graphic with no source file
   available in this session, so it needs fixing by hand in whatever
   tool made it (Canva-style), not something fixable from LaTeX.
6. [x] `figures.py`'s `ORDER`/`STYLE`/`LABEL`/`ABLATION_ORDER` updated.
   `evaluate.py`'s `--models` default already covered every registered
   model automatically, no change needed there.

### One finding this pass surfaced that is NOT yet fixed

`BNN-INT8IO` was trained before the capacity confound was found and
still carries the original unmatched (32,514-parameter) architecture.
Its downlink is genuinely smaller than MLP-INT8's, but once the uplink
is counted (128.78 KB, the full unmatched Float32 size), its round
trip is 142.01 KB against MLP-INT8's 82.75 KB — 71.6% MORE, not less.
main.tex's Section V.F (int8 comparison) has been rewritten to state
this honestly: BNN-INT8IO cannot currently be recommended over
MLP-INT8, and a matched-capacity int8IO variant is flagged as future
work (Limitations item 5, Conclusion's five future directions). No
retraining was done for this — it's an accurate description of what
exists, not a new experiment.

---

## All eight models complete — remaining work moves to the paper

All planned runs (BNN, MLP, CNN, LSTM, MLP-INT8, BNN-INT8IO,
BNN-MATCHED, BNN-FULL) are done at the fixed T=45, seed=42
configuration on the same RTX 4060 and are directly comparable. What
is left is writing the results into `main.tex`, including the
ablation section above, not further training.

Optional, not yet decided: SignSGD aggregation ablation, class-weighted
loss re-run, learning-rate decay re-run, multi-seed runs for error bars.

---

## Non-IID Dirichlet partitioning + external baseline reproduction, T=45, seed=42

Run 2026-09-15/18 on the local RTX 4060. Two new pieces of work, both
completed: (1) the six headline models retrained under label-skew
non-IID client partitioning (Hsu et al. 2019, `--partition dirichlet`)
at two severities, alpha=0.1 (severe) and alpha=0.5 (moderate), against
the existing IID baseline; (2) a faithful reproduction of BiPruneFL
(S. Lee, H. Jang, IEEE Access 2025) as an external SOTA baseline,
`BiPruneFL-Repro` in `MODEL_REGISTRY`, trained at IID under the same
protocol as everything else.

Implementation details, all in `code/`: `dirichlet_client_indices()`
and the `--partition`/`--alpha` flags in `federated_train.py`;
`edgepop_ops.py` (the Biprop/edge-popup mechanism BiPruneFL-Repro
reproduces — frozen random weight, learned pruning score, top-k mask
via a straight-through estimator) and the `BiPruneFLReproModel` class
in `models.py`. Full design reasoning, including the two points where
the original paper's paywalled full text forced a documented
substitute (the amplitude-gain formula, the aggregation protocol) and
the evidence-based correction to keep the input/output layers full
precision (checked against the actual `chrundle/biprop` reference
implementation rather than assumed), is in the code comments and
`main.tex`'s ablation section.

### Non-IID results — a non-monotonic finding, not a simple trend

| Model | IID (%) | alpha=0.5 (%) | alpha=0.1 (%) |
|---|---:|---:|---:|
| **BNN-MATCHED** | **97.75** | **89.47** | 68.20 |
| MLP | 92.08 | 73.18 | 71.07 |
| LSTM | 96.22 | 79.95 | **72.00** |
| CNN | 89.18 | 83.60 | 69.85 |
| BNN-INT8IO | 85.08 | 79.60 | 69.16 |
| MLP-INT8 | 84.59 | 80.11 | 70.80 |

**BNN-MATCHED is not simply "more sensitive to non-IID data" — it is
the best model at IID and at moderate skew, and specifically the
*worst* of all six at severe skew.** At alpha=0.5 it leads by 5.9
points over the next-best model (CNN, 83.60%); at alpha=0.1 every
other model beats it, by up to 3.8 points (LSTM). This is a threshold
effect, not a gradual decline, and it has to be reported as such: a
reviewer who only sees "accuracy drops under non-IID" would be reading
a different, less accurate story than what the data shows. Worth
investigating further (not yet done) whether this is specific to how
re-binarization interacts with highly skewed per-client updates.

### BiPruneFL-Repro — a real external comparison, not a strawman

At IID (`runs/results_best.csv`), `BiPruneFL-Repro` reaches **97.96%
best accuracy** (round 41), MCC 0.9777, FPR 0.06% — matching or
slightly exceeding BNN-MATCHED's 97.75% / 0.9754 / 0.07% on every
security metric, at the same parameter budget (15,938 vs 15,746,
capacity-controlled by design). Final-round accuracy ended low (80.86%
at round 45, same FedAvg+Adam oscillation seen in every model in this
project) — best-round is the metric this project reports, per the
"Third confirmation that final-round accuracy is noise" decision
earlier in this file.

**The honest differentiator is communication, not accuracy.**
BiPruneFL-Repro's score tensor has to be transmitted in full Float32
every round for FedAvg to average it correctly — there is no
downlink/uplink asymmetry the way BNN-MATCHED has, because the frozen
weights need never be sent after round 1 but the *trainable* part
(the score) is exactly as expensive as an ordinary float parameter.
Its round-trip payload is **63.77 KB**, identical to plain MLP's, and
**2.7x BNN-MATCHED's packed downlink (23.52 KB)**. So the reproduced
external baseline is a genuine, non-strawman competitor on accuracy —
it is not a competitor on the efficiency axis this paper's whole
argument rests on. That is the correct, defensible claim for `main.tex`:
"comparable to state-of-the-art accuracy, at a fraction of the
communication cost," not "beats everything on every axis."

### What is now fully complete

All 13 planned training runs are done: 8 models at IID (from the
earlier pass) + 6 headline models x 2 non-IID severities (12 runs,
one, BNN-MATCHED, shared its IID number so effectively 12 new runs)
+ BiPruneFL-Repro. `runs/results_best.csv`, `runs/results_best_dir0.1.csv`
and `runs/results_best_dir0.5.csv` all hold verified, non-fabricated
numbers, checked directly against the CSV files during this run, not
estimated.

### Still open (unchanged from before this pass)

Matched-capacity `BNN-INT8IO` (still carries the old unmatched
architecture), multi-seed runs (single seed=42 throughout — the
Colab-vs-local-hardware finding earlier in this file, up to 5.96
points difference from hardware alone, is itself the strongest
argument that single-seed claims aren't reliable), non-IID robustness
investigation (why the threshold effect above happens), Raspberry Pi
hardware energy measurement (device in hand, `code/pi_benchmark.py`
ready, not yet run — needs the Pi on the network), and `final dig.pdf`'s
stale "Rounds (T): e.g., 20" text (cosmetic, needs manual fixing in
the original design tool).

---

## Follow-up queue in progress (started 2026-09-18): matched BNN-INT8IO + sign-flip diagnostics

`run_investigation.ps1` runs seven training runs plus four evaluations.
Sign-flip diagnostics use seed=43 on purpose: seed 42 checkpoints are the
ones cited in the manuscript, and the paper's own Limitations section
documents that a fixed seed does not guarantee identical results across
hardware/library versions, so they were left untouched.

**Finished so far:**

| Run | Best acc (%) | Best round | Final acc (%) |
|---|---:|---:|---:|
| BNN-INT8IO-MATCHED, seed 42, IID | 81.75 | 32 | 76.38 |
| BNN-MATCHED, seed 43, IID | 97.05 | 38 | 94.99 |

Matched-capacity BNN-INT8IO (same 31-128-64-32-34 topology as MLP-INT8,
15,938-scale parameters) scores 81.75%, below both the old unmatched
BNN-INT8IO (85.08%) and MLP-INT8 (84.59%). The extra layer had therefore
mildly helped the int8-I/O variant, and the comparison to MLP-INT8 is
now fair. BNN-MATCHED at seed 43 reaches 97.05% against 97.75% at seed
42, a second-seed data point supporting that the IID headline is not a
seed fluke.

**Still to run:** BNN-MATCHED seed 43 at alpha=0.5 (in progress, resumed
from round 17/45), BNN-MATCHED seed 43 at alpha=0.1, and BNN-INT8IO
seed 43 at IID / alpha=0.5 / alpha=0.1. Each records `sign_flip_rate`
(fraction of binary weights that change sign between consecutive
rounds' re-binarized global model) in its `history.pkl`.

**To resume after any interruption** (safe to rerun; finished runs skip,
the interrupted one resumes from its last checkpoint):

```
cd F:\UIU\11th\green\GFIDS_BNN
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_investigation.ps1
```

Progress: `logs\investigation.log`. Caution once it finishes:
`evaluate.py` writes `results_{suffix}[_dir{alpha}].csv` without a seed
tag, so the seed-43 evaluation calls overwrite the seed-42 results CSVs;
rerun the seed-42 evaluations afterwards to restore them.

## FINAL: follow-up queue finished (2026-09-22)

- BNN-INT8IO-MATCHED (seed 42, IID): 81.75% acc, MCC 0.8029, FPR 0.59, 15,746 params, downlink 8.72 KB, uplink 62.27, round trip 70.99 KB (14.2% below MLP-INT8's 82.75 KB, 2.84 pts lower accuracy).
- Seed 43 best accuracy: BNN-MATCHED IID 97.05 / a0.5 96.72 / a0.1 70.94; BNN-INT8IO IID 85.94 / a0.5 79.29 / a0.1 64.82.
- Sign-flip (mean % of binary weights flipping per round, last 20 rounds): BNN-MATCHED IID 0.0234, a0.5 0.0029, a0.1 0.0010; BNN-INT8IO 0.0635, 0.0056, 0.0019. Fewest flips at severe skew, so the "re-binarization noise" hypothesis is refuted. The a0.5-to-a0.1 cliff replicates; the severe-skew ranking does not.
- `runs/results_best*.csv` are seed 42; seed-43 copies are `results_best_seed43_*.csv` (evaluate.py names carry no seed tag, so it overwrites).
- Manuscript fully rewritten in simple wording (local only, 14 pages, compiles clean).
