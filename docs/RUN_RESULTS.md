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

## Pending

- BNN diagnostic at longer T, to locate the plateau
- BNN-INT8IO, MLP-INT8, MLP, CNN, LSTM at whatever T is settled on
- Optional: class-weighted variants, extra seeds
