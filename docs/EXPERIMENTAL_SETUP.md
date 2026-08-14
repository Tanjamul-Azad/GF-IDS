# Experimental Setup

Full configuration behind the reported GF-IDS results, and notes on how each
metric is defined.

---

## Dataset

**CICIoT2023** — a real-time IoT security benchmark containing traffic from
105 IoT devices across smart home, healthcare and industrial environments.
Roughly 6.53 million records with 46 features, spanning 33 attack types
grouped into seven families (DDoS, DoS, Reconnaissance, Brute Force,
Spoofing, Mirai, Web-based) plus benign traffic, giving a 34-class problem.

Source: https://www.unb.ca/cic/datasets/iotdataset-2023.html

The dataset is not redistributed in this repository.

## Preprocessing

Applied by `code/preprocessing.py`, in order:

1. **Cleaning** — infinite values and NaNs are dropped, then exact duplicate
   records are removed so repeated rows cannot bias local gradient updates.
2. **Correlation pruning** — the absolute Pearson correlation matrix is
   computed over the numeric features and only its upper triangle is
   inspected, so each correlated pair is considered once and a single member
   of the pair is dropped. Several thresholds were explored during
   development (0.90, 0.85, 0.80, 0.78, 0.76); `CORR_THRESHOLD` in the script
   is the value used for the reported arrays. This reduces 46 features to 31.
3. **Min-Max scaling** to `[0, 1]`, fitted on the training split only and
   then applied to the test split, so no test statistics leak into training.
   Network flow features span several orders of magnitude, so without this a
   handful of large-magnitude features would dominate.
4. **Label encoding** of the class strings to integers `[0, 33]`.

The processed arrays are cached as `.npy` files so that training runs can be
restarted without repeating the pipeline.

## Federated setup

The training corpus is shuffled and split into five equal IID shards, one per
client. Each shard is further divided 80/20 into a local training and local
test portion; only the local training portion drives federated updates. A
separate global test set of 1,170,467 samples is used for every round-by-round
evaluation, so all four models are scored on identical held-out data.

| Parameter | Value |
|---|---|
| Clients (K) | 5 |
| Partitioning | IID |
| Communication rounds (T) | 20 |
| Local epochs per round (E) | 5 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Loss | Categorical cross-entropy |
| Training samples per client | 857,184 |
| Global test set | 1,170,467 |
| Framework | PyTorch |
| Hardware | NVIDIA T4 GPU (single-GPU simulation) |

### Aggregation

The server forms a sample-count-weighted average of the client parameters:

```
W_global = sum_k (n_k / n) * W_k
```

where `n_k` is the number of training samples held by client `k` and
`n = sum_k n_k`. Weighting by sample count rather than taking an unweighted
mean keeps clients with more data proportionally represented.

## Model architectures

All four models are trained under the same federated configuration and the
same loss, so that differences trace back to architecture rather than to the
training protocol.

### BNN-FL (proposed)

| Layer | Type | Shape | Precision | Activation |
|---|---|---|---|---|
| input_layer | Dense | 31 → 128 | Float32 | ReLU |
| hidden1 | BinaryDense | 128 → 128 | Binary (±1) | BatchNorm → Hardtanh |
| hidden2 | BinaryDense | 128 → 64 | Binary (±1) | BatchNorm → Hardtanh |
| hidden3 | BinaryDense | 64 → 32 | Binary (±1) | BatchNorm → Hardtanh |
| output_layer | Dense | 32 → 34 | Float32 | Softmax |

`BinaryLinear` applies `sign()` to its weight matrix on every forward pass,
so the effective weights are constrained to `{-1, +1}` while the module keeps
an underlying real-valued weight tensor. That real-valued tensor is what gets
stored in the state dict and aggregated by the server.

Parameter count: 31,680 weights + 386 biases + 448 BatchNorm affine
parameters = **32,514**.

### Baselines

**MLP-FL** — three hidden layers (128 / 64 / 32) with BatchNorm and ReLU,
ending in a 34-class Softmax. Hidden widths deliberately mirror the BNN so
the comparison isolates the effect of binarization.
Parameter count: 15,296 + 258 + 384 = **15,938**.

**CNN-FL** — two 1D convolution blocks (64 and 128 filters, kernel size 3),
each followed by ReLU, BatchNorm and MaxPool, then a two-layer classifier
head. Captures local structure between adjacent flow statistics.
Parameter count: **84,962**.

**LSTM-FL** — the 31 features are treated as a length-31 sequence of scalars
and passed through two stacked LSTM layers (64 units, dropout 0.2) before a
two-layer head. Stands in for recurrent approaches used in earlier FL-IDS work.
Parameter count: **53,634**.

## Metrics

### Security

Accuracy, macro F1, macro precision, macro recall, Matthews Correlation
Coefficient and macro false positive rate.

Macro averaging and MCC matter here because the class distribution is
severely skewed: the largest attack class holds roughly 848,000 test
instances while the smallest hold fewer than 200. Accuracy alone is
dominated by the majority classes and gives an over-optimistic picture,
whereas macro F1 weights every class equally and MCC stays informative under
heavy imbalance.

Macro FPR is computed from the confusion matrix as the mean of the
one-vs-rest false positive rates.

### Efficiency

**Trainable parameters** — `sum(p.numel() for p in model.parameters())`.

**Payload per round** — reported two ways, because they answer different
questions:

- *Ideal payload* prices each binarized hidden weight at 1 bit and every
  other parameter at 32 bits, describing an idealized 1-bit encoding of the
  binary layers.
- *Measured payload* is the size of the tensors as actually serialised by the
  training loop, which transfers full float32 state dicts.

State clearly which of the two any given claim refers to.

**FLOPs per inference** — measured with the `thop` library on a single
sample.

> `thop` dispatches on the exact module type. `BinaryLinear` subclasses
> `nn.Linear`, so it is not matched by the built-in `nn.Linear` rule and
> contributes zero FLOPs unless a custom handler is registered.
> `evaluate.py` therefore runs `profile(..., verbose=True)`, which prints a
> warning line for every unhandled module type. Check that output before
> quoting FLOPs, and register a handler for `BinaryLinear` if the binary
> layers should be included in the count.

**Energy proxy** — a hardware-independent estimate obtained by multiplying
the FLOP count by a nominal 0.5 pJ/FLOP coefficient. Since it is derived
directly from the FLOP count, it inherits whatever that count includes or
omits. Note the unit conversion: 1 nJ = 1000 pJ.

**Inference latency** — wall-clock time over a fixed 10,000-sample slice of
the global test set at batch size 1024, measured on the T4. FLOP savings do
not automatically translate into latency savings, because standard GPUs do
not accelerate bit-packed operations without custom kernels.

## Known limitations

- **Simulated federation.** Clients are simulated sequentially on a single
  GPU. Real deployments involve heterogeneous devices, variable network
  latency and asynchronous updates, none of which are modelled here.
- **IID partitioning.** Data is split uniformly at random. Real IoT
  deployments are typically non-IID, which is harder for FedAvg.
- **Class imbalance.** Several rare classes have fewer than 150 test samples
  and are effectively undetectable at this sample count, which drags macro F1
  well below accuracy. Class-balancing strategies such as federated
  oversampling or server-side loss reweighting are untested here.
- **Single run per model.** Reported numbers come from one run per
  architecture. Round-to-round accuracy fluctuates noticeably, so multi-seed
  runs with error bars would characterise stability more reliably.
- **FedAvg only.** No comparison against FedProx, FedAdam or other
  aggregation strategies.
