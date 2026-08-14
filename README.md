# GF-IDS: Green Federated Intrusion Detection for IoT

Code and figures for:

> **Green Federated Intrusion Detection for IoT: Evaluating the
> Computation–Communication Trade-off using Binarized Neural Networks**
>
> Md. Tanzamul Azad, Israt Jerin Porshi, Md. Asif Mustoba Sazzad,
> Tahmidur Rahman Osmani, Md. Arefin Iqram, Md. Motaharul Islam
>
> Department of Computer Science and Engineering,
> United International University, Dhaka, Bangladesh

---

## Overview

GF-IDS integrates a **hybrid-precision Binarized Neural Network (BNN)** into a
**Federated Learning (FL)** loop for multi-class intrusion detection on IoT
network traffic. The goal is to cut the computation, communication and energy
cost of FL-based IDS without giving up detection performance.

The hidden layers are constrained to binary weights `{-1, +1}` while the input
and output layers stay in Float32. Binarizing the input layer would discard
magnitude information before any features are extracted, and binarizing the
output layer would break the Softmax probability estimates that multi-class
classification depends on.

The framework is evaluated on **CICIoT2023** across all 34 classes
(33 attack types plus benign) against three full-precision federated
baselines: MLP-FL, CNN-FL and LSTM-FL.

## Repository layout

```
code/
  models.py            MLP, CNN, LSTM and the hybrid-precision BNN
  preprocessing.py     cleaning, correlation pruning, scaling, encoding
  federated_train.py   FedAvg simulation over K = 5 clients, T = 20 rounds
  evaluate.py          security metrics + efficiency metrics
  figures.py           regenerates every figure in the paper
paper/
  *.png, final dig.pdf figures reported in the manuscript
docs/
  EXPERIMENTAL_SETUP.md  full configuration and reproduction notes
```

The manuscript source is not included while the paper is under review.


## Setup

```bash
pip install -r requirements.txt
```

The CICIoT2023 dataset is not redistributed here. Download it from the
Canadian Institute for Cybersecurity and extract it so that
`./ciciot2023/CICIOT23/` contains the `train/` and `test/` CSV folders:

https://www.unb.ca/cic/datasets/iotdataset-2023.html

## Reproducing the experiments

```bash
python code/preprocessing.py
```
Cleans the raw CSVs, prunes correlated features, scales to `[0, 1]`,
label-encodes the classes and writes `.npy` arrays into `./data/`.
With the configuration used for the paper this yields 31 features and
34 classes.

```bash
python code/federated_train.py --model BNN
python code/federated_train.py --model MLP
python code/federated_train.py --model CNN
python code/federated_train.py --model LSTM
```
Runs the federated simulation and writes checkpoints plus a per-round
accuracy history into `./runs/`.

```bash
python code/evaluate.py
python code/figures.py
```
Computes the metric tables and regenerates the figures.

## Configuration

| Parameter | Value |
|---|---|
| Dataset | CICIoT2023 |
| Selected features | 31 |
| Output classes | 34 (33 attacks + benign) |
| FL clients (K) | 5 |
| Partitioning | IID |
| Communication rounds (T) | 20 |
| Local epochs per round (E) | 5 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Train / test split | 80% / 20% |
| Global test set | 1,170,467 samples |
| Framework | PyTorch |
| Acceleration | NVIDIA T4 GPU |

See [`docs/EXPERIMENTAL_SETUP.md`](docs/EXPERIMENTAL_SETUP.md) for the full
description, including how the efficiency metrics are defined.

## Reading the efficiency numbers

Two different payload figures are reported by `evaluate.py` and they answer
different questions:

- **`IdealPayload(KB)`** prices each binarized hidden weight at 1 bit and
  everything else at 32 bits. It describes an idealized 1-bit encoding of the
  binary layers.
- **`MeasuredPayload(KB)`** is the size of the tensors as actually serialised
  by the training loop, which moves full float32 state dicts.

Quote whichever matches the claim being made, and state which one it is.
The same caution applies to `FLOPs(M)`: `thop` dispatches on the exact module
type, so `BinaryLinear` (a subclass of `nn.Linear`) is not matched by the
built-in rule and contributes zero unless a custom handler is registered.
`evaluate.py` runs `thop` with `verbose=True` so unhandled modules are
reported; check that output before quoting FLOPs.

## Status

The manuscript is under review. Results, figures and manuscript text in this
repository correspond to that submission and may change during revision.

## Citation

```bibtex
@article{azad2026gfids,
  title   = {Green Federated Intrusion Detection for IoT: Evaluating the
             Computation--Communication Trade-off using Binarized Neural
             Networks},
  author  = {Azad, Md. Tanzamul and Porshi, Israt Jerin and
             Sazzad, Md. Asif Mustoba and Osmani, Tahmidur Rahman and
             Iqram, Md. Arefin and Islam, Md. Motaharul},
  journal = {Under review},
  year    = {2026}
}
```

## Contact

Md. Tanzamul Azad — i.m.tanjamul@gmail.com
Corresponding author: Md. Motaharul Islam — motaharul@cse.uiu.ac.bd
