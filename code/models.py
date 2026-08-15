"""
Model definitions for GF-IDS.

Four federated architectures are compared on CICIoT2023:
  MLP-FL, CNN-FL, LSTM-FL (full-precision Float32 baselines)
  BNN-FL  (proposed hybrid-precision binarized network)

Extracted from the Colab notebooks used for the reported experiments.
INPUT_DIM and NUM_CLASSES are set by the preprocessing stage
(31 features, 34 classes for the configuration reported in the paper).
"""

import torch
import torch.nn as nn

from binary_ops import BinaryActivation, BinaryLinear
from quant_ops import QuantLinear


# ─────────────────────────────────────────────────────────────
# Baseline 1 — MLP
# Three hidden layers (128 / 64 / 32), BatchNorm + ReLU.
# Deliberately mirrors the BNN hidden widths so that the
# comparison isolates the effect of binarization.
# ─────────────────────────────────────────────────────────────
class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64),        nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Baseline 2 — 1D CNN
# Two conv blocks (64 and 128 filters, kernel size 3) over the
# feature vector, followed by a two-layer classifier head.
# ─────────────────────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2),
        )
        flat = 128 * (input_dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(flat, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────
# Baseline 3 — LSTM
# Features are treated as a length-`input_dim` sequence of
# scalars; two stacked LSTM layers (64 units, dropout 0.2).
# ─────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─────────────────────────────────────────────────────────────
# Proposed — BNN (hybrid precision)
#
# BinaryLinear applies sign() to the weight matrix at every
# forward pass, so the effective weights are constrained to
# {-1, +1}. The underlying real-valued weight tensor is kept
# by the module and is what gets stored and aggregated.
#
# Precision layout:
#   input_layer   Float32   31 -> 128
#   hidden1       Binary   128 -> 128
#   hidden2       Binary   128 -> 64
#   hidden3       Binary    64 -> 32
#   output_layer  Float32   32 -> 34
#
# Input and output layers are left in full precision: binarizing
# the input discards magnitude information before any features
# are extracted, and binarizing the output breaks the Softmax
# probability estimates needed for multi-class classification.
# ─────────────────────────────────────────────────────────────
# Weights are binarized through SignSTE, so gradients reach the
# underlying real-valued weights via the straight-through estimator
# (see binary_ops.py). With binarize_activations=True the tensor
# flowing between hidden layers is binarized too, which is what
# reduces the hidden-layer matrix products to XNOR + popcount.
# ─────────────────────────────────────────────────────────────
class BNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, binarize_activations=True):
        super().__init__()
        act = BinaryActivation if binarize_activations else nn.Hardtanh

        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden1 = nn.Sequential(
            BinaryLinear(128, 128), nn.BatchNorm1d(128), act())
        self.hidden2 = nn.Sequential(
            BinaryLinear(128, 64), nn.BatchNorm1d(64), act())
        self.hidden3 = nn.Sequential(
            BinaryLinear(64, 32), nn.BatchNorm1d(32), act())
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.output_layer(x)


# ─────────────────────────────────────────────────────────────
# Baseline 4 — INT8 quantized MLP
#
# Same topology and same training configuration as MLPModel, with
# every dense layer constrained to an 8 bit grid instead of Float32.
# This is the baseline that answers "why one bit rather than eight",
# and it is the strongest competitor to the proposed model on
# communication cost, since int8 already divides the payload by four
# while keeping accuracy essentially intact.
# ─────────────────────────────────────────────────────────────
class MLPInt8Model(nn.Module):
    def __init__(self, input_dim, num_classes, bits=8):
        super().__init__()
        self.net = nn.Sequential(
            QuantLinear(input_dim, 128, bits=bits), nn.ReLU(),
            nn.BatchNorm1d(128),
            QuantLinear(128, 64, bits=bits), nn.ReLU(),
            nn.BatchNorm1d(64),
            QuantLinear(64, 32, bits=bits), nn.ReLU(),
            QuantLinear(32, num_classes, bits=bits),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Proposed variant — BNN with int8 input and output layers
#
# The hybrid design keeps the input and output layers in Float32
# because binarizing them costs too much accuracy. Those two layers
# then dominate the payload, since together they hold 5,056 of the
# model's weights against 26,624 in the binarized hidden layers.
# Quantizing them to 8 bits rather than leaving them at 32 removes
# most of that remaining cost without the accuracy penalty that
# binarizing them would cause.
# ─────────────────────────────────────────────────────────────
class BNNInt8IOModel(nn.Module):
    def __init__(self, input_dim, num_classes,
                 binarize_activations=True, bits=8):
        super().__init__()
        act = BinaryActivation if binarize_activations else nn.Hardtanh

        self.input_layer = QuantLinear(input_dim, 128, bits=bits)
        self.hidden1 = nn.Sequential(
            BinaryLinear(128, 128), nn.BatchNorm1d(128), act())
        self.hidden2 = nn.Sequential(
            BinaryLinear(128, 64), nn.BatchNorm1d(64), act())
        self.hidden3 = nn.Sequential(
            BinaryLinear(64, 32), nn.BatchNorm1d(32), act())
        self.output_layer = QuantLinear(32, num_classes, bits=bits)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.output_layer(x)


MODEL_REGISTRY = {
    "MLP": MLPModel,
    "CNN": CNNModel,
    "LSTM": LSTMModel,
    "MLP-INT8": MLPInt8Model,
    "BNN": BNNModel,
    "BNN-INT8IO": BNNInt8IOModel,
}
