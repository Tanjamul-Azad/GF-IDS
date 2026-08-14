"""
Evaluation of trained GF-IDS models.

Reports two families of metrics.

Security metrics, computed from predictions on the global test set:
    accuracy, macro F1, macro precision, macro recall, MCC, macro FPR

Efficiency metrics:
    trainable parameter count
    per-round payload size
    FLOPs per inference (thop)
    wall-clock inference latency

Payload size accounting
-----------------------
`payload_bits` below prices any hidden-layer weight tensor at 1 bit
per parameter and everything else at 32 bits. This is an ANALYTICAL
figure describing an idealized 1-bit encoding of the binarized
layers; it is not a measurement of the bytes the simulation actually
transfers, since `federated_train.py` moves full float32 state dicts.
`measured_payload_bits` is provided alongside it to report the size
of the tensors as they are actually serialised.

Report whichever figure matches the claim being made, and say which
one it is.
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

from models import MODEL_REGISTRY

DATA_DIR = "./data/"
RUN_DIR = "./runs/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, X, y, batch_size=1024):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=batch_size)
    preds, labels = [], []
    for X_b, y_b in loader:
        preds.append(model(X_b.to(device)).argmax(dim=1).cpu().numpy())
        labels.append(y_b.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def security_metrics(preds, labels, num_classes):
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    # Macro false positive rate, averaged over the one-vs-rest splits.
    cm = confusion_matrix(labels, preds, labels=np.arange(num_classes))
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fpr = (FP / (FP + TN + 1e-10)).mean()

    return {"Accuracy(%)": round(acc * 100, 2),
            "Macro F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "MCC": round(mcc, 4),
            "FPR(%)": round(fpr * 100, 2)}


def payload_bits(model):
    """Idealized payload: 1 bit per binarized hidden weight, else 32."""
    total = 0
    for name, param in model.named_parameters():
        is_hidden_weight = ("hidden" in name) and name.endswith("weight")
        total += param.numel() * (1 if is_hidden_weight else 32)
    return total


def measured_payload_bits(model):
    """Actual serialised size of the state dict, in bits."""
    return sum(p.numel() * p.element_size() * 8
               for p in model.state_dict().values())


def efficiency_metrics(model, X_test, y_test, input_dim):
    params = sum(p.numel() for p in model.parameters())

    ideal_kb = payload_bits(model) / 8 / 1024
    measured_kb = measured_payload_bits(model) / 8 / 1024

    # FLOPs. NOTE: thop dispatches on the exact module type, so a
    # subclass such as BinaryLinear is not matched by the built-in
    # nn.Linear rule and contributes zero unless a custom handler is
    # registered. verbose=True prints a warning for every unhandled
    # module - check that output before quoting these numbers.
    try:
        from thop import profile
        dummy = torch.randn(1, input_dim).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=True)
        flops_m = round(flops / 1e6, 3)
    except ImportError:
        flops_m = None

    # Wall-clock latency over a fixed 10k-sample slice.
    import time
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test[:10000]),
                      torch.LongTensor(y_test[:10000])),
        batch_size=1024)
    start = time.time()
    with torch.no_grad():
        for X_b, _ in loader:
            model(X_b.to(device))
    inf_ms = round((time.time() - start) * 1000, 1)

    return {"Parameters": params,
            "FLOPs(M)": flops_m,
            "IdealPayload(KB)": round(ideal_kb, 2),
            "MeasuredPayload(KB)": round(measured_kb, 2),
            "InfTime(ms)": inf_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    input_dim = X_test.shape[1]
    num_classes = int(y_test.max()) + 1

    rows = []
    for name in args.models:
        ckpt = os.path.join(RUN_DIR, f"{name}_final.pt")
        if not os.path.exists(ckpt):
            print(f"Skipping {name}: {ckpt} not found")
            continue

        model = MODEL_REGISTRY[name](input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        preds, labels = predict(model, X_test, y_test)
        row = {"Model": name}
        row.update(security_metrics(preds, labels, num_classes))
        row.update(efficiency_metrics(model, X_test, y_test, input_dim))
        rows.append(row)

        print(f"\n{'=' * 40}\n  {name}\n{'=' * 40}")
        for k, v in row.items():
            print(f"  {k:22s} {v}")

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        os.makedirs(RUN_DIR, exist_ok=True)
        df.to_csv(os.path.join(RUN_DIR, "results.csv"), index=False)
        print("\n" + df.to_string(index=False))
        print(f"\nSaved to {os.path.join(RUN_DIR, 'results.csv')}")


if __name__ == "__main__":
    main()
