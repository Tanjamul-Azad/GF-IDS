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
Three figures are reported so the communication claim can be stated
precisely:

  PackedPayload    binary weight matrices actually bit-packed with
                   numpy.packbits, everything else at native dtype.
                   This is the real transmitted size.
  AnalyticPayload  the same thing computed analytically (1 bit per
                   binary weight, 32 bits otherwise). Should agree
                   with PackedPayload up to byte-alignment padding.
  Float32Payload   the whole state dict sent uncompressed, i.e. what
                   a naive implementation would transmit.

Quote PackedPayload for communication-efficiency claims, and use
Float32Payload to show what the packing buys.

Operation counts
----------------
FLOPs and BOPs are reported separately. Binary layers run as XNOR +
popcount rather than multiply-accumulate, so folding them into a
single FLOP number would misrepresent both the cost and the saving.
A custom thop handler is registered for BinaryLinear, since thop
dispatches on exact module type and would otherwise skip it silently.
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

from binary_ops import BinaryLinear, binary_weight_keys
from models import MODEL_REGISTRY
from quant_ops import quant_weight_keys

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


def precision_map(model):
    """Bits per element for every weight tensor that is not Float32.

    Binary layers contribute 1 bit per weight, int8 layers 8 bits.
    Anything absent from this map keeps its native dtype.
    """
    bits = {k: 1 for k in binary_weight_keys(model)}
    bits.update(quant_weight_keys(model))
    return bits


def payload_bits(model):
    """Payload with each weight tensor priced at its own precision.

    Biases and BatchNorm parameters stay Float32, matching the hybrid
    precision design. A quantized tensor also carries one Float32
    scale factor, which is counted here because the receiver cannot
    reconstruct the weights without it.

    Iterates the full state dict rather than named_parameters(),
    because FedAvg transports the whole state dict, including the
    BatchNorm running statistics. Those are buffers rather than
    parameters but still cross the network every round, so counting
    only named_parameters() understates the payload of every model
    that uses BatchNorm.
    """
    bits = precision_map(model)
    total = 0
    for name, tensor in model.state_dict().items():
        if name in bits:
            total += tensor.numel() * bits[name]
            if bits[name] > 1:
                total += 32          # per-tensor scale factor
        else:
            total += tensor.numel() * tensor.element_size() * 8
    return total


def packed_payload_bits(model):
    """Payload measured by actually encoding each tensor.

    Serialises the state dict the way a real client would. Binary
    weight matrices are packed with numpy.packbits at eight weights
    per byte; int8 layers are cast to one byte per weight plus a
    Float32 scale; everything else keeps its native dtype. This is a
    measurement of the encoded bytes rather than an analytical
    estimate.
    """
    bits = precision_map(model)
    total_bits = 0
    for name, tensor in model.state_dict().items():
        w = tensor.detach().cpu().numpy()
        if bits.get(name) == 1:
            packed = np.packbits((w > 0).astype(np.uint8).reshape(-1))
            total_bits += packed.nbytes * 8
        elif bits.get(name) == 8:
            total_bits += w.astype(np.int8).nbytes * 8 + 32
        else:
            total_bits += tensor.numel() * tensor.element_size() * 8
    return total_bits


def float32_payload_bits(model):
    """Payload if the whole state dict is sent as-is, uncompressed."""
    return sum(t.numel() * t.element_size() * 8
               for t in model.state_dict().values())


def count_binary_linear(module, x, y):
    """thop handler for BinaryLinear.

    thop dispatches on the exact module type, so BinaryLinear - being
    a subclass of nn.Linear - is not matched by the built-in rule and
    would silently contribute zero operations. Registering this
    handler makes the binary layers count like any other dense layer,
    so FLOPs across models are compared on the same basis.

    Binary layers execute as XNOR + popcount rather than
    multiply-accumulate, so these are reported separately as binary
    operations (BOPs) rather than folded into the FLOP total.
    """
    module.total_ops += torch.DoubleTensor(
        [module.in_features * module.out_features])


def efficiency_metrics(model, X_test, y_test, input_dim):
    params = sum(p.numel() for p in model.parameters())

    packed_kb = packed_payload_bits(model) / 8 / 1024
    analytic_kb = payload_bits(model) / 8 / 1024
    float32_kb = float32_payload_bits(model) / 8 / 1024

    # Operation counts, split by precision so the comparison is
    # explicit about what is a float multiply-accumulate and what is
    # a bitwise operation.
    try:
        from thop import profile
        dummy = torch.randn(1, input_dim).to(device)

        total_ops, _ = profile(model, inputs=(dummy,), verbose=False,
                               custom_ops={BinaryLinear: count_binary_linear})
        # Binary layers alone, to separate BOPs from FLOPs.
        bops = sum(m.in_features * m.out_features
                   for m in model.modules()
                   if isinstance(m, BinaryLinear))
        flops_m = round((total_ops - bops) / 1e6, 4)
        bops_m = round(bops / 1e6, 4)
    except ImportError:
        flops_m = bops_m = None

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
            "BOPs(M)": bops_m,
            "PackedPayload(KB)": round(packed_kb, 2),
            "AnalyticPayload(KB)": round(analytic_kb, 2),
            "Float32Payload(KB)": round(float32_kb, 2),
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
