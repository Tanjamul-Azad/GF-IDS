"""
Fast end to end smoke test on synthetic data.

Runs one federated round of every model on a few thousand random
samples, so that a mistake in the training loop surfaces in seconds
rather than after hours on the real dataset. Also checks that two runs
launched with the same seed produce identical results, which is what
makes a comparison between models meaningful.

    python smoke_test.py

Expects nothing on disk and needs no GPU.
"""

import os
import shutil

import numpy as np
import torch

import federated_train as ft
from models import MODEL_REGISTRY

TMP = "./_smoke/"
N_TRAIN, N_TEST, INPUT_DIM, NUM_CLASSES = 4000, 800, 31, 34


def make_fake_data():
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(TMP + "X_train.npy", rng.random((N_TRAIN, INPUT_DIM), dtype=np.float32))
    np.save(TMP + "y_train.npy", rng.integers(0, NUM_CLASSES, N_TRAIN))
    np.save(TMP + "X_test.npy", rng.random((N_TEST, INPUT_DIM), dtype=np.float32))
    np.save(TMP + "y_test.npy", rng.integers(0, NUM_CLASSES, N_TEST))


def main():
    make_fake_data()
    ft.DATA_DIR = TMP
    ft.OUT_DIR = TMP
    # single process loading keeps the test fast and portable
    ft.get_loader = lambda X, y, batch_size=256: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=batch_size, shuffle=True)

    X_tr, X_te, y_tr, y_te = ft.load_data()

    print("=" * 62)
    print("  Smoke test: 1 round, 1 epoch, synthetic data")
    print("=" * 62)

    failures = []
    for name in MODEL_REGISTRY:
        accs = []
        for _ in range(2):
            ft.set_seed(42)
            clients = ft.split_clients(X_tr, y_tr)
            _, hist = ft.federated_training(
                name, clients, X_te, y_te, INPUT_DIM, NUM_CLASSES,
                rounds=1, epochs=1, seed=42)
            accs.append(round(hist[-1]["accuracy"], 8))

        same = accs[0] == accs[1]
        status = "deterministic" if same else "NOT DETERMINISTIC"
        print(f"\n  {name:12s} acc={accs[0]:.6f}  {status}")
        if not same:
            failures.append(name)

    shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 62)
    if failures:
        print(f"  FAIL: non-deterministic under a fixed seed: {failures}")
    else:
        print(f"  All {len(MODEL_REGISTRY)} models ran and are reproducible")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
