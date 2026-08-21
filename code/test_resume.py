"""
Verify that an interrupted run resumes correctly.

Trains 3 rounds, then resumes to 5 and checks that the history is
continuous, no round is repeated or skipped, and a completed run is
recognised as complete rather than retrained.

    python test_resume.py
"""

import os
import shutil

import numpy as np
import torch

import federated_train as ft

TMP = "./_resume/"
INPUT_DIM, NUM_CLASSES = 31, 34


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)

    rng = np.random.default_rng(0)
    np.save(TMP + "X_train.npy", rng.random((3000, INPUT_DIM), dtype=np.float32))
    np.save(TMP + "y_train.npy", rng.integers(0, NUM_CLASSES, 3000))
    np.save(TMP + "X_test.npy", rng.random((600, INPUT_DIM), dtype=np.float32))
    np.save(TMP + "y_test.npy", rng.integers(0, NUM_CLASSES, 600))

    ft.DATA_DIR = TMP
    ft.OUT_DIR = TMP
    ft.get_loader = lambda X, y, batch_size=256: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=batch_size, shuffle=True)

    X_tr, X_te, y_tr, y_te = ft.load_data()

    def run(rounds, resume):
        ft.set_seed(42)
        clients = ft.split_clients(X_tr, y_tr)
        return ft.federated_training(
            "BNN", clients, X_te, y_te, INPUT_DIM, NUM_CLASSES,
            rounds=rounds, epochs=1, seed=42, resume=resume)

    print("\n--- phase 1: 3 rounds, no resume ---")
    _, h1 = run(3, resume=False)

    print("\n--- phase 2: resume to 5 rounds ---")
    _, h2 = run(5, resume=True)

    print("\n--- phase 3: resume again at 5 (already done) ---")
    _, h3 = run(5, resume=True)

    rounds2 = [h["round"] for h in h2]
    ok = []
    ok.append(("phase 1 produced 3 rounds", len(h1) == 3))
    ok.append(("phase 2 produced 5 rounds", len(h2) == 5))
    ok.append(("round numbers are 1..5 with no gaps or repeats",
               rounds2 == [1, 2, 3, 4, 5]))
    ok.append(("first 3 rounds carried over unchanged",
               [h["accuracy"] for h in h2[:3]] ==
               [h["accuracy"] for h in h1]))
    ok.append(("completed run is not retrained", len(h3) == 5))

    shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 56)
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    n = sum(1 for _, p in ok if p)
    print(f"  {n}/{len(ok)} checks passed")
    print("=" * 56)
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
