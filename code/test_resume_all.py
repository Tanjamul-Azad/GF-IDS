"""
Verify --resume works identically for every model, not just BNN.

Same check as test_resume.py (train 3 rounds, resume to 5, confirm
continuity) but looped over the full MODEL_REGISTRY, so a bug specific
to CNN, LSTM, or the int8 variants is caught here rather than after
hours of real training.

    python test_resume_all.py
"""

import os
import shutil

import numpy as np
import torch

import federated_train as ft
from models import MODEL_REGISTRY

TMP = "./_resume_all/"
INPUT_DIM, NUM_CLASSES = 31, 34


def make_fake_data():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(TMP + "X_train.npy", rng.random((3000, INPUT_DIM), dtype=np.float32))
    np.save(TMP + "y_train.npy", rng.integers(0, NUM_CLASSES, 3000))
    np.save(TMP + "X_test.npy", rng.random((600, INPUT_DIM), dtype=np.float32))
    np.save(TMP + "y_test.npy", rng.integers(0, NUM_CLASSES, 600))


def main():
    make_fake_data()
    ft.DATA_DIR = TMP
    ft.OUT_DIR = TMP
    ft.get_loader = lambda X, y, batch_size=256: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=batch_size, shuffle=True)

    X_tr, X_te, y_tr, y_te = ft.load_data()

    def run(model, rounds, resume):
        ft.set_seed(42)
        clients = ft.split_clients(X_tr, y_tr)
        return ft.federated_training(
            model, clients, X_te, y_te, INPUT_DIM, NUM_CLASSES,
            rounds=rounds, epochs=1, seed=42, resume=resume)

    print("=" * 62)
    print("  --resume check across all models")
    print("=" * 62)

    results = []
    for name in MODEL_REGISTRY:
        # fresh tmp per model so checkpoints from different models
        # can never collide or mask a bug in another model's run
        for f in os.listdir(TMP):
            if f.startswith(name) or f == f"{name}_seed42_checkpoint.pt":
                os.remove(os.path.join(TMP, f))

        _, h1 = run(name, 3, resume=False)
        _, h2 = run(name, 5, resume=True)
        _, h3 = run(name, 5, resume=True)

        rounds2 = [h["round"] for h in h2]
        continuous = rounds2 == [1, 2, 3, 4, 5]
        carried = ([h["accuracy"] for h in h2[:3]] ==
                   [h["accuracy"] for h in h1])
        not_retrained = len(h3) == 5

        ok = len(h1) == 3 and len(h2) == 5 and continuous and carried and not_retrained
        results.append((name, ok))
        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {name:12s}  "
              f"3-round={len(h1)} resumed-to-5={len(h2)} "
              f"continuous={continuous} carried-over={carried} "
              f"no-retrain={not_retrained}")

    shutil.rmtree(TMP, ignore_errors=True)

    print("\n" + "=" * 62)
    n = sum(1 for _, ok in results if ok)
    print(f"  {n}/{len(results)} models resume correctly")
    if n < len(results):
        print(f"  FAILED: {[name for name, ok in results if not ok]}")
    print("=" * 62)
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
