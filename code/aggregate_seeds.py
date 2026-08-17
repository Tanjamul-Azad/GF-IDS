"""
Aggregate repeated runs of the same model into mean and std.

Reads every {model}_seed{N}_history.pkl in --runs, groups them by
model name, and reports mean plus standard deviation of peak and
final accuracy across seeds. Prints a small table and writes
seed_summary.csv, which is what a "mean +/- std over N seeds" claim
in the paper should point to instead of a single run.

    python federated_train.py --model BNN --seed 42
    python federated_train.py --model BNN --seed 123
    python federated_train.py --model BNN --seed 2024
    python aggregate_seeds.py
"""

import argparse
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="./runs/")
    args = parser.parse_args()

    pattern = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)_history\.pkl$")
    groups = {}

    for path in glob.glob(os.path.join(args.runs, "*_history.pkl")):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        with open(path, "rb") as f:
            history = pickle.load(f)
        groups.setdefault(m["model"], []).append({
            "seed": int(m["seed"]),
            "peak": max(h["accuracy"] for h in history) * 100,
            "final": history[-1]["accuracy"] * 100,
            "peak_round": max(history, key=lambda h: h["accuracy"])["round"],
        })

    if not groups:
        print(f"No *_seed*_history.pkl files found in {args.runs}. "
              f"Run federated_train.py with --seed a few times first.")
        return

    rows = []
    for model, runs in sorted(groups.items()):
        peaks = np.array([r["peak"] for r in runs])
        finals = np.array([r["final"] for r in runs])
        rounds = np.array([r["peak_round"] for r in runs])
        rows.append({
            "Model": model,
            "N seeds": len(runs),
            "Seeds": sorted(r["seed"] for r in runs),
            "Peak Acc mean": round(peaks.mean(), 2),
            "Peak Acc std": round(peaks.std(), 2),
            "Final Acc mean": round(finals.mean(), 2),
            "Final Acc std": round(finals.std(), 2),
            "Peak Round mean": round(rounds.mean(), 1),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out = os.path.join(args.runs, "seed_summary.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    single_seed = df[df["N seeds"] == 1]["Model"].tolist()
    if single_seed:
        print(f"\nOnly one seed found for: {single_seed}. "
              f"A mean/std claim needs at least 3.")


if __name__ == "__main__":
    main()
