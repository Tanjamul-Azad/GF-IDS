"""Accuracy per round for BNN-FL under each aggregation rule (CICIoT2023, seed 42).

Reads the per-round histories written by federated_train.py and draws
fig_agg.pdf for Section V (Comparison Against Alternative Aggregation Rules).
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import figures as F  # noqa: E402  (shared style helpers)

RUNS = [
    ("", "FedAvg + re-binarize (ours)", "#d95f02", "-", "o"),
    ("_fedprox", r"FedProx ($\mu$=0.01)", "#1f77b4", "--", "s"),
    ("_signsgdlat", "SignSGD vote, step 0.01", "#7570b3", ":", "^"),
    ("_signsgdlat5", "SignSGD vote, step 0.05", "#e7298a", ":", "v"),
    ("_signsgdlat1", "SignSGD vote, step 0.001", "#66a61e", "-.", "d"),
]

fig, ax = plt.subplots(figsize=(F.COL_SINGLE, 2.7))
for tag, label, color, ls, marker in RUNS:
    path = os.path.join("runs", f"BNN-MATCHED_seed42{tag}_history.pkl")
    with open(path, "rb") as f:
        hist = pickle.load(f)
    acc = [h["accuracy"] * 100 for h in hist]
    ax.plot(range(1, len(acc) + 1), acc, color=color, linestyle=ls,
            marker=marker, markersize=2.5, markevery=4, linewidth=1.1,
            label=label)
ax.set_xlabel("Communication round")
ax.set_ylabel("Global test accuracy (%)")
ax.set_ylim(0, 102)
ax.legend(fontsize=5.8, loc="center right", bbox_to_anchor=(1.0, 0.53), edgecolor="black", framealpha=1.0)
ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
F._style_axes(ax)
fig.tight_layout()
F._save(fig, "fig_agg")
