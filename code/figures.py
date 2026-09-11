"""
Figure generation for the GF-IDS paper.

Every figure is written as a vector PDF sized for the IEEEtran column
grid: 3.5 in for a single column, 7.16 in for a double-column float.

Colour and accessibility
------------------------
Colours come from the Okabe-Ito palette, which stays distinguishable
under the common forms of colour vision deficiency. Every model also
carries its own marker and line style, so the figures still read
correctly when the paper is printed in greyscale. A model keeps the
same colour, marker and line style in every figure it appears in.

Run federated_train.py for each model first, then evaluate.py, then
this script. It reads the saved checkpoints, the history pickles and
the metrics table those two produce.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from federated_train import run_tag
from models import MODEL_REGISTRY

DATA_DIR = "./data/"
RUN_DIR = "./runs/"
FIG_DIR = "./figures/"
SEED = 42

COL_SINGLE = 3.5
COL_DOUBLE = 7.16

# Okabe-Ito, colour-vision-deficiency safe.
STYLE = {
    "BNN":        {"c": "#D55E00", "m": "o", "ls": "-"},
    "MLP":        {"c": "#0072B2", "m": "s", "ls": "--"},
    "CNN":        {"c": "#E69F00", "m": "^", "ls": "-."},
    "LSTM":       {"c": "#009E73", "m": "D", "ls": ":"},
    "MLP-INT8":   {"c": "#CC79A7", "m": "v", "ls": "--"},
    "BNN-INT8IO": {"c": "#56B4E9", "m": "P", "ls": "-"},
}
ORDER = ["BNN", "BNN-INT8IO", "MLP", "MLP-INT8", "CNN", "LSTM"]
LABEL = {m: ("BNN-FL (proposed)" if m == "BNN" else f"{m}-FL") for m in STYLE}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path, format="pdf", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def load_model(name, input_dim, num_classes, seed=SEED, suffix="best"):
    """Load one trained model.

    Checkpoints are named by federated_train.run_tag(), so the seed has
    to be part of the filename or nothing matches.
    """
    tag = run_tag(name, False, seed)
    model = MODEL_REGISTRY[name](input_dim, num_classes).to(device)
    model.load_state_dict(
        torch.load(os.path.join(RUN_DIR, f"{tag}_{suffix}.pt"),
                   map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, y, return_probs=False):
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=1024)
    preds, labels, probs = [], [], []
    for X_b, y_b in loader:
        out = model(X_b.to(device))
        preds.append(out.argmax(dim=1).cpu().numpy())
        labels.append(y_b.numpy())
        if return_probs:
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if return_probs:
        return preds, labels, np.concatenate(probs)
    return preds, labels


def fig_convergence(histories):
    """Accuracy and loss against communication round, all models."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))
    for name in ORDER:
        if name not in histories:
            continue
        h = histories[name]
        s = STYLE[name]
        rounds = [x["round"] for x in h]
        axes[0].plot(rounds, [x["accuracy"] * 100 for x in h],
                     color=s["c"], marker=s["m"], linestyle=s["ls"],
                     markersize=2.6, linewidth=1.0, markevery=3,
                     label=LABEL[name])
        losses = [x.get("loss") for x in h]
        if all(v is not None for v in losses):
            axes[1].plot(rounds, losses, color=s["c"], marker=s["m"],
                         linestyle=s["ls"], markersize=2.6, linewidth=1.0,
                         markevery=3, label=LABEL[name])

    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Global test accuracy (%)")
    axes[0].set_title("(a) Accuracy")
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Global test loss")
    axes[1].set_title("(b) Loss")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        _style_axes(ax)
    axes[0].legend(ncol=2, edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    _save(fig, "fig_convergence")


def fig_perclass_f1(f1_a, f1_b, class_names, name_a="MLP", name_b="BNN"):
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.9))
    x = np.arange(len(class_names))
    w = 0.4
    ax.bar(x - w / 2, f1_a, w, label=LABEL[name_a],
           color=STYLE[name_a]["c"], edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, f1_b, w, label=LABEL[name_b],
           color=STYLE[name_b]["c"], edgecolor="black", linewidth=0.3,
           hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=5.5)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend(edgecolor="black", framealpha=1.0, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_perclass_f1")


def fig_roc(y_true, probs_a, probs_b, num_classes, name_a="MLP", name_b="BNN"):
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    for probs, name in ((probs_a, name_a), (probs_b, name_b)):
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        ax.plot(fpr, tpr, color=STYLE[name]["c"],
                linestyle=STYLE[name]["ls"], linewidth=1.2,
                label=f"{LABEL[name]} (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", edgecolor="black", framealpha=1.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_roc")


def fig_communication(payload):
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.5))
    names = [m for m in ORDER if m in payload]
    values = [payload[m] for m in names]
    bars = ax.bar(range(len(names)), values,
                  color=[STYLE[m]["c"] for m in names],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABEL[m].replace(" (proposed)", "") for m in names],
                       rotation=30, ha="right")
    ax.set_ylabel("Uplink payload per round (KB)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_communication")


def fig_efficiency_dashboard(flops, iops, bops, payload, params):
    """Three measured efficiency views: operations, payload, model size."""
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.5))
    names = [m for m in ORDER if m in params]
    idx = np.arange(len(names))
    colors = [STYLE[m]["c"] for m in names]
    short = [LABEL[m].replace(" (proposed)", "").replace("-FL", "")
             for m in names]

    # (a) operation mix, stacked by precision
    f = np.array([flops.get(m, 0) * 1e6 for m in names])
    i = np.array([iops.get(m, 0) * 1e6 for m in names])
    b = np.array([bops.get(m, 0) * 1e6 for m in names])
    axes[0].bar(idx, f, color="#555555", edgecolor="black", linewidth=0.4,
                label="Float MAC")
    axes[0].bar(idx, i, bottom=f, color="#AAAAAA", edgecolor="black",
                linewidth=0.4, label="Int8 MAC", hatch="//")
    axes[0].bar(idx, b, bottom=f + i, color="#FFFFFF", edgecolor="black",
                linewidth=0.4, label="Binary op", hatch="xx")
    axes[0].set_ylabel("Operations per inference")
    axes[0].set_title("(a) Operation mix")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=6, edgecolor="black", framealpha=1.0)

    axes[1].bar(idx, [payload[m] for m in names], color=colors,
                edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Payload per round (KB)")
    axes[1].set_title("(b) Communication")

    axes[2].bar(idx, [params[m] for m in names], color=colors,
                edgecolor="black", linewidth=0.4)
    axes[2].set_ylabel("Trainable parameters")
    axes[2].set_title("(c) Model size")

    for ax in axes:
        ax.set_xticks(idx)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_efficiency")


def fig_misclassification(preds, labels, class_names, top_n=10):
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_norm[i, j] >= 0.10:
                pairs.append((class_names[i], class_names[j],
                              cm_norm[i, j], int(cm[i].sum())))
    pairs = sorted(pairs, key=lambda p: -p[2])[:top_n]
    if not pairs:
        print("  no misclassification pair above the 0.10 threshold")
        return

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.9))
    names = [f"{p[0]} $\\rightarrow$ {p[1]}" for p in pairs]
    values = [p[2] for p in pairs]
    supports = [p[3] for p in pairs]
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=STYLE["BNN"]["c"], edgecolor="black",
                   linewidth=0.4, height=0.65)
    for bar, val, sup in zip(bars, values, supports):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} (n={sup:,})", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassification rate")
    ax.set_xlim(0, max(values) * 1.45)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_misclassification")


def fig_pareto(accuracy, payload):
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    for m in ORDER:
        if m not in accuracy or m not in payload:
            continue
        s = STYLE[m]
        ax.scatter(payload[m], accuracy[m], s=90 if m == "BNN" else 45,
                   marker=s["m"], color=s["c"], edgecolor="black",
                   linewidth=0.6, zorder=3, label=LABEL[m])
        ax.annotate(LABEL[m].replace(" (proposed)", "").replace("-FL", ""),
                    (payload[m], accuracy[m]), textcoords="offset points",
                    xytext=(5, 4), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Uplink payload per round (KB, log scale)")
    ax.set_ylabel("Best accuracy (%)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_pareto")


def fig_precision_tiers(accuracy, payload):
    """What each precision choice costs and buys, per architecture."""
    pairs = [("MLP", "MLP-INT8"), ("BNN", "BNN-INT8IO")]
    pairs = [(a, b) for a, b in pairs if a in accuracy and b in accuracy]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.4))
    idx = np.arange(len(pairs) * 2)
    names = [m for p in pairs for m in p]
    colors = [STYLE[m]["c"] for m in names]
    short = [LABEL[m].replace(" (proposed)", "").replace("-FL", "")
             for m in names]

    axes[0].bar(idx, [accuracy[m] for m in names], color=colors,
                edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("Best accuracy (%)")
    axes[0].set_title("(a) Detection accuracy")
    axes[0].set_ylim(0, 105)
    for k, m in enumerate(names):
        axes[0].text(k, accuracy[m] + 1.5, f"{accuracy[m]:.2f}",
                     ha="center", fontsize=6)

    axes[1].bar(idx, [payload[m] for m in names], color=colors,
                edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Payload per round (KB)")
    axes[1].set_title("(b) Communication cost")
    for k, m in enumerate(names):
        axes[1].text(k, payload[m] + 0.8, f"{payload[m]:.2f}",
                     ha="center", fontsize=6)

    for ax in axes:
        ax.set_xticks(idx)
        ax.set_xticklabels(short, rotation=25, ha="right", fontsize=6.5)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
        _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_precision_tiers")


def fig_class_distribution(y_test, class_names):
    counts = np.bincount(y_test, minlength=len(class_names))
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.5))
    idx = np.arange(len(class_names))
    colors = ["#999999" if counts[c] >= 150 else STYLE["BNN"]["c"]
              for c in order]
    ax.bar(idx, counts[order], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([class_names[c] for c in order], rotation=90,
                       fontsize=5.5)
    ax.set_ylabel("Test samples (log scale)")
    ax.axhline(150, color="black", linestyle=":", linewidth=0.8)
    ax.text(len(class_names) - 0.5, 165, "150 samples", ha="right",
            fontsize=6)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_class_distribution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--suffix", default="best", choices=["final", "best"])
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    input_dim = X_test.shape[1]
    num_classes = int(y_test.max()) + 1

    names_path = os.path.join(DATA_DIR, "class_names.npy")
    class_names = (list(np.load(names_path, allow_pickle=True))
                   if os.path.exists(names_path)
                   else [str(i) for i in range(num_classes)])

    histories = {}
    for name in MODEL_REGISTRY:
        path = os.path.join(RUN_DIR,
                            f"{run_tag(name, False, args.seed)}_history.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                histories[name] = pickle.load(f)
    if histories:
        fig_convergence(histories)

    fig_class_distribution(y_test, class_names)

    mlp = load_model("MLP", input_dim, num_classes, args.seed, args.suffix)
    bnn = load_model("BNN", input_dim, num_classes, args.seed, args.suffix)
    preds_mlp, labels, probs_mlp = predict(mlp, X_test, y_test, True)
    preds_bnn, _, probs_bnn = predict(bnn, X_test, y_test, True)

    f1_mlp = f1_score(labels, preds_mlp, average=None,
                      labels=np.arange(num_classes), zero_division=0)
    f1_bnn = f1_score(labels, preds_bnn, average=None,
                      labels=np.arange(num_classes), zero_division=0)
    fig_perclass_f1(f1_mlp, f1_bnn, class_names)
    fig_roc(labels, probs_mlp, probs_bnn, num_classes)
    fig_misclassification(preds_bnn, labels, class_names)

    # Efficiency panels read the table evaluate.py wrote, so the figures
    # and the tables cannot drift apart.
    results_csv = os.path.join(RUN_DIR, f"results_{args.suffix}.csv")
    if os.path.exists(results_csv):
        import pandas as pd
        df = pd.read_csv(results_csv).set_index("Model")
        payload = df["PackedPayload(KB)"].to_dict()
        accuracy = df["Accuracy(%)"].to_dict()
        fig_communication(payload)
        fig_efficiency_dashboard(df["FLOPs(M)"].to_dict(),
                                 df["IOPs(M)"].to_dict(),
                                 df["BOPs(M)"].to_dict(),
                                 payload, df["Parameters"].to_dict())
        fig_pareto(accuracy, payload)
        fig_precision_tiers(accuracy, payload)
    else:
        print(f"  {results_csv} not found, skipping efficiency figures")

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
