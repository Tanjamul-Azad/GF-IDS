"""
Figure generation for the GF-IDS paper.

Produces:
  Fig. 2  per-class F1 comparison, MLP-FL vs BNN-FL
  Fig. 3  macro-averaged one-vs-rest ROC curves
  Fig. 4  round-by-round convergence for all four models
  Fig. 5  per-round communication overhead
  Fig. 6  FLOPs / energy proxy / parameter count dashboard
  Fig. 7  top-10 misclassification pairs for BNN-FL
  Fig. 8  accuracy vs communication cost (Pareto view)

Run `evaluate.py` and `federated_train.py` first; this script reads
the saved checkpoints and history pickles.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from models import MODEL_REGISTRY

DATA_DIR = "./data/"
RUN_DIR = "./runs/"
FIG_DIR = "./figures/"

COLORS = {"MLP": "#2196F3", "CNN": "#FF9800",
          "LSTM": "#4CAF50", "BNN": "#D32F2F"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _style_axes(ax):
    """Solid black frame, matching the figure style used in the paper."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def load_model(name, input_dim, num_classes):
    model = MODEL_REGISTRY[name](input_dim, num_classes).to(device)
    model.load_state_dict(
        torch.load(os.path.join(RUN_DIR, f"{name}_final.pt"),
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


def fig_perclass_f1(f1_mlp, f1_bnn, class_names):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(class_names))
    width = 0.4
    ax.bar(x - width / 2, f1_mlp, width, label="MLP-FL",
           color=COLORS["MLP"], edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, f1_bnn, width, label="BNN-FL (Proposed)",
           color=COLORS["BNN"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1-Score: MLP-FL vs BNN-FL on CICIoT2023",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, edgecolor="black")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_perclass_f1.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_roc(y_true, probs_mlp, probs_bnn, num_classes):
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(6, 5))
    for probs, name in ((probs_mlp, "MLP-FL"), (probs_bnn, "BNN-FL")):
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.4f})",
                color=COLORS["MLP" if name == "MLP-FL" else "BNN"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Averaged ROC Curve (One-vs-Rest)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", edgecolor="black")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_roc.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_convergence(histories):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        rounds = [h["round"] for h in hist]
        accs = [h["accuracy"] * 100 for h in hist]
        ax.plot(rounds, accs, marker="o", markersize=4,
                label=f"{name}-FL", color=COLORS[name])
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Federated Learning Convergence - CICIoT2023",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 21))
    ax.legend(edgecolor="black")
    ax.grid(alpha=0.25, linestyle="--")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_convergence.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_communication(bytes_per_round):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(bytes_per_round.keys())
    values = [bytes_per_round[n] for n in names]
    bars = ax.bar(names, values, color=[COLORS[n] for n in names],
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.1f} KB", ha="center", fontsize=9)
    ax.set_ylabel("Bytes per Round (KB)", fontsize=11)
    ax.set_title("Communication Overhead per Federated Round",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_communication.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_green_dashboard(flops, energy_nj, params):
    """Three-panel efficiency dashboard.

    energy_nj is the FLOP-count energy proxy: FLOPs x 0.5 pJ/FLOP,
    converted to nJ (1 nJ = 1000 pJ).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [(flops, "FLOPs (Millions)", "FLOPs per Inference"),
              (energy_nj, "Energy Proxy (nJ)", "Energy Proxy per Inference"),
              (params, "Trainable Parameters", "Model Size")]
    for ax, (data, ylabel, title) in zip(axes, panels):
        names = list(data.keys())
        values = [data[n] for n in names]
        bars = ax.bar(names, values, color=[COLORS[n] for n in names],
                      edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, values):
            label = f"{val:,.0f}" if val >= 100 else f"{val:g}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        _style_axes(ax)
    fig.suptitle("Computational and Model-Size Efficiency Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_green_dashboard.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_misclassification(preds, labels, class_names, top_n=10):
    cm = confusion_matrix(labels, preds,
                          labels=np.arange(len(class_names)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_norm[i, j] >= 0.10:
                pairs.append((class_names[i], class_names[j],
                              cm_norm[i, j], int(cm[i].sum())))
    pairs = sorted(pairs, key=lambda p: -p[2])[:top_n]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels_list = [f"{p[0]} -> {p[1]}" for p in pairs]
    values = [p[2] for p in pairs]
    supports = [p[3] for p in pairs]
    colors = plt.cm.Reds(np.linspace(0.45, 0.85, len(values)))
    y_pos = np.arange(len(labels_list))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.6)
    for bar, val, sup in zip(bars, values, supports):
        ax.text(bar.get_width() + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}  (n={sup:,})", va="center", fontsize=8.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_list, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassification Rate", fontsize=11)
    ax.set_title("Top Misclassification Pairs - BNN-FL on CICIoT2023",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, max(values) * 1.35)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_misclassification.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def fig_pareto(accuracy, bytes_per_round):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    markers = {"MLP": "o", "CNN": "s", "LSTM": "^", "BNN": "*"}
    for name in accuracy:
        ax.scatter(bytes_per_round[name], accuracy[name],
                   s=250 if name == "BNN" else 110,
                   marker=markers[name], color=COLORS[name],
                   edgecolor="black", linewidth=0.7,
                   label=f"{name}-FL", zorder=3)
    ax.set_xlabel("Bytes per Round (KB) - Lower is Better", fontsize=11)
    ax.set_ylabel("Final Accuracy (%) - Higher is Better", fontsize=11)
    ax.set_title("Accuracy vs Communication Cost (Pareto Frontier)",
                 fontsize=12, fontweight="bold")
    ax.legend(edgecolor="black")
    ax.grid(alpha=0.25, linestyle="--")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_pareto.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    input_dim = X_test.shape[1]
    num_classes = int(y_test.max()) + 1

    names_path = os.path.join(DATA_DIR, "class_names.npy")
    class_names = (list(np.load(names_path, allow_pickle=True))
                   if os.path.exists(names_path)
                   else [str(i) for i in range(num_classes)])

    mlp = load_model("MLP", input_dim, num_classes)
    bnn = load_model("BNN", input_dim, num_classes)

    preds_mlp, labels, probs_mlp = predict(mlp, X_test, y_test, True)
    preds_bnn, _, probs_bnn = predict(bnn, X_test, y_test, True)

    f1_mlp = f1_score(labels, preds_mlp, average=None,
                      labels=np.arange(num_classes), zero_division=0)
    f1_bnn = f1_score(labels, preds_bnn, average=None,
                      labels=np.arange(num_classes), zero_division=0)

    fig_perclass_f1(f1_mlp, f1_bnn, class_names)
    fig_roc(labels, probs_mlp, probs_bnn, num_classes)
    fig_misclassification(preds_bnn, labels, class_names)

    histories = {}
    for name in MODEL_REGISTRY:
        path = os.path.join(RUN_DIR, f"{name}_history.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                histories[name] = pickle.load(f)
    if histories:
        fig_convergence(histories)

    # Efficiency panels read from the results table written by
    # evaluate.py, so that figures and tables cannot drift apart.
    results_csv = os.path.join(RUN_DIR, "results.csv")
    if os.path.exists(results_csv):
        import pandas as pd
        df = pd.read_csv(results_csv).set_index("Model")
        flops = df["FLOPs(M)"].to_dict()
        params = df["Parameters"].to_dict()
        payload = df["IdealPayload(KB)"].to_dict()
        accuracy = df["Accuracy(%)"].to_dict()
        # 0.5 pJ per FLOP, expressed in nJ
        energy = {k: (v * 1e6) * 0.5 / 1000 for k, v in flops.items()}
        fig_communication(payload)
        fig_green_dashboard(flops, energy, params)
        fig_pareto(accuracy, payload)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
