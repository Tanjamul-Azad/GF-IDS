"""
Federated training loop for GF-IDS.

Simulates K = 5 IID clients over T = 20 communication rounds.
Each round every client starts from the current global model,
trains locally for E = 5 epochs with Adam, and returns its
parameters; the server forms a sample-count-weighted average
(FedAvg) and redistributes the result.

Usage
    python federated_train.py --model BNN
    python federated_train.py --model MLP --rounds 20 --epochs 5

Reproduces the per-round accuracy tables reported in the paper.
"""

import argparse
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from binary_ops import binary_weight_keys, clip_all_binary_weights
from models import MODEL_REGISTRY
from quant_ops import fake_quant, quant_weight_keys

# ── Configuration (matches the reported experiments) ─────────
DATA_DIR = "./data/"
OUT_DIR = "./runs/"
NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
CLIENT_TEST_FRACTION = 0.2
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def set_seed(seed):
    """Seed every source of randomness the run depends on.

    Without this the client partition and the weight initialization
    differ on every run, so two models cannot be said to have been
    trained under the same conditions and no result can be reproduced.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_clients(X_train, y_train, num_clients=NUM_CLIENTS):
    """Partition the training set into IID shards, one per client.

    Each shard is further split into a local train/test portion;
    only the local train portion is used for federated updates.

    Seeded by set_seed(), so every model in a comparison receives the
    identical partition.
    """
    indices = np.random.permutation(len(X_train))
    X_shuf, y_shuf = X_train[indices], y_train[indices]
    csize = len(X_shuf) // num_clients

    clients = []
    for i in range(num_clients):
        X_c = X_shuf[i * csize:(i + 1) * csize]
        y_c = y_shuf[i * csize:(i + 1) * csize]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_c, y_c, test_size=CLIENT_TEST_FRACTION,
            random_state=SEED, stratify=y_c)
        clients.append({"X_train": X_tr, "X_test": X_te,
                        "y_train": y_tr, "y_test": y_te})
        print(f"Client {i + 1}: Train={X_tr.shape[0]:,}")
    return clients


def get_loader(X, y, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


def train_one_round(model, loader, optimizer, criterion, epochs):
    """Local training: E epochs of Adam on one client's shard.

    After each step the latent weights of any binary layer are
    clamped back into [-1, 1]. Without that clamp they drift outside
    the STE gradient window, the surrogate gradient goes to zero and
    those layers stop learning.
    """
    model.train()
    for _ in range(epochs):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            clip_all_binary_weights(model)


@torch.no_grad()
def evaluate(model, X, y, batch_size=1024):
    """Top-1 accuracy on the held-out global test set."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=batch_size)
    correct = total = 0
    for X_b, y_b in loader:
        preds = model(X_b.to(device)).argmax(dim=1).cpu()
        correct += (preds == y_b).sum().item()
        total += len(y_b)
    return correct / total


def fedavg(local_states, local_sizes, rebinarize_keys=None,
           requantize_keys=None):
    """Sample-count-weighted average, with optional re-binarization.

    Aggregation runs on the real-valued weights:

        W = sum_k (n_k / n) * W_k

    Averaging must happen in full precision. Averaging already-binary
    values would produce fractional results that are neither valid
    binary weights nor meaningful magnitudes.

    The averaged binary layers are then mapped back onto {-1, +1}:

        W_global = sign(W)

    so that what is distributed to the clients is a genuinely binary
    global model rather than a real-valued one.
    """
    total = sum(local_sizes)
    new_state = {}
    for key in local_states[0].keys():
        new_state[key] = sum(
            local_states[i][key] * (local_sizes[i] / total)
            for i in range(len(local_states))
        )

    if rebinarize_keys:
        for key in rebinarize_keys:
            if key in new_state:
                s = torch.sign(new_state[key])
                new_state[key] = torch.where(
                    s == 0, torch.ones_like(s), s)

    # The int8 counterpart: snap the averaged weights back onto the
    # quantization grid, so the distributed model is genuinely 8 bit
    # and the baseline is treated on the same terms as the proposal.
    if requantize_keys:
        for key, bits in requantize_keys.items():
            if key in new_state:
                new_state[key] = fake_quant(new_state[key], bits)

    return new_state


def federated_training(model_name, clients, X_test, y_test,
                       input_dim, num_classes,
                       rounds=ROUNDS, epochs=LOCAL_EPOCHS,
                       lr=LEARNING_RATE, rebinarize=True, seed=None):
    ModelClass = MODEL_REGISTRY[model_name]

    print(f"\n{'=' * 45}\n  Training: {model_name}\n{'=' * 45}")
    global_model = ModelClass(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    history = []

    # Only weight matrices are snapped back to their target precision
    # after aggregation; biases and BatchNorm parameters stay full
    # precision in every model.
    rebin_keys = binary_weight_keys(global_model) if rebinarize else None
    requant_keys = quant_weight_keys(global_model) if rebinarize else None
    if rebin_keys:
        print(f"  Re-binarizing after aggregation: {sorted(rebin_keys)}")
    if requant_keys:
        print(f"  Re-quantizing after aggregation: "
              f"{sorted(requant_keys)}")

    for t in range(rounds):
        local_states, local_sizes = [], []

        for client in clients:
            local_model = ModelClass(input_dim, num_classes).to(device)
            local_model.load_state_dict(
                copy.deepcopy(global_model.state_dict()))
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            loader = get_loader(client["X_train"], client["y_train"])
            train_one_round(local_model, loader, optimizer, criterion, epochs)
            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_sizes.append(len(client["X_train"]))

        global_model.load_state_dict(
            fedavg(local_states, local_sizes, rebin_keys, requant_keys))

        acc = evaluate(global_model, X_test, y_test)
        history.append({"round": t + 1, "accuracy": acc})
        print(f"  Round {t + 1:2d}/{rounds} - Accuracy: {acc:.4f}")

        if (t + 1) % 5 == 0:
            torch.save(global_model.state_dict(),
                       os.path.join(OUT_DIR, f"{model_name}_checkpoint.pt"))

    tag = f"{model_name}_seed{seed}" if seed is not None else model_name
    torch.save(global_model.state_dict(),
               os.path.join(OUT_DIR, f"{tag}_final.pt"))
    with open(os.path.join(OUT_DIR, f"{tag}_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    best = max(history, key=lambda h: h["accuracy"])
    print(f"\n  Best:  Round {best['round']} - {best['accuracy'] * 100:.2f}%")
    print(f"  Final: Round {rounds} - {history[-1]['accuracy'] * 100:.2f}%")
    return global_model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed for the client split and weight init; "
                             "vary it to obtain error bars")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    X_train, X_test, y_train, y_test = load_data()
    input_dim = X_train.shape[1]
    num_classes = int(y_train.max()) + 1
    print(f"Device: {device}, INPUT_DIM={input_dim}, "
          f"NUM_CLASSES={num_classes}")

    clients = split_clients(X_train, y_train)
    federated_training(args.model, clients, X_test, y_test,
                       input_dim, num_classes,
                       rounds=args.rounds, epochs=args.epochs, lr=args.lr,
                       seed=args.seed)


if __name__ == "__main__":
    main()
