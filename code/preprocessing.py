"""
CICIoT2023 preprocessing pipeline for GF-IDS.

Steps
  1. Load all train/test CSV shards
  2. Remove infinite values, NaNs and duplicate records
  3. Drop redundant features by Pearson correlation
  4. Min-Max scale every remaining feature to [0, 1]
  5. Label-encode the 34 class strings to integers [0, 33]
  6. Persist the arrays as .npy so training can restart cheaply

Running this on the dataset used for the paper yields
INPUT_DIM = 31 and NUM_CLASSES = 34.

Note on the correlation threshold: several values were explored
(0.90 / 0.85 / 0.80 / 0.78 / 0.76). CORR_THRESHOLD below is the
value used to generate the arrays behind the reported results.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── Configuration ────────────────────────────────────────────
DATA_ROOT = "./ciciot2023/CICIOT23/"          # extracted dataset
OUT_DIR = "./data/"                            # where .npy files land
CORR_THRESHOLD = 0.80                          # Pearson |r| pruning threshold


def load_folder(folder_path):
    """Concatenate every CSV shard in a directory into one DataFrame."""
    all_dfs = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            all_dfs.append(df)
            print(f"  Loaded: {file} -> {df.shape}")
    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return pd.concat(all_dfs, ignore_index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load ──
    print("Loading train...")
    train_df = load_folder(os.path.join(DATA_ROOT, "train/"))
    print(f"Train total: {train_df.shape}")

    print("\nLoading test...")
    test_df = load_folder(os.path.join(DATA_ROOT, "test/"))
    print(f"Test total: {test_df.shape}")

    # ── 2. Clean ──
    # Infinite/missing rows and exact duplicates are removed so that
    # repeated records cannot bias local gradient updates.
    for df in (train_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
    print(f"\nAfter cleaning - Train: {train_df.shape}, Test: {test_df.shape}")

    # ── 3. Feature / label split ──
    X_train = train_df.drop(columns=["label"]).select_dtypes(include="number")
    X_test = test_df.drop(columns=["label"]).select_dtypes(include="number")
    print(f"\nFeatures before selection: {X_train.shape[1]}")

    # ── 4. Pearson correlation pruning ──
    # Only the upper triangle is inspected so each correlated pair
    # is considered once and just one member of the pair is dropped.
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    print(f"Dropping {len(to_drop)} correlated features "
          f"(|r| > {CORR_THRESHOLD}): {to_drop}")

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)
    print(f"Features after selection: {X_train.shape[1]}")

    # ── 5. Min-Max scaling ──
    # Fitted on train only, then applied to test, to avoid leakage.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Min: {X_train_scaled.min():.2f}, Max: {X_train_scaled.max():.2f}")

    # ── 6. Label encoding ──
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df["label"])
    y_test_enc = le.transform(test_df["label"])
    print(f"Classes: {len(le.classes_)}")

    # ── 7. Persist ──
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train_enc)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test_enc)
    np.save(os.path.join(OUT_DIR, "class_names.npy"), le.classes_)

    print(f"\nINPUT_DIM={X_train_scaled.shape[1]}, "
          f"NUM_CLASSES={len(le.classes_)}")
    print(f"X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
    print(f"Selected features: {X_train.columns.tolist()}")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
