"""Prepare UNSW-NB15 (official 82,332-record partition, 10 classes) for GF-IDS.

Same pipeline as the CICIoT2023 preprocessing: drop duplicates, label-encode
the three categorical columns, min-max scale (fit on train only), prune
features that correlate above |r| = 0.80, stratified 80/20 train/test split.
Output: ./data_unsw/{X,y}_{train,test}.npy and class_names.npy
"""
import numpy as np, pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SRC = r"F:\UIU\11th\green\datasets\UNSW-NB15\UNSW_NB15_testing-set.csv"
OUT = "./data_unsw/"
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(SRC).drop(columns=["id", "label"]).drop_duplicates()
y_names = sorted(d["attack_cat"].unique())
y = LabelEncoder().fit(y_names).transform(d["attack_cat"])
X = d.drop(columns=["attack_cat"]).copy()
for c in ("proto", "service", "state"):
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
sc = MinMaxScaler().fit(X_tr)
X_tr = pd.DataFrame(sc.transform(X_tr), columns=X.columns)
X_te = pd.DataFrame(np.clip(sc.transform(X_te), 0, 1), columns=X.columns)
corr = X_tr.corr().abs().fillna(0).values
keep = []
for i in range(corr.shape[0]):
    if all(corr[i, j] <= 0.80 for j in keep):
        keep.append(i)
cols = [X.columns[i] for i in keep]
print("features kept:", len(cols), "of", X.shape[1])
print("classes:", y_names)
np.save(OUT + "X_train.npy", X_tr[cols].values.astype(np.float32))
np.save(OUT + "X_test.npy", X_te[cols].values.astype(np.float32))
np.save(OUT + "y_train.npy", y_tr.astype(np.int64))
np.save(OUT + "y_test.npy", y_te.astype(np.int64))
np.save(OUT + "class_names.npy", np.array(y_names, dtype=object), allow_pickle=True)
print("train", X_tr.shape[0], "test", X_te.shape[0], "class counts", np.bincount(y_tr))
