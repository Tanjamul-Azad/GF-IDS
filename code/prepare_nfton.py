"""Prepare NF-ToN-IoT-v2 (NetFlow features of the ToN-IoT IoT testbed) for GF-IDS.

10 classes (Benign + 9 attacks). Official train/test files are randomly
sampled (natural class balance kept): 1,000,000 flows for training and
250,000 for the global test set. IP addresses and the ephemeral source
port are dropped (they identify hosts, not behaviour). Then: drop
duplicates, min-max scale (fit on train), prune |r| > 0.80 features.
Output: ./data_nfton/ (same layout as ./data/).
"""
import numpy as np, pandas as pd, os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
D = r"F:\UIU\11th\green\datasets\NF-ToN-IoT-v2" + "\\"
OUT = "./data_nfton/"; os.makedirs(OUT, exist_ok=True)
DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "Label"]

def sample(path, n_target, seed):
    rng = np.random.RandomState(seed); parts = []; total = 0
    for ch in pd.read_csv(path, chunksize=500_000):
        total += len(ch); parts.append(ch)
        if len(parts) % 1 == 0: parts[-1] = ch.sample(frac=0.12, random_state=rng.randint(1e9))
    df = pd.concat(parts).drop(columns=DROP).drop_duplicates()
    print(path.split("\\")[-1], "rows scanned", total, "kept", len(df))
    return df.sample(n=min(n_target, len(df)), random_state=seed)

tr = sample(D + "NF-ToN-IoT-v2-train.csv", 1_000_000, 42)
te = sample(D + "NF-ToN-IoT-v2-test.csv", 250_000, 43)
enc = LabelEncoder().fit(tr["Attack"]); names = list(enc.classes_)
y_tr, y_te = enc.transform(tr["Attack"]), enc.transform(te["Attack"])
X_tr, X_te = tr.drop(columns=["Attack"]).astype(np.float64), te.drop(columns=["Attack"]).astype(np.float64)
X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0); X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0)
sc = MinMaxScaler().fit(X_tr)
A = pd.DataFrame(sc.transform(X_tr), columns=X_tr.columns)
B = pd.DataFrame(np.clip(sc.transform(X_te), 0, 1), columns=X_tr.columns)
corr = A.sample(min(len(A), 200_000), random_state=0).corr().abs().fillna(0).values
keep = []
for i in range(corr.shape[0]):
    if all(corr[i, j] <= 0.80 for j in keep): keep.append(i)
cols = [A.columns[i] for i in keep]
print("features kept:", len(cols), "of", A.shape[1], cols)
print("classes:", names)
np.save(OUT + "X_train.npy", A[cols].values.astype(np.float32)); np.save(OUT + "X_test.npy", B[cols].values.astype(np.float32))
np.save(OUT + "y_train.npy", y_tr.astype(np.int64)); np.save(OUT + "y_test.npy", y_te.astype(np.int64))
np.save(OUT + "class_names.npy", np.array(names, dtype=object), allow_pickle=True)
print("train", len(A), "test", len(B), "train class counts", np.bincount(y_tr), "test", np.bincount(y_te))
