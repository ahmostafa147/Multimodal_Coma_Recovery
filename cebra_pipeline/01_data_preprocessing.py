"""
01_data_preprocessing.py — uniform 5-min resolution.

Combinations (set RUN_TAG + FEATURE_KEYS accordingly; 02 picks LABEL_KEYS):
  i)   FEATURE_KEYS=['features']
  ii)  FEATURE_KEYS=['features']
  iii) FEATURE_KEYS=['features', 'cebra_features']
  iv)  FEATURE_KEYS=['features', 'cebra_features']
  v)   FEATURE_KEYS=['features', 'cebra_features']

All label arrays (predictions, probabilities, cpc_scores) are saved
unconditionally; 02 selects which to use as objectives.
"""
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _constants import DATA_DIR, OUT_DIR

# ── CONFIG ──────────────────────────────────────────────────────────
RUN_TAG        = 'combo_i'                        # change per run
os.makedirs(OUT_DIR, exist_ok=True)
FEATURE_KEYS   = ['features']                     # 'features' (PPNet) +/- 'cebra_features' (EEG)
PCA_KEY        = 'features'                       # only this block gets PCA'd; others scaled raw
PCA_COMPONENTS = 50
SEED           = 42
BIN_SEC        = 300                              # uniform 5-min resolution

np.random.seed(SEED)

# ── Load ────────────────────────────────────────────────────────────
print(f"=== Loading (RUN_TAG={RUN_TAG}) ===")
train_data = np.load(f'{DATA_DIR}/PPNet_data_train.npz', allow_pickle=True)
test_data  = np.load(f'{DATA_DIR}/PPNet_data_test.npz',  allow_pickle=True)

for name, d in [("TRAIN", train_data), ("TEST", test_data)]:
    print(f"\n{name}:")
    for k in d.keys():
        print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")
    print(f"  predictions unique: {np.unique(d['predictions'])}")
    print(f"  patients: {len(np.unique(d['patient_ids']))}")

# ── Sort by (patient_id, time) → contiguous patients + temporal order
print("\n=== Sorting by (patient_id, time) ===")
def get_order(d):
    return np.lexsort((d['times'].astype(float), d['patient_ids']))

train_order = get_order(train_data)
test_order  = get_order(test_data)

def take(d, k, order):
    return d[k][order]

# ── Build X by concatenating selected feature blocks ────────────────
print(f"\n=== Building X from {FEATURE_KEYS} ===")
blocks_train, blocks_test = [], []
pca = None
scalers = {}

def sanitize(A_train, A_test, key):
    """Replace inf/NaN in a feature block with the train column median."""
    A_train = np.asarray(A_train, dtype=np.float64)
    A_test  = np.asarray(A_test,  dtype=np.float64)
    bad_tr = ~np.isfinite(A_train)
    bad_te = ~np.isfinite(A_test)
    n_tr, n_te = int(bad_tr.sum()), int(bad_te.sum())
    if n_tr == 0 and n_te == 0:
        return A_train.astype(np.float32), A_test.astype(np.float32)
    finite = np.where(np.isfinite(A_train), A_train, np.nan)
    med = np.nanmedian(finite, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)        # all-NaN columns -> 0
    if n_tr:
        rows, cols = np.where(bad_tr)
        A_train[rows, cols] = med[cols]
    if n_te:
        rows, cols = np.where(bad_te)
        A_test[rows, cols] = med[cols]
    print(f"    sanitized {key}: replaced {n_tr} train + {n_te} test non-finite cells "
          f"with train column median")
    return A_train.astype(np.float32), A_test.astype(np.float32)

for key in FEATURE_KEYS:
    A = take(train_data, key, train_order).astype(np.float32)
    B = take(test_data,  key, test_order).astype(np.float32)
    A, B = sanitize(A, B, key)

    if key == PCA_KEY and PCA_COMPONENTS is not None:
        print(f"  PCA on {key}: {A.shape[1]} -> {PCA_COMPONENTS}")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        A = pca.fit_transform(A)
        B = pca.transform(B)
        print(f"    explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    sc = StandardScaler()
    A = sc.fit_transform(A).astype(np.float32)
    B = sc.transform(B).astype(np.float32)
    scalers[key] = sc

    print(f"  {key} -> block shape: {A.shape}")
    blocks_train.append(A)
    blocks_test.append(B)

X_train = np.concatenate(blocks_train, axis=1) if len(blocks_train) > 1 else blocks_train[0]
X_test  = np.concatenate(blocks_test,  axis=1) if len(blocks_test)  > 1 else blocks_test[0]
print(f"\n  X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── Patient boundaries (contiguity guaranteed by lexsort) ───────────
print("\n=== Patient boundaries ===")
def compute_boundaries(pids):
    unique_pids, inverse = np.unique(pids, return_inverse=True)
    N = len(pids)
    starts = np.zeros(N, dtype=np.int64)
    ends   = np.zeros(N, dtype=np.int64)
    for pid_idx in range(len(unique_pids)):
        idxs = np.where(inverse == pid_idx)[0]
        starts[idxs] = idxs[0]
        ends[idxs]   = idxs[-1] + 1
    return starts, ends

pid_train = take(train_data, 'patient_ids', train_order)
pid_test  = take(test_data,  'patient_ids', test_order)
pat_starts_train, pat_ends_train = compute_boundaries(pid_train)
pat_starts_test,  pat_ends_test  = compute_boundaries(pid_test)
print(f"  Train: {len(np.unique(pid_train))} patients, {len(pid_train)} bins")
print(f"  Test:  {len(np.unique(pid_test))} patients, {len(pid_test)} bins")

# ── Save ────────────────────────────────────────────────────────────
print("\n=== Saving ===")
def save_split(path, X, d, order, pat_starts, pat_ends):
    cpc = take(d, 'cpc_scores', order).astype(np.float32)
    cpc_binary = (cpc >= 3).astype(np.int64)   # 0 = good (CPC<=2), 1 = poor (CPC>=3)
    np.savez(
        path,
        X             = X,
        predictions   = take(d, 'predictions',   order),
        probabilities = take(d, 'probabilities', order),
        cpc_scores    = cpc,
        cpc_binary    = cpc_binary,
        activations   = take(d, 'activations',   order),  # raw PPNet prototype activations (for 06)
        patient_ids   = take(d, 'patient_ids',   order),
        chunks        = take(d, 'chunks',        order),
        times         = take(d, 'times',         order),
        bin_durations = np.full(len(order), BIN_SEC, dtype=np.float32),
        pat_starts    = pat_starts,
        pat_ends      = pat_ends,
    )

save_split(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_train.npz',
           X_train, train_data, train_order, pat_starts_train, pat_ends_train)
save_split(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_test.npz',
           X_test,  test_data,  test_order,  pat_starts_test,  pat_ends_test)

with open(f'{OUT_DIR}/cebra_pca_scaler_{RUN_TAG}.pkl', 'wb') as f:
    pickle.dump({
        'pca':            pca,
        'scalers':        scalers,
        'feature_keys':   FEATURE_KEYS,
        'pca_key':        PCA_KEY,
        'pca_components': PCA_COMPONENTS,
        'bin_sec':        BIN_SEC,
    }, f)

print(f"Done. cebra_prep_{RUN_TAG}_{{train,test}}.npz, cebra_pca_scaler_{RUN_TAG}.pkl")
