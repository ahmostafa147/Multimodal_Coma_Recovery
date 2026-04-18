import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

DATA_DIR = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model'
BIN_DURATION = 300  # 5 minutes in seconds
DROP_PARTIAL = False  # True = drop last bin if < BIN_DURATION; False = keep it
SEED = 42

# ═══════════════════════════════════════════════════════════════════
# A1: Load and inspect
# ═══════════════════════════════════════════════════════════════════
print("=== A1: Loading data ===")
train_data = np.load(f'{DATA_DIR}/PPNet_data_train.npz', allow_pickle=True)
test_data  = np.load(f'{DATA_DIR}/PPNet_data_test.npz',  allow_pickle=True)

for name, d in [("TRAIN", train_data), ("TEST", test_data)]:
    print(f"\n{name}:")
    for k in d.keys():
        arr = d[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  labels unique: {np.unique(d['predictions'])}")
    print(f"  patients unique: {len(np.unique(d['patient_ids']))}")
    print(f"  resolutions unique: {np.unique(d['resolutions'])}")
    pids = d['patient_ids']
    breaks = np.where(pids[1:] != pids[:-1])[0] + 1
    n_blocks = len(breaks) + 1
    n_unique = len(np.unique(pids))
    if n_blocks != n_unique:
        print(f"  WARNING: patients NOT contiguous! {n_blocks} blocks vs {n_unique} patients")
    else:
        print(f"  Patients contiguous: {n_unique} patients, {n_blocks} blocks")

# ═══════════════════════════════════════════════════════════════════
# A2: Resample to uniform 5-min bins per (patient, chunk)
# ═══════════════════════════════════════════════════════════════════
print("\n=== A2: Resampling to 5-min resolution ===")

def resample_to_bins(features, predictions, activations, patient_ids, chunks,
                     resolutions, times, cpc_scores,
                     bin_duration=BIN_DURATION, drop_partial=DROP_PARTIAL):
    """Aggregate rows into fixed-duration bins per (patient, chunk).

    Bins are exactly bin_duration seconds, except possibly the last bin per
    segment which is <= bin_duration (kept or dropped based on drop_partial).

    - features & activations: duration-weighted average
    - label: duration-weighted majority vote
    - time: midpoint of bin
    - cpc_score: carried from patient
    - chunk: kept from source chunk
    - bin_duration_actual: true duration of each bin
    """
    out = {k: [] for k in ['features', 'activations', 'labels', 'patient_ids',
                            'chunks', 'times', 'cpc_scores', 'bin_durations']}

    cpc_lookup = {}
    for pid, cpc in zip(patient_ids, cpc_scores):
        cpc_lookup[pid] = cpc

    seg_ids = np.array([f"{p}_{c}" for p, c in zip(patient_ids, chunks)])
    unique_segs, seg_inverse = np.unique(seg_ids, return_inverse=True)

    dropped_bins = 0
    kept_partial = 0

    def emit_bin(bin_feats, bin_acts, bin_labels, bin_durations, bin_start_time,
                 accum, s_pid, s_chunk):
        durations = np.array(bin_durations)
        weights = durations / durations.sum()

        out['features'].append(
            np.average(np.array(bin_feats), axis=0, weights=weights))
        out['activations'].append(
            np.average(np.array(bin_acts), axis=0, weights=weights))

        label_weight = {}
        for lbl, w in zip(bin_labels, weights):
            label_weight[lbl] = label_weight.get(lbl, 0) + w
        out['labels'].append(max(label_weight, key=label_weight.get))

        out['times'].append(bin_start_time + accum / 2)
        out['patient_ids'].append(s_pid)
        out['chunks'].append(s_chunk)
        out['cpc_scores'].append(cpc_lookup[s_pid])
        out['bin_durations'].append(accum)

    for seg_idx in range(len(unique_segs)):
        mask = seg_inverse == seg_idx
        s_feat = features[mask]
        s_act  = activations[mask]
        s_labels = predictions[mask]
        s_res  = resolutions[mask].astype(float)
        s_times = times[mask].astype(float)
        s_pid  = patient_ids[mask][0]
        s_chunk = chunks[mask][0]

        bin_feats, bin_acts, bin_labels, bin_durations = [], [], [], []
        bin_start_time = s_times[0]
        accum = 0.0

        for i in range(len(s_feat)):
            row_dur = s_res[i]

            # If adding this row would exceed bin_duration AND bin is non-empty,
            # emit current bin first, then start new bin with this row
            if accum + row_dur > bin_duration and len(bin_feats) > 0:
                emit_bin(bin_feats, bin_acts, bin_labels, bin_durations,
                         bin_start_time, accum, s_pid, s_chunk)
                bin_feats, bin_acts, bin_labels, bin_durations = [], [], [], []
                accum = 0.0
                bin_start_time = s_times[i]

            bin_feats.append(s_feat[i])
            bin_acts.append(s_act[i])
            bin_labels.append(s_labels[i])
            bin_durations.append(row_dur)
            accum += row_dur

            # Emit if we hit exactly bin_duration
            if accum == bin_duration:
                emit_bin(bin_feats, bin_acts, bin_labels, bin_durations,
                         bin_start_time, accum, s_pid, s_chunk)
                bin_feats, bin_acts, bin_labels, bin_durations = [], [], [], []
                accum = 0.0
                if i + 1 < len(s_feat):
                    bin_start_time = s_times[i + 1]

        # Handle leftover rows at end of segment
        if len(bin_feats) > 0:
            if accum < bin_duration:
                if drop_partial:
                    dropped_bins += 1
                else:
                    kept_partial += 1
                    emit_bin(bin_feats, bin_acts, bin_labels, bin_durations,
                             bin_start_time, accum, s_pid, s_chunk)
            else:
                emit_bin(bin_feats, bin_acts, bin_labels, bin_durations,
                         bin_start_time, accum, s_pid, s_chunk)

    print(f"  Dropped partial bins: {dropped_bins}")
    print(f"  Kept partial bins: {kept_partial}")

    return {
        'features':      np.array(out['features'], dtype=np.float32),
        'activations':   np.array(out['activations'], dtype=np.float32),
        'labels':        np.array(out['labels']),
        'patient_ids':   np.array(out['patient_ids']),
        'chunks':        np.array(out['chunks']),
        'times':         np.array(out['times']),
        'cpc_scores':    np.array(out['cpc_scores']),
        'bin_durations': np.array(out['bin_durations']),
    }

print("Resampling train...")
train_resampled = resample_to_bins(
    train_data['features'], train_data['predictions'], train_data['activations'],
    train_data['patient_ids'], train_data['chunks'],
    train_data['resolutions'], train_data['times'], train_data['cpc_scores'])

print("Resampling test...")
test_resampled = resample_to_bins(
    test_data['features'], test_data['predictions'], test_data['activations'],
    test_data['patient_ids'], test_data['chunks'],
    test_data['resolutions'], test_data['times'], test_data['cpc_scores'])

X_train_raw = train_resampled['features']
X_test_raw  = test_resampled['features']
y_train = train_resampled['labels'] - 1  # shift 1-8 -> 0-7
y_test  = test_resampled['labels'] - 1
pid_train = train_resampled['patient_ids']
pid_test  = test_resampled['patient_ids']

print(f"\n  Train: {X_train_raw.shape[0]} bins from {len(np.unique(pid_train))} patients")
print(f"  Test:  {X_test_raw.shape[0]} bins from {len(np.unique(pid_test))} patients")
print(f"  Bin durations (train): min={train_resampled['bin_durations'].min():.0f}s, "
      f"max={train_resampled['bin_durations'].max():.0f}s, "
      f"mean={train_resampled['bin_durations'].mean():.0f}s")

# ═══════════════════════════════════════════════════════════════════
# A3: PCA fit on train, transform both
# ═══════════════════════════════════════════════════════════════════
print("\n=== A3: PCA (1275 -> 50) ===")
pca = PCA(n_components=50, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca  = pca.transform(X_test_raw)
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ═══════════════════════════════════════════════════════════════════
# A4: StandardScaler fit on train, transform both
# ═══════════════════════════════════════════════════════════════════
print("\n=== A4: Scaling ===")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_pca)
X_test  = scaler.transform(X_test_pca)

# ═══════════════════════════════════════════════════════════════════
# A5: Pre-compute patient boundaries (for training -- time objective
#     crosses chunks, only respects patient boundaries)
# ═══════════════════════════════════════════════════════════════════
print("\n=== A5: Patient boundaries ===")
def compute_boundaries(patient_ids):
    """Returns (starts, ends) arrays: for each sample, [start, end) of its patient."""
    unique_pids, inverse = np.unique(patient_ids, return_inverse=True)
    N = len(patient_ids)
    starts = np.zeros(N, dtype=np.int64)
    ends = np.zeros(N, dtype=np.int64)
    for pid_idx in range(len(unique_pids)):
        idxs = np.where(inverse == pid_idx)[0]
        s, e = idxs[0], idxs[-1] + 1
        starts[idxs] = s
        ends[idxs] = e
    return starts, ends

pat_starts_train, pat_ends_train = compute_boundaries(pid_train)
pat_starts_test,  pat_ends_test  = compute_boundaries(pid_test)

_, first_idx = np.unique(pid_train, return_index=True)
pat_lens = (pat_ends_train - pat_starts_train)[first_idx]
print(f"  Train patients: {len(first_idx)}")
print(f"  Test patients:  {len(np.unique(pid_test))}")
print(f"  Bins per patient (train): min={pat_lens.min()}, max={pat_lens.max()}, median={np.median(pat_lens):.0f}")

# ═══════════════════════════════════════════════════════════════════
# Save everything for Segment B
# ═══════════════════════════════════════════════════════════════════
print("\n=== Saving ===")
np.savez(f'{DATA_DIR}/cebra_prep_train.npz',
         X=X_train, y=y_train, patient_ids=pid_train,
         chunks=train_resampled['chunks'],
         times=train_resampled['times'],
         activations=train_resampled['activations'],
         cpc_scores=train_resampled['cpc_scores'],
         bin_durations=train_resampled['bin_durations'],
         pat_starts=pat_starts_train, pat_ends=pat_ends_train)

np.savez(f'{DATA_DIR}/cebra_prep_test.npz',
         X=X_test, y=y_test, patient_ids=pid_test,
         chunks=test_resampled['chunks'],
         times=test_resampled['times'],
         activations=test_resampled['activations'],
         cpc_scores=test_resampled['cpc_scores'],
         bin_durations=test_resampled['bin_durations'],
         pat_starts=pat_starts_test, pat_ends=pat_ends_test)

with open(f'{DATA_DIR}/cebra_pca_scaler.pkl', 'wb') as f:
    pickle.dump({'pca': pca, 'scaler': scaler}, f)

print("Done. Saved cebra_prep_train.npz, cebra_prep_test.npz, cebra_pca_scaler.pkl")
