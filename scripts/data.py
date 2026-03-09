"""Data loading and preprocessing. Patients are stored in contiguous time-sorted blocks."""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def load_patient_ids(csv_path):
    return pd.read_csv(csv_path)['patient_id'].tolist()


_META_COLS = {'patient', 'seg_no', 'file', 'row_in_seg', 'rel_sec',
              'pat_ICARE', 'cpc', 'cpc_bin', 'ROSC(minutes)', 'age',
              'sex', 'vfib', 'time_to_CA(seconds)'}


def _detect_features(df):
    return [c for c in df.columns if c not in _META_COLS]


def _clean_nans(X, strategy='mean'):
    X = np.where(np.isinf(X), np.nan, X)
    n = np.isnan(X).sum()
    if n == 0:
        return X, None
    print(f"  NaN: {n} ({n / X.size * 100:.1f}%)")
    if strategy == 'drop':
        mask = ~np.isnan(X).any(axis=1)
        print(f"  Dropped {(~mask).sum()} rows")
        return X[mask], mask
    if strategy == 'median':
        fills = np.nanmedian(X, axis=0)
    elif strategy == 'zero':
        return np.nan_to_num(X, nan=0.0), None
    elif strategy == 'ffill':
        return pd.DataFrame(X).ffill().fillna(0).values.astype(np.float32), None
    else:  # mean
        fills = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(fills, idx[1])
    return X, None


def _read_csv(args):
    path, labels_df, feat_cols, patient_set = args
    df = pd.read_csv(path, encoding='utf-8-sig')
    if 'patient' not in df.columns:
        return None
    pid = df['patient'].iloc[0]
    if pid not in patient_set:
        return None
    df = df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')
    df = df.dropna(subset=['cpc_bin'])
    if len(df) == 0:
        return None
    df = df.sort_values('rel_sec')
    return {
        'pid': pid,
        'neural': df[feat_cols].values.astype(np.float32),
        'cpc_bin': (df['cpc_bin'] == 'poor').astype(np.int32).values,
        'rel_sec': df['rel_sec'].values.astype(np.float32),
        'names': np.array([pid] * len(df)),
    }


def load_and_preprocess(neural_dir, labels_path, patient_ids, nan_strategy='mean', n_workers=4):
    """Load CSVs for given patients, sort by patient then time, clean NaNs."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    labels_df = pd.read_csv(labels_path)
    patient_set = set(patient_ids)
    csvs = sorted(Path(neural_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {neural_dir}")

    # Filter to CSVs that have a 'patient' column
    csvs = [f for f in csvs if 'patient' in pd.read_csv(f, encoding='utf-8-sig', nrows=0).columns]

    # Detect feature columns
    sample = pd.read_csv(csvs[0], encoding='utf-8-sig', nrows=5)
    sample = sample.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')
    feat_cols = _detect_features(sample)
    print(f"Features: {len(feat_cols)}, CSVs: {len(csvs)}")

    # Parallel load
    args = [(f, labels_df, feat_cols, patient_set) for f in csvs]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(tqdm(ex.map(_read_csv, args), total=len(csvs), desc="Loading"))
    results = sorted([r for r in results if r is not None], key=lambda r: r['pid'])

    neural = np.concatenate([r['neural'] for r in results])
    cpc_bin = np.concatenate([r['cpc_bin'] for r in results])
    rel_sec = np.concatenate([r['rel_sec'] for r in results])
    names = np.concatenate([r['names'] for r in results])
    del results; gc.collect()

    print(f"Cleaning NaNs ({nan_strategy})...")
    neural, mask = _clean_nans(neural, nan_strategy)
    if mask is not None:
        cpc_bin, rel_sec, names = cpc_bin[mask], rel_sec[mask], names[mask]

    unique = sorted(set(names))
    pid_map = {p: i for i, p in enumerate(unique)}
    patient_ids_int = np.array([pid_map[n] for n in names], dtype=np.int32)

    print(f"Loaded: {neural.shape[0]} samples, {len(unique)} patients")
    print(f"  cpc_bin dist: {np.bincount(cpc_bin).tolist()}")

    return {
        'neural': neural, 'cpc_bin': cpc_bin,
        'patient_ids': patient_ids_int, 'patient_names': names,
        'rel_sec': rel_sec, 'feature_names': np.array(feat_cols),
    }


def load_or_cache(npz_path, neural_dir, labels_path, patient_ids,
                  nan_strategy='mean', n_workers=4):
    """Load cached NPZ if it exists, otherwise preprocess and save."""
    if Path(npz_path).exists():
        print(f"Cached: {npz_path}")
        return dict(np.load(npz_path, allow_pickle=True))
    data = load_and_preprocess(neural_dir, labels_path, patient_ids, nan_strategy, n_workers)
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **data)
    print(f"Saved: {npz_path}")
    return data
