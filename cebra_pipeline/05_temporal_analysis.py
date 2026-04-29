"""05_temporal_analysis.py — temporal / dynamics analysis on CEBRA embeddings."""
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from collections import Counter

from _constants import (CLASS_NAMES, CLASS_COLOR_LIST, CLASS_COLORS,
                        DISPLAY_ORDER, DISPLAY_NAMES, N_CLASSES, BIN_SEC,
                        OUT_DIR)

RUN_TAG = 'combo_i'

# ── Load ────────────────────────────────────────────────────────────
print(f"=== Loading (RUN_TAG={RUN_TAG}) ===")
train_prep = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_train.npz', allow_pickle=True)
test_prep  = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_test.npz',  allow_pickle=True)
X_train_emb = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_train.npz')['embedding']
X_test_emb  = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_test.npz')['embedding']

y_train    = train_prep['predictions'].astype(int)
y_test     = test_prep['predictions'].astype(int)
pid_test   = test_prep['patient_ids']
times_test = test_prep['times'].astype(float)

print(f"  Train: {X_train_emb.shape}, Test: {X_test_emb.shape}")
print(f"  Test patients: {len(np.unique(pid_test))}")

# ═══════════════════════════════════════════════════════════════════
# D1: Train centroids (unit-normalized for cosine)
# ═══════════════════════════════════════════════════════════════════
print("\n=== D1: Train centroids ===")
centroids = np.zeros((N_CLASSES, X_train_emb.shape[1]))
for c in range(N_CLASSES):
    m = y_train == c
    if m.any():
        centroids[c] = X_train_emb[m].mean(axis=0)
        centroids[c] /= np.linalg.norm(centroids[c]) + 1e-12

# ═══════════════════════════════════════════════════════════════════
# D2: Centroid similarity over time (top-N test patients)
# ═══════════════════════════════════════════════════════════════════
print("\n=== D2: Distance to centroids over time ===")
unique_pids = np.unique(pid_test)
n_patients = len(unique_pids)

pid_counts = Counter(pid_test)
top_pids = [pid for pid, _ in pid_counts.most_common(4)]

n_pat = len(top_pids)
fig = plt.figure(figsize=(14, 3 * n_pat))
gs = GridSpec(n_pat * 2, 2, figure=fig,
              height_ratios=[5, 1] * n_pat,
              width_ratios=[50, 1],
              hspace=0.4, wspace=0.05)

label_cmap = ListedColormap(CLASS_COLOR_LIST)
im = None
for p_i, pid in enumerate(top_pids):
    ax_heat  = fig.add_subplot(gs[p_i * 2,     0])
    ax_label = fig.add_subplot(gs[p_i * 2 + 1, 0])

    mask = pid_test == pid
    emb_pat = X_test_emb[mask]
    t_pat = times_test[mask]
    y_pat = y_test[mask]
    order = np.argsort(t_pat)
    emb_pat, t_pat, y_pat = emb_pat[order], t_pat[order], y_pat[order]
    t_hours = (t_pat - t_pat[0]) / 3600.0

    # cosine sim to centroids -> reorder rows to display order
    sim = 1.0 - cdist(emb_pat, centroids, metric='cosine')           # (T, 8) data order
    sim_disp = sim[:, DISPLAY_ORDER]

    im = ax_heat.imshow(sim_disp.T, aspect='auto', cmap='RdYlBu_r',
                        vmin=-0.5, vmax=1.0,
                        extent=[t_hours[0], t_hours[-1], N_CLASSES - 0.5, -0.5],
                        interpolation='nearest')
    ax_heat.set_yticks(range(N_CLASSES))
    ax_heat.set_yticklabels(DISPLAY_NAMES, fontsize=7)
    ax_heat.set_title(f'Patient {pid} ({len(t_hours)} bins)', fontsize=9)
    ax_heat.tick_params(axis='x', labelbottom=False)

    ax_label.imshow(y_pat[None, :], aspect='auto', cmap=label_cmap,
                    vmin=0, vmax=N_CLASSES - 1,
                    extent=[t_hours[0], t_hours[-1], 0, 1],
                    interpolation='nearest')
    ax_label.set_yticks([])
    ax_label.set_ylabel('Truth', fontsize=7, rotation=0, ha='right', va='center')
    if p_i == n_pat - 1:
        ax_label.set_xlabel('Time (hours)')
    else:
        ax_label.tick_params(axis='x', labelbottom=False)

cbar_ax = fig.add_subplot(gs[:, 1])
fig.colorbar(im, cax=cbar_ax, label='Cosine Similarity to Centroid')

legend_patches = [Patch(facecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in DISPLAY_ORDER]
fig.legend(handles=legend_patches, loc='lower center', ncol=N_CLASSES,
           fontsize=7, frameon=False, bbox_to_anchor=(0.45, -0.02))

fig.suptitle(f'Similarity to Class Centroids Over Time — Test ({RUN_TAG})', fontsize=13)
plt.savefig(f'{OUT_DIR}/cebra_d2_centroid_distances_over_time_{RUN_TAG}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_d2_centroid_distances_over_time_{RUN_TAG}.png")

# ═══════════════════════════════════════════════════════════════════
# D3: kNN entropy + state changes
# ═══════════════════════════════════════════════════════════════════
print("\n=== D3: Prediction entropy over time ===")
# CEBRA embeddings are L2-normalized (unit sphere) -> euclidean kNN is
# equivalent to cosine kNN, but euclidean uses BallTree (much faster than
# brute-force cosine when N is large).
def _l2(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean', n_jobs=-1)
knn.fit(_l2(X_train_emb), y_train)
probs_test = knn.predict_proba(_l2(X_test_emb))
y_pred = knn.predict(_l2(X_test_emb)).astype(int)

def entropy(p):
    p = p.clip(1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)
H = entropy(probs_test)

fig, axes = plt.subplots(len(top_pids), 1,
                         figsize=(14, 3.5 * len(top_pids)), sharex=False)
if len(top_pids) == 1:
    axes = [axes]

for ax_i, pid in enumerate(top_pids):
    mask = pid_test == pid
    t_pat = times_test[mask]
    order = np.argsort(t_pat)
    t_hours = (t_pat[order] - t_pat[order][0]) / 3600.0
    H_pat = H[mask][order]
    y_pred_pat = y_pred[mask][order]

    ax = axes[ax_i]
    for i in range(len(t_hours) - 1):
        ax.plot(t_hours[i:i+2], H_pat[i:i+2],
                color=CLASS_COLORS[y_pred_pat[i]], linewidth=1, alpha=0.8)

    changes = np.where(y_pred_pat[1:] != y_pred_pat[:-1])[0] + 1
    if len(changes):
        ax.scatter(t_hours[changes], H_pat[changes], c='black', s=15, marker='|',
                   zorder=5, label=f'{len(changes)} transitions')

    ax.set_ylabel('Entropy (bits)')
    ax.set_title(f'Patient {pid} — Prediction Entropy ({len(changes)} state changes)')
    ax.set_ylim(-0.1, np.log2(N_CLASSES) + 0.2)
    ax.axhline(np.log2(N_CLASSES), color='gray', ls='--', lw=0.5, alpha=0.5)

axes[-1].set_xlabel('Time (hours)')
plt.suptitle(f'kNN Prediction Entropy Over Time ({RUN_TAG})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_d3_entropy_over_time_{RUN_TAG}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_d3_entropy_over_time_{RUN_TAG}.png")

# ═══════════════════════════════════════════════════════════════════
# D4a: Transition matrix (display order)
# ═══════════════════════════════════════════════════════════════════
print("\n=== D4a: Transition matrix ===")
trans_counts = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for pid in unique_pids:
    mask = pid_test == pid
    order = np.argsort(times_test[mask])
    yp = y_pred[mask][order]
    for i in range(len(yp) - 1):
        trans_counts[yp[i], yp[i + 1]] += 1
row_sum = trans_counts.sum(axis=1, keepdims=True)
trans_prob = np.divide(trans_counts, row_sum, where=row_sum > 0,
                       out=np.zeros_like(trans_counts, dtype=float))

tc_disp = trans_counts[np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]
tp_disp = trans_prob  [np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
im0 = axes[0].imshow(tc_disp, cmap='YlOrRd', aspect='equal')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        v = tc_disp[i, j]
        axes[0].text(j, i, f'{v}', ha='center', va='center', fontsize=7,
                     fontweight='bold',
                     color='white' if v > tc_disp.max() * 0.6 else 'black')
axes[0].set_xticks(range(N_CLASSES)); axes[0].set_yticks(range(N_CLASSES))
axes[0].set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(DISPLAY_NAMES, fontsize=8)
axes[0].set_xlabel('To'); axes[0].set_ylabel('From')
axes[0].set_title('Transition Counts')
fig.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(tp_disp, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        v = tp_disp[i, j]
        axes[1].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                     fontweight='bold', color='white' if v > 0.5 else 'black')
axes[1].set_xticks(range(N_CLASSES)); axes[1].set_yticks(range(N_CLASSES))
axes[1].set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(DISPLAY_NAMES, fontsize=8)
axes[1].set_xlabel('To'); axes[1].set_ylabel('From')
axes[1].set_title('Transition Probabilities')
fig.colorbar(im1, ax=axes[1], shrink=0.8)
plt.suptitle(f'State Transition Matrix — kNN ({RUN_TAG})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_d4a_transition_matrix_{RUN_TAG}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_d4a_transition_matrix_{RUN_TAG}.png")

# ═══════════════════════════════════════════════════════════════════
# D4b: Sojourn times (display order)
# ═══════════════════════════════════════════════════════════════════
print("\n=== D4b: Sojourn times ===")
sojourn_times = {c: [] for c in range(N_CLASSES)}
for pid in unique_pids:
    mask = pid_test == pid
    order = np.argsort(times_test[mask])
    yp = y_pred[mask][order]
    cur, run = yp[0], 1
    for i in range(1, len(yp)):
        if yp[i] == cur:
            run += 1
        else:
            sojourn_times[cur].append(run)
            cur, run = yp[i], 1
    sojourn_times[cur].append(run)

fig, ax = plt.subplots(figsize=(12, 5))
data = []
for c in DISPLAY_ORDER:
    if sojourn_times[c]:
        data.append(np.log10(np.array(sojourn_times[c]) * (BIN_SEC / 60.0) + 1))
    else:
        data.append(np.array([0]))

vp = ax.violinplot(data, positions=range(N_CLASSES), showmedians=True, showextrema=False)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(CLASS_COLORS[DISPLAY_ORDER[i]])
    body.set_alpha(0.7)
vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(1.5)

tick_vals = [np.log10(v + 1) for v in [5, 15, 30, 60, 120, 360, 720]]
tick_labels = ['5m', '15m', '30m', '1h', '2h', '6h', '12h']
ax.set_yticks(tick_vals); ax.set_yticklabels(tick_labels)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Sojourn Time')
ax.set_title(f'Sojourn Times per State — kNN ({RUN_TAG})')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_d4b_sojourn_times_{RUN_TAG}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_d4b_sojourn_times_{RUN_TAG}.png")

print("  Median sojourn times (minutes):")
for c in DISPLAY_ORDER:
    if sojourn_times[c]:
        v = np.array(sojourn_times[c]) * (BIN_SEC / 60.0)
        print(f"    {CLASS_NAMES[c]:15s}: median={np.median(v):.0f}, mean={np.mean(v):.0f}, "
              f"max={np.max(v):.0f}, n_episodes={len(sojourn_times[c])}")

# ═══════════════════════════════════════════════════════════════════
# D4c: First-passage time (display order)
# ═══════════════════════════════════════════════════════════════════
print("\n=== D4c: First-passage times ===")
first_passage = {c: [] for c in range(N_CLASSES)}
for pid in unique_pids:
    mask = pid_test == pid
    order = np.argsort(times_test[mask])
    yp = y_pred[mask][order]
    ts = times_test[mask][order]
    t0 = ts[0]
    seen = set()
    for i in range(len(yp)):
        c = int(yp[i])
        if c not in seen:
            seen.add(c)
            first_passage[c].append((ts[i] - t0) / 60.0)   # minutes

fig, ax = plt.subplots(figsize=(12, 5))
data_fp = []
for c in DISPLAY_ORDER:
    data_fp.append(np.array(first_passage[c]) / 60.0 if first_passage[c] else np.array([0]))

bp = ax.boxplot(data_fp, positions=list(range(N_CLASSES)), widths=0.6,
                patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5))
for patch, c in zip(bp['boxes'], DISPLAY_ORDER):
    patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.7)
for i, c in enumerate(DISPLAY_ORDER):
    jitter = np.random.uniform(-0.15, 0.15, len(data_fp[i]))
    ax.scatter(i + jitter, data_fp[i], c=CLASS_COLORS[c], s=5, alpha=0.3, zorder=3)

ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('First Passage Time (hours)')
ax.set_title(f'First Passage Time to Each State — kNN ({RUN_TAG})')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_d4c_first_passage_{RUN_TAG}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_d4c_first_passage_{RUN_TAG}.png")

print("  Median first-passage times (hours):")
for c in DISPLAY_ORDER:
    if first_passage[c]:
        v = np.array(first_passage[c]) / 60.0
        print(f"    {CLASS_NAMES[c]:15s}: median={np.median(v):.1f}h, "
              f"n_patients={len(first_passage[c])}/{n_patients}")

# ═══════════════════════════════════════════════════════════════════
# D4d: Path length (no class ordering)
# ═══════════════════════════════════════════════════════════════════
print("\n=== D4d: Path length ===")
path_lengths, hours_per_patient = [], []
for pid in unique_pids:
    mask = pid_test == pid
    order = np.argsort(times_test[mask])
    emb_pat = X_test_emb[mask][order]
    t_pat = times_test[mask][order]
    if len(emb_pat) > 1:
        e_norm = emb_pat / (np.linalg.norm(emb_pat, axis=1, keepdims=True) + 1e-10)
        consec = 1 - (e_norm[:-1] * e_norm[1:]).sum(axis=1)
        path_lengths.append(consec.sum())
    else:
        path_lengths.append(0.0)
    hours_per_patient.append(max((t_pat.max() - t_pat.min()) / 3600.0, BIN_SEC/3600.0))

path_lengths = np.array(path_lengths)
hours_per_patient = np.array(hours_per_patient)
path_rate = path_lengths / hours_per_patient

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(path_lengths, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Total Path Length (cosine)'); axes[0].set_ylabel('Number of Patients')
axes[0].set_title('Total Embedding Path Length')
axes[0].axvline(np.median(path_lengths), color='red', ls='--',
                label=f'Median={np.median(path_lengths):.1f}')
axes[0].legend()

axes[1].hist(path_rate, bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Path Length per Hour'); axes[1].set_ylabel('Number of Patients')
axes[1].set_title('Embedding Drift Rate')
axes[1].axvline(np.median(path_rate), color='red', ls='--',
                label=f'Median={np.median(path_rate):.1f}')
axes[1].legend()

plt.suptitle(f'Embedding Path Length ({RUN_TAG})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_d4d_path_length_{RUN_TAG}.png',
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_d4d_path_length_{RUN_TAG}.png")

print(f"  Path length: median={np.median(path_lengths):.2f}, mean={np.mean(path_lengths):.2f}")
print(f"  Drift rate:  median={np.median(path_rate):.2f}/hr, mean={np.mean(path_rate):.2f}/hr")

print("\n" + "=" * 50)
print(f"SEGMENT D COMPLETE ({RUN_TAG})")
print("=" * 50)
