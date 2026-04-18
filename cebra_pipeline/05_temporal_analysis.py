import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from collections import Counter

DATA_DIR = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model'

label_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA',
               'Burst Supp.', 'Continuous', 'Discontinuous']
N_CLASSES = len(label_names)

# ── Load ────────────────────────────────────────────────────────────
print("=== Loading ===")
train_prep = np.load(f'{DATA_DIR}/cebra_prep_train.npz', allow_pickle=True)
test_prep  = np.load(f'{DATA_DIR}/cebra_prep_test.npz', allow_pickle=True)
X_train_emb = np.load(f'{DATA_DIR}/cebra_embeddings_train.npz')['embedding']
X_test_emb  = np.load(f'{DATA_DIR}/cebra_embeddings_test.npz')['embedding']

y_train = train_prep['y']
y_test  = test_prep['y']
pid_test   = test_prep['patient_ids']
times_test = test_prep['times'].astype(float)

print(f"  Train: {X_train_emb.shape}, Test: {X_test_emb.shape}")
print(f"  Test patients: {len(np.unique(pid_test))}")

# ═══════════════════════════════════════════════════════════════════
# D1: Compute label centroids from TRAIN embeddings
# ═══════════════════════════════════════════════════════════════════
print("\n=== D1: Train centroids ===")
centroids = np.zeros((N_CLASSES, X_train_emb.shape[1]))
for c in range(N_CLASSES):
    mask = y_train == c
    if mask.any():
        centroids[c] = X_train_emb[mask].mean(axis=0)
        # Normalize to unit sphere for cosine consistency
        centroids[c] /= np.linalg.norm(centroids[c])
print("  Centroids computed (unit-normalized from train set)")

# ═══════════════════════════════════════════════════════════════════
# D2: Per test patient — distance to each centroid over time
# ═══════════════════════════════════════════════════════════════════
print("\n=== D2: Distance to centroids over time ===")

unique_pids = np.unique(pid_test)
n_patients = len(unique_pids)

# Pick up to 4 patients with the most time points for visualization
pid_counts = Counter(pid_test)
top_pids = [pid for pid, _ in pid_counts.most_common(4)]

colors = ['#E63946', '#2A9D8F', '#E9C46A', '#7209B7',
          '#0077B6', '#F72585', '#8D99AE', '#6A4C3C']

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

n_pat = len(top_pids)
fig = plt.figure(figsize=(14, 3 * n_pat))
gs = GridSpec(n_pat * 2, 2, figure=fig,
             height_ratios=[5, 1] * n_pat,
             width_ratios=[50, 1],  # main + colorbar column
             hspace=0.4, wspace=0.05)

im = None
for p_i, pid in enumerate(top_pids):
    ax_heat = fig.add_subplot(gs[p_i * 2, 0])
    ax_label = fig.add_subplot(gs[p_i * 2 + 1, 0])

    mask = pid_test == pid
    emb_pat = X_test_emb[mask]
    t_pat = times_test[mask]
    y_pat = y_test[mask]

    order = np.argsort(t_pat)
    emb_pat = emb_pat[order]
    t_pat = t_pat[order]
    y_pat = y_pat[order]

    t_hours = (t_pat - t_pat[0]) / 3600.0

    # Cosine similarity to each centroid: (T, 8)
    sim = 1.0 - cdist(emb_pat, centroids, metric='cosine')

    # Heatmap: rows = classes, cols = time
    im = ax_heat.imshow(sim.T, aspect='auto', cmap='RdYlBu_r',
                        vmin=-0.5, vmax=1.0,
                        extent=[t_hours[0], t_hours[-1], N_CLASSES - 0.5, -0.5],
                        interpolation='nearest')
    ax_heat.set_yticks(range(N_CLASSES))
    ax_heat.set_yticklabels(label_names, fontsize=7)
    ax_heat.set_title(f'Patient {pid} ({len(t_hours)} bins)', fontsize=9)
    ax_heat.tick_params(axis='x', labelbottom=False)

    # True label strip: colored bar showing ground-truth label over time
    # Build a 1-row image from label indices, use a ListedColormap
    from matplotlib.colors import ListedColormap
    label_cmap = ListedColormap(colors)
    ax_label.imshow(y_pat[None, :], aspect='auto', cmap=label_cmap,
                    vmin=0, vmax=N_CLASSES - 1,
                    extent=[t_hours[0], t_hours[-1], 0, 1],
                    interpolation='nearest')
    ax_label.set_yticks([])
    ax_label.set_ylabel('Ground\nTruth', fontsize=7, rotation=0, ha='right', va='center')
    if p_i == n_pat - 1:
        ax_label.set_xlabel('Time (hours)')
    else:
        ax_label.tick_params(axis='x', labelbottom=False)

# Colorbar in the right column
cbar_ax = fig.add_subplot(gs[:, 1])
fig.colorbar(im, cax=cbar_ax, label='Cosine Similarity to Centroid')

# Legend for label colors
legend_patches = [Patch(facecolor=colors[c], label=label_names[c]) for c in range(N_CLASSES)]
fig.legend(handles=legend_patches, loc='lower center', ncol=N_CLASSES,
           fontsize=7, frameon=False, bbox_to_anchor=(0.45, -0.02))

fig.suptitle('Similarity to Class Centroids Over Time (Test Patients)', fontsize=13)
plt.savefig(f'{DATA_DIR}/cebra_d2_centroid_distances_over_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_d2_centroid_distances_over_time.png")

# ═══════════════════════════════════════════════════════════════════
# D3: kNN predicted probabilities → entropy → state changes
# ═══════════════════════════════════════════════════════════════════
print("\n=== D3: Prediction entropy over time ===")

knn = KNeighborsClassifier(n_neighbors=10, metric='cosine', n_jobs=-1)
knn.fit(X_train_emb, y_train)
probs_test = knn.predict_proba(X_test_emb)  # (N_test, 8)

# Entropy per sample
def entropy(p):
    p = p.clip(1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

H = entropy(probs_test)
y_pred = knn.predict(X_test_emb)

fig, axes = plt.subplots(len(top_pids), 1, figsize=(14, 3.5 * len(top_pids)), sharex=False)
if len(top_pids) == 1:
    axes = [axes]

for ax_i, pid in enumerate(top_pids):
    mask = pid_test == pid
    t_pat = times_test[mask]
    order = np.argsort(t_pat)
    t_hours = (t_pat[order] - t_pat[order][0]) / 3600.0
    H_pat = H[mask][order]
    y_pred_pat = y_pred[mask][order]
    y_true_pat = y_test[mask][order]

    ax = axes[ax_i]

    # Color entropy line by predicted class
    for i in range(len(t_hours) - 1):
        ax.plot(t_hours[i:i+2], H_pat[i:i+2], color=colors[y_pred_pat[i]],
                linewidth=1, alpha=0.8)

    # Mark state changes (predicted label changes)
    changes = np.where(y_pred_pat[1:] != y_pred_pat[:-1])[0] + 1
    if len(changes) > 0:
        ax.scatter(t_hours[changes], H_pat[changes], c='black', s=15,
                   marker='|', zorder=5, label=f'{len(changes)} transitions')

    ax.set_ylabel('Entropy (bits)')
    ax.set_title(f'Patient {pid} — Prediction Entropy ({len(changes)} state changes)')
    ax.set_ylim(-0.1, np.log2(N_CLASSES) + 0.2)
    ax.axhline(np.log2(N_CLASSES), color='gray', ls='--', lw=0.5, alpha=0.5)

axes[-1].set_xlabel('Time (hours)')
plt.suptitle('kNN Prediction Entropy Over Time (Test Patients)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_d3_entropy_over_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_d3_entropy_over_time.png")

# ═══════════════════════════════════════════════════════════════════
# D4: Transition matrix, sojourn times, first-passage time, path length
# ═══════════════════════════════════════════════════════════════════
print("\n=== D4: Transition analysis ===")

# --- D4a: Transition matrix (across all test patients) ---
print("\n  D4a: Transition matrix")
trans_counts = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

for pid in unique_pids:
    mask = pid_test == pid
    t_pat = times_test[mask]
    order = np.argsort(t_pat)
    y_pred_pat = y_pred[mask][order]

    for i in range(len(y_pred_pat) - 1):
        trans_counts[y_pred_pat[i], y_pred_pat[i + 1]] += 1

# Normalize rows to get transition probabilities
row_sums = trans_counts.sum(axis=1, keepdims=True)
trans_prob = np.divide(trans_counts, row_sums, where=row_sums > 0,
                       out=np.zeros_like(trans_counts, dtype=float))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
im0 = axes[0].imshow(trans_counts, cmap='YlOrRd', aspect='equal')
n = N_CLASSES
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f'{trans_counts[i, j]}', ha='center', va='center',
                     fontsize=7, fontweight='bold',
                     color='white' if trans_counts[i, j] > trans_counts.max() * 0.6 else 'black')
axes[0].set_xticks(range(n))
axes[0].set_yticks(range(n))
axes[0].set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(label_names, fontsize=8)
axes[0].set_xlabel('To')
axes[0].set_ylabel('From')
axes[0].set_title('Transition Counts')
fig.colorbar(im0, ax=axes[0], shrink=0.8)

# Probabilities
im1 = axes[1].imshow(trans_prob, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
for i in range(n):
    for j in range(n):
        val = trans_prob[i, j]
        axes[1].text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=7, fontweight='bold',
                     color='white' if val > 0.5 else 'black')
axes[1].set_xticks(range(n))
axes[1].set_yticks(range(n))
axes[1].set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(label_names, fontsize=8)
axes[1].set_xlabel('To')
axes[1].set_ylabel('From')
axes[1].set_title('Transition Probabilities')
fig.colorbar(im1, ax=axes[1], shrink=0.8)

plt.suptitle('State Transition Matrix (kNN Predictions, Test Set)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_d4a_transition_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_d4a_transition_matrix.png")

# --- D4b: Sojourn times (how long patient stays in each state) ---
print("\n  D4b: Sojourn times")
sojourn_times = {c: [] for c in range(N_CLASSES)}  # in number of bins

for pid in unique_pids:
    mask = pid_test == pid
    t_pat = times_test[mask]
    order = np.argsort(t_pat)
    y_pred_pat = y_pred[mask][order]

    # Run-length encoding
    current = y_pred_pat[0]
    run_len = 1
    for i in range(1, len(y_pred_pat)):
        if y_pred_pat[i] == current:
            run_len += 1
        else:
            sojourn_times[current].append(run_len)
            current = y_pred_pat[i]
            run_len = 1
    sojourn_times[current].append(run_len)

fig, ax = plt.subplots(figsize=(12, 5))
data_to_plot = []
for c in range(N_CLASSES):
    if len(sojourn_times[c]) > 0:
        data_to_plot.append(np.log10(np.array(sojourn_times[c]) * 5 + 1))
    else:
        data_to_plot.append(np.array([0]))

vp = ax.violinplot(data_to_plot, positions=range(N_CLASSES), showmedians=True, showextrema=False)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors[i])
    body.set_alpha(0.7)
vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(1.5)

# Custom y-axis: log10(minutes) → readable labels
tick_vals = [np.log10(v + 1) for v in [5, 15, 30, 60, 120, 360, 720]]
tick_labels = ['5m', '15m', '30m', '1h', '2h', '6h', '12h']
ax.set_yticks(tick_vals)
ax.set_yticklabels(tick_labels)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Sojourn Time')
ax.set_title('Sojourn Times per State (kNN Predictions, Test Set)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_d4b_sojourn_times.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_d4b_sojourn_times.png")

# Print median sojourn times
print("\n  Median sojourn times (minutes):")
for c in range(N_CLASSES):
    if len(sojourn_times[c]) > 0:
        vals = np.array(sojourn_times[c]) * 5
        print(f"    {label_names[c]:15s}: median={np.median(vals):.0f}, "
              f"mean={np.mean(vals):.0f}, max={np.max(vals):.0f}, "
              f"n_episodes={len(sojourn_times[c])}")

# --- D4c: First-passage time (time to first reach each state, per patient) ---
print("\n  D4c: First-passage times")
first_passage = {c: [] for c in range(N_CLASSES)}  # in minutes from recording start

for pid in unique_pids:
    mask = pid_test == pid
    t_pat = times_test[mask]
    order = np.argsort(t_pat)
    y_pred_pat = y_pred[mask][order]
    t_sorted = t_pat[order]
    t0 = t_sorted[0]

    seen = set()
    for i in range(len(y_pred_pat)):
        c = y_pred_pat[i]
        if c not in seen:
            seen.add(c)
            fp_minutes = (t_sorted[i] - t0) / 60.0
            first_passage[c].append(fp_minutes)

fig, ax = plt.subplots(figsize=(12, 5))
data_fp = []
for c in range(N_CLASSES):
    if len(first_passage[c]) > 0:
        data_fp.append(np.array(first_passage[c]) / 60.0)  # convert to hours
    else:
        data_fp.append(np.array([0]))

bp = ax.boxplot(data_fp, positions=list(range(N_CLASSES)), widths=0.6,
                patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for c in range(N_CLASSES):
    jitter = np.random.uniform(-0.15, 0.15, len(data_fp[c]))
    ax.scatter(c + jitter, data_fp[c], c=colors[c], s=5, alpha=0.3, zorder=3)

ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('First Passage Time (hours)')
ax.set_title('First Passage Time to Each State (kNN Predictions, Test Set)')
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_d4c_first_passage.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_d4c_first_passage.png")

print("\n  Median first-passage times (hours):")
for c in range(N_CLASSES):
    if len(first_passage[c]) > 0:
        vals = np.array(first_passage[c]) / 60.0
        print(f"    {label_names[c]:15s}: median={np.median(vals):.1f}h, "
              f"n_patients={len(first_passage[c])}/{n_patients}")

# --- D4d: Path length (total embedding distance traveled per patient) ---
print("\n  D4d: Path length")
path_lengths = []
path_pids = []

for pid in unique_pids:
    mask = pid_test == pid
    t_pat = times_test[mask]
    order = np.argsort(t_pat)
    emb_pat = X_test_emb[mask][order]

    # Sum of consecutive cosine distances
    if len(emb_pat) > 1:
        consecutive_dists = np.array([
            1 - np.dot(emb_pat[i], emb_pat[i+1]) /
            (np.linalg.norm(emb_pat[i]) * np.linalg.norm(emb_pat[i+1]) + 1e-10)
            for i in range(len(emb_pat) - 1)
        ])
        path_lengths.append(consecutive_dists.sum())
    else:
        path_lengths.append(0.0)
    path_pids.append(pid)

path_lengths = np.array(path_lengths)

# Also compute per-hour normalized path length
hours_per_patient = []
for pid in unique_pids:
    mask = pid_test == pid
    t_pat = times_test[mask]
    duration_h = (t_pat.max() - t_pat.min()) / 3600.0
    hours_per_patient.append(max(duration_h, 1/12))  # floor at 5 min
hours_per_patient = np.array(hours_per_patient)
path_rate = path_lengths / hours_per_patient

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(path_lengths, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Total Path Length (cosine)')
axes[0].set_ylabel('Number of Patients')
axes[0].set_title('Total Embedding Path Length')
axes[0].axvline(np.median(path_lengths), color='red', ls='--', label=f'Median={np.median(path_lengths):.1f}')
axes[0].legend()

axes[1].hist(path_rate, bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Path Length per Hour (cosine/hr)')
axes[1].set_ylabel('Number of Patients')
axes[1].set_title('Embedding Drift Rate')
axes[1].axvline(np.median(path_rate), color='red', ls='--', label=f'Median={np.median(path_rate):.1f}')
axes[1].legend()

plt.suptitle('Embedding Path Length (Test Patients)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_d4d_path_length.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_d4d_path_length.png")

print(f"\n  Path length: median={np.median(path_lengths):.2f}, "
      f"mean={np.mean(path_lengths):.2f}, std={np.std(path_lengths):.2f}")
print(f"  Drift rate:  median={np.median(path_rate):.2f}/hr, "
      f"mean={np.mean(path_rate):.2f}/hr")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("SEGMENT D COMPLETE")
print("=" * 50)
print(f"  Plots saved:")
print(f"    D2: cebra_d2_centroid_distances_over_time.png")
print(f"    D3: cebra_d3_entropy_over_time.png")
print(f"    D4a: cebra_d4a_transition_matrix.png")
print(f"    D4b: cebra_d4b_sojourn_times.png")
print(f"    D4c: cebra_d4c_first_passage.png")
print(f"    D4d: cebra_d4d_path_length.png")
print("=" * 50)
