import numpy as np
np.random.seed(42)
import torch
import cebra.models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             confusion_matrix, silhouette_score)
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model'

# ── Load ────────────────────────────────────────────────────────────
print("=== Loading ===")
train_prep = np.load(f'{DATA_DIR}/cebra_prep_train.npz', allow_pickle=True)
test_prep  = np.load(f'{DATA_DIR}/cebra_prep_test.npz', allow_pickle=True)
X_train_emb = np.load(f'{DATA_DIR}/cebra_embeddings_train.npz')['embedding']
X_test_emb  = np.load(f'{DATA_DIR}/cebra_embeddings_test.npz')['embedding']

y_train = train_prep['y']
y_test  = test_prep['y']

label_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA',
               'Burst Supp.', 'Continuous', 'Discontinuous']

print(f"  Train: {X_train_emb.shape}, Test: {X_test_emb.shape}")
print(f"  Train labels: {np.unique(y_train)}, Test labels: {np.unique(y_test)}")

# ═══════════════════════════════════════════════════════════════════
# C1: Held-out contrastive loss (L_n - log(n))
# ═══════════════════════════════════════════════════════════════════
print("\n=== C1: Held-out contrastive loss ===")

def compute_infonce(embeddings, labels, n_samples=2048, temperature=1.0, n_runs=10):
    """Compute InfoNCE loss on held-out data using discrete label matching."""
    losses = []
    for _ in range(n_runs):
        idx = np.random.choice(len(embeddings), n_samples * 2, replace=False)
        ref_idx = idx[:n_samples]
        neg_idx = idx[n_samples:]

        # Find positives: random sample with same label
        pos_idx = np.zeros(n_samples, dtype=int)
        for i, ri in enumerate(ref_idx):
            same_label = np.where(labels == labels[ri])[0]
            pos_idx[i] = np.random.choice(same_label)

        ref = torch.tensor(embeddings[ref_idx], dtype=torch.float32)
        pos = torch.tensor(embeddings[pos_idx], dtype=torch.float32)
        neg = torch.tensor(embeddings[neg_idx], dtype=torch.float32)

        pos_sim = torch.einsum('ni,ni->n', ref, pos) / temperature
        neg_sim = torch.einsum('ni,mi->nm', ref, neg) / temperature

        # InfoNCE = align + uniform
        with torch.no_grad():
            c, _ = neg_sim.max(dim=1, keepdim=True)
        pos_sim = pos_sim - c.squeeze(1)
        neg_sim = neg_sim - c
        align = (-pos_sim).mean()
        uniform = torch.logsumexp(neg_sim, dim=1).mean()
        loss = (align + uniform).item()
        losses.append(loss)

    raw = np.mean(losses)
    normalized = raw - np.log(n_samples)  # L_n - log(n)
    return raw, normalized

train_raw, train_norm = compute_infonce(X_train_emb, y_train)
test_raw, test_norm   = compute_infonce(X_test_emb, y_test)
print(f"  Train InfoNCE: {train_raw:.3f} (normalized: {train_norm:.3f})")
print(f"  Test  InfoNCE: {test_raw:.3f} (normalized: {test_norm:.3f})")

# ═══════════════════════════════════════════════════════════════════
# C2: kNN classifier
# ═══════════════════════════════════════════════════════════════════
print("\n=== C2: kNN classifier ===")
knn = KNeighborsClassifier(n_neighbors=10, metric='cosine', n_jobs=-1)
knn.fit(X_train_emb, y_train)
y_pred_knn = knn.predict(X_test_emb)

bal_acc_knn = balanced_accuracy_score(y_test, y_pred_knn)
f1_knn      = f1_score(y_test, y_pred_knn, average='macro')
cm_knn      = confusion_matrix(y_test, y_pred_knn)

print(f"  Balanced Accuracy: {bal_acc_knn:.4f}")
print(f"  Macro F1:          {f1_knn:.4f}")

# ═══════════════════════════════════════════════════════════════════
# C3: Logistic regression
# ═══════════════════════════════════════════════════════════════════
print("\n=== C3: Logistic regression ===")
lr = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=42)
lr.fit(X_train_emb, y_train)
y_pred_lr = lr.predict(X_test_emb)

bal_acc_lr = balanced_accuracy_score(y_test, y_pred_lr)
f1_lr      = f1_score(y_test, y_pred_lr, average='macro')
cm_lr      = confusion_matrix(y_test, y_pred_lr)

print(f"  Balanced Accuracy: {bal_acc_lr:.4f}")
print(f"  Macro F1:          {f1_lr:.4f}")

# ═══════════════════════════════════════════════════════════════════
# C4: Confusion matrices (side by side)
# ═══════════════════════════════════════════════════════════════════
print("\n=== C4: Confusion matrices ===")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, cm, title in [(axes[0], cm_knn, f'kNN (Bal Acc={bal_acc_knn:.3f}, F1={f1_knn:.3f})'),
                       (axes[1], cm_lr,  f'Logistic (Bal Acc={bal_acc_lr:.3f}, F1={f1_lr:.3f})')]:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(label_names)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = 'white' if val > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('CEBRA Embedding — Classification on Test Set', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_confusion_matrices.png")

# ═══════════════════════════════════════════════════════════════════
# C5: Label centroid distance matrix (test set)
# ═══════════════════════════════════════════════════════════════════
print("\n=== C5: Centroid distance matrix ===")
centroids = np.zeros((len(label_names), X_test_emb.shape[1]))
for cls_id in range(len(label_names)):
    mask = y_test == cls_id
    if mask.any():
        centroids[cls_id] = X_test_emb[mask].mean(axis=0)

dist_matrix = squareform(pdist(centroids, metric='cosine'))

fig, ax = plt.subplots(figsize=(9, 7))
n = len(label_names)
im = ax.imshow(dist_matrix, cmap='RdYlBu_r', aspect='equal')
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{dist_matrix[i, j]:.3f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='black')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(label_names, fontsize=9)
ax.set_title('Centroid Cosine Distance (Test Set)')
fig.colorbar(im, ax=ax, label='Cosine Distance')
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/cebra_centroid_distances.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved cebra_centroid_distances.png")

# Print nearest/farthest pairs
triu = np.triu_indices(len(label_names), k=1)
dists = dist_matrix[triu]
pair_names = [(label_names[i], label_names[j]) for i, j in zip(*triu)]
sorted_pairs = sorted(zip(dists, pair_names))
print("\n  Nearest pairs:")
for d, (a, b) in sorted_pairs[:3]:
    print(f"    {a} <-> {b}: {d:.4f}")
print("  Farthest pairs:")
for d, (a, b) in sorted_pairs[-3:]:
    print(f"    {a} <-> {b}: {d:.4f}")

# ═══════════════════════════════════════════════════════════════════
# C6: Silhouette score (test set)
# ═══════════════════════════════════════════════════════════════════
print("\n=== C6: Silhouette score ===")
# Subsample if test set is large (silhouette is O(n^2))
n_sil = min(len(X_test_emb), 10000)
sil_idx = np.random.choice(len(X_test_emb), n_sil, replace=False)
sil = silhouette_score(X_test_emb[sil_idx], y_test[sil_idx], metric='cosine')
print(f"  Silhouette score (cosine, n={n_sil}): {sil:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("SUMMARY (Test Set)")
print("=" * 50)
print(f"  InfoNCE (normalized):  {test_norm:.3f}")
print(f"  kNN Balanced Acc:      {bal_acc_knn:.4f}")
print(f"  kNN Macro F1:          {f1_knn:.4f}")
print(f"  LogReg Balanced Acc:   {bal_acc_lr:.4f}")
print(f"  LogReg Macro F1:       {f1_lr:.4f}")
print(f"  Silhouette (cosine):   {sil:.4f}")
print("=" * 50)
