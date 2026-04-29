"""
04_classification_evaluation.py — embedding evaluation.

Two prediction tasks fit on the train embedding, scored on test:
  T1: 8-class predictions   -> Macro F1, confusion matrix
  T2: binary CPC outcome    -> AUC, Sensitivity@FPR<=5%, Brier

Both tasks fit two classifiers and compared: RandomForest, CatBoost.

Embedding-level metrics: InfoNCE (held-out contrastive), Silhouette.
Plus: centroid cosine-distance matrix on test set.
"""
import numpy as np
np.random.seed(42)
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, confusion_matrix,
                             silhouette_score, roc_auc_score, roc_curve,
                             brier_score_loss)
from scipy.spatial.distance import pdist, squareform
from catboost import CatBoostClassifier

from _constants import CLASS_NAMES, DISPLAY_ORDER, DISPLAY_NAMES, N_CLASSES, OUT_DIR

RUN_TAG = 'combo_i'

# ── Load ────────────────────────────────────────────────────────────
print(f"=== Loading (RUN_TAG={RUN_TAG}) ===")
train_prep = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_train.npz', allow_pickle=True)
test_prep  = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_test.npz',  allow_pickle=True)
X_train_emb = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_train.npz')['embedding']
X_test_emb  = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_test.npz')['embedding']

y_train      = train_prep['predictions'].astype(int)
y_test       = test_prep['predictions'].astype(int)
cpc_train    = train_prep['cpc_binary'].astype(int)
cpc_test     = test_prep['cpc_binary'].astype(int)

print(f"  Train: {X_train_emb.shape}  Test: {X_test_emb.shape}")
print(f"  CPC binary: train good/poor = {(cpc_train==0).sum()}/{(cpc_train==1).sum()}, "
      f"test = {(cpc_test==0).sum()}/{(cpc_test==1).sum()}")

# ═══════════════════════════════════════════════════════════════════
# C1: Held-out contrastive loss
# ═══════════════════════════════════════════════════════════════════
print("\n=== C1: Held-out InfoNCE ===")

def compute_infonce(emb, labels, n_samples=2048, temperature=1.0, n_runs=10):
    losses = []
    for _ in range(n_runs):
        idx = np.random.choice(len(emb), n_samples * 2, replace=False)
        ref_idx, neg_idx = idx[:n_samples], idx[n_samples:]
        pos_idx = np.zeros(n_samples, dtype=int)
        for i, ri in enumerate(ref_idx):
            same = np.where(labels == labels[ri])[0]
            pos_idx[i] = np.random.choice(same)
        ref = torch.tensor(emb[ref_idx], dtype=torch.float32)
        pos = torch.tensor(emb[pos_idx], dtype=torch.float32)
        neg = torch.tensor(emb[neg_idx], dtype=torch.float32)
        pos_sim = torch.einsum('ni,ni->n', ref, pos) / temperature
        neg_sim = torch.einsum('ni,mi->nm', ref, neg) / temperature
        with torch.no_grad():
            c, _ = neg_sim.max(dim=1, keepdim=True)
        pos_sim = pos_sim - c.squeeze(1)
        neg_sim = neg_sim - c
        loss = ((-pos_sim).mean() + torch.logsumexp(neg_sim, dim=1).mean()).item()
        losses.append(loss)
    raw = float(np.mean(losses))
    return raw, raw - float(np.log(n_samples))

train_raw, train_norm = compute_infonce(X_train_emb, y_train)
test_raw,  test_norm  = compute_infonce(X_test_emb,  y_test)
print(f"  Train: {train_raw:.3f}  (normalized {train_norm:.3f})")
print(f"  Test:  {test_raw:.3f}  (normalized {test_norm:.3f})")

# ═══════════════════════════════════════════════════════════════════
# C2: Silhouette (on test, by predictions label)
# ═══════════════════════════════════════════════════════════════════
print("\n=== C2: Silhouette (cosine, predictions labels) ===")
n_sil = min(len(X_test_emb), 10000)
sil_idx = np.random.choice(len(X_test_emb), n_sil, replace=False)
silhouette = silhouette_score(X_test_emb[sil_idx], y_test[sil_idx], metric='cosine')
print(f"  Silhouette (n={n_sil}): {silhouette:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════════
def make_rf(seed=42, **kw):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed, **kw)

def make_cb(seed=42, **kw):
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                              random_seed=seed, verbose=False, **kw)

def sens_at_fpr(y_true, y_score, fpr_target=0.05):
    """Sensitivity (TPR) at the strictest threshold with FPR <= fpr_target.
    (FPR <= 5% == specificity >= 95%, the standard high-specificity operating point.)"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= fpr_target
    return float(tpr[valid].max()) if valid.any() else 0.0

# ═══════════════════════════════════════════════════════════════════
# T1: 8-class predictions — RF vs CatBoost
# ═══════════════════════════════════════════════════════════════════
print("\n=== T1: 8-class predictions ===")
results_t1 = {}
for name, model in [('RF', make_rf()), ('CatBoost', make_cb())]:
    model.fit(X_train_emb, y_train)
    y_pred = model.predict(X_test_emb).astype(int).reshape(-1)
    bal = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    cm  = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))
    results_t1[name] = dict(y_pred=y_pred, bal=bal, f1=f1m, cm=cm)
    print(f"  {name:8s}  Macro F1={f1m:.4f}  BalAcc={bal:.4f}")

# Confusion matrices side-by-side, reordered to display order
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (name, r) in zip(axes, results_t1.items()):
    cm = r['cm'][np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    n = N_CLASSES
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if v > 0.4 else 'black')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{name}: Macro F1={r["f1"]:.3f}, BalAcc={r["bal"]:.3f}')
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle(f'T1 — 8-class predictions ({RUN_TAG})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_t1_confusion_{RUN_TAG}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_t1_confusion_{RUN_TAG}.png")

# ═══════════════════════════════════════════════════════════════════
# T2: binary CPC — RF vs CatBoost
# ═══════════════════════════════════════════════════════════════════
print("\n=== T2: Binary CPC (good vs poor) ===")
results_t2 = {}
for name, model in [('RF', make_rf()), ('CatBoost', make_cb())]:
    model.fit(X_train_emb, cpc_train)
    proba = model.predict_proba(X_test_emb)[:, 1]
    auc   = roc_auc_score(cpc_test, proba)
    sens  = sens_at_fpr(cpc_test, proba, fpr_target=0.05)
    brier = brier_score_loss(cpc_test, proba)
    results_t2[name] = dict(proba=proba, auc=auc, sens=sens, brier=brier)
    print(f"  {name:8s}  AUC={auc:.4f}  Sens@FPR<=5%={sens:.4f}  Brier={brier:.4f}")

# ROC + Brier comparison plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, r in results_t2.items():
    fpr, tpr, _ = roc_curve(cpc_test, r['proba'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
axes[0].axvline(0.05, ls=':', color='black', alpha=0.5, label='FPR=5%')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC — binary CPC')
axes[0].legend(loc='lower right'); axes[0].grid(alpha=0.3)

names = list(results_t2.keys())
auc_vals  = [results_t2[n]['auc']   for n in names]
sens_vals = [results_t2[n]['sens']  for n in names]
brier_vals= [results_t2[n]['brier'] for n in names]
x = np.arange(len(names)); w = 0.25
axes[1].bar(x - w, auc_vals,   w, label='AUC',           color='#3AA4F3')
axes[1].bar(x,     sens_vals,  w, label='Sens@FPR<=5%',  color='#FFA219')
axes[1].bar(x + w, brier_vals, w, label='Brier',         color='#7C6494')
axes[1].set_xticks(x); axes[1].set_xticklabels(names)
axes[1].set_ylabel('Score'); axes[1].set_title('CPC metrics')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

plt.suptitle(f'T2 — Binary CPC ({RUN_TAG})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_t2_cpc_{RUN_TAG}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_t2_cpc_{RUN_TAG}.png")

# ═══════════════════════════════════════════════════════════════════
# C5: Centroid cosine-distance matrix (test set, display order)
# ═══════════════════════════════════════════════════════════════════
print("\n=== C5: Centroid distance matrix ===")
centroids = np.zeros((N_CLASSES, X_test_emb.shape[1]))
for c in range(N_CLASSES):
    m = y_test == c
    if m.any():
        centroids[c] = X_test_emb[m].mean(axis=0)

dist = squareform(pdist(centroids, metric='cosine'))
dist_disp = dist[np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(dist_disp, cmap='RdYlBu_r', aspect='equal')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, f'{dist_disp[i, j]:.3f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='black')
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(DISPLAY_NAMES, fontsize=9)
ax.set_title(f'Centroid Cosine Distance (Test, {RUN_TAG})')
fig.colorbar(im, ax=ax, label='Cosine Distance')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cebra_centroid_distances_{RUN_TAG}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved cebra_centroid_distances_{RUN_TAG}.png")

# Nearest / farthest pairs (in original encoding, names mapped via DISPLAY)
triu = np.triu_indices(N_CLASSES, k=1)
dists = dist[triu]
pair_names = [(CLASS_NAMES[i], CLASS_NAMES[j]) for i, j in zip(*triu)]
sorted_pairs = sorted(zip(dists, pair_names))
print("  Nearest pairs:")
for d, (a, b) in sorted_pairs[:3]:
    print(f"    {a} <-> {b}: {d:.4f}")
print("  Farthest pairs:")
for d, (a, b) in sorted_pairs[-3:]:
    print(f"    {a} <-> {b}: {d:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"SUMMARY ({RUN_TAG})")
print("=" * 60)
print(f"  InfoNCE (test, normalized):  {test_norm:.3f}")
print(f"  Silhouette (test, cosine):   {silhouette:.4f}")
print("  Predictions (8-class):")
for name, r in results_t1.items():
    print(f"    {name:8s}  Macro F1={r['f1']:.4f}  BalAcc={r['bal']:.4f}")
print("  CPC (binary):")
for name, r in results_t2.items():
    print(f"    {name:8s}  AUC={r['auc']:.4f}  Sens@FPR<=5%={r['sens']:.4f}  Brier={r['brier']:.4f}")
print("=" * 60)
