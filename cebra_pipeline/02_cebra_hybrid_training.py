import cebra
import cebra.models
import cebra.distributions
import numpy as np
import torch
from tqdm import trange

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model'

# ── Config ──────────────────────────────────────────────────────────
OUTPUT_DIM = 3
TIME_OFFSET = 72       # 72 x 5min = 6 hours
BATCH_SIZE = 1024
MAX_ITER = 20000
TEMPERATURE = 0.35
NUM_UNITS = 32
LR = 3e-4

# ── Load preprocessed data ─────────────────────────────────────────
print("Loading preprocessed data...")
d = np.load(f'{DATA_DIR}/cebra_prep_train.npz', allow_pickle=True)
X_train    = d['X'].astype(np.float32)
y_train    = d['y']
pat_starts = d['pat_starts']
pat_ends   = d['pat_ends']

N, D = X_train.shape
X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
pat_starts_t = torch.tensor(pat_starts, dtype=torch.long, device=device)
pat_ends_t   = torch.tensor(pat_ends, dtype=torch.long, device=device)

print(f"  {N} samples, {D} features, {len(np.unique(y_train))} classes")

# ── Build model ────────────────────────────────────────────────────
model = cebra.models.init(
    "offset10-model",
    num_neurons=D,
    num_units=NUM_UNITS,
    num_output=OUTPUT_DIM,
).to(device)

offset = model.get_offset()  # Offset(5, 5)

criterion = cebra.models.criterions.FixedCosineInfoNCE(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITER)

# ── Discrete distribution (behavior objective) ────────────────────
discrete_dist = cebra.distributions.discrete.DiscreteEmpirical(y_tensor.cpu())

# ── Helpers ────────────────────────────────────────────────────────
def expand_index_within_patient(idx):
    """idx: (B,) -> (B, 10) conv windows, clamped within patient."""
    offsets = torch.arange(-offset.left, offset.right, device=idx.device)
    expanded = idx[:, None] + offsets[None, :]
    lo = pat_starts_t[idx][:, None]
    hi = pat_ends_t[idx][:, None] - 1
    return expanded.clamp(min=lo, max=hi)

def get_batch(idx):
    """idx: (B,) -> (B, D, 10) input tensor."""
    windows = expand_index_within_patient(idx)
    return X_tensor[windows].transpose(2, 1)

# ── Training loop ──────────────────────────────────────────────────
print(f"Training hybrid (discrete + bidirectional time) for {MAX_ITER} iterations...")
print(f"  TIME_OFFSET={TIME_OFFSET} ({TIME_OFFSET * 5} min)")
print(f"  TEMPERATURE={TEMPERATURE}, NUM_UNITS={NUM_UNITS}, LR={LR}")

pbar = trange(MAX_ITER)
for step in pbar:
    # 1) Sample references and negatives
    ref_idx = torch.randint(0, N, (BATCH_SIZE,), device=device)
    neg_idx = torch.randint(0, N, (BATCH_SIZE,), device=device)

    # 2) Behavior positives: same discrete label (global)
    ref_labels = y_tensor[ref_idx]
    beh_pos_idx = discrete_dist.sample_conditional(ref_labels.cpu()).to(device)

    # 3) Time positives: bidirectional, clamped within patient
    direction = torch.randint(0, 2, (BATCH_SIZE,), device=device) * 2 - 1  # -1 or +1
    time_pos_idx = ref_idx + direction * TIME_OFFSET
    time_pos_idx = time_pos_idx.clamp(
        min=pat_starts_t[ref_idx],
        max=pat_ends_t[ref_idx] - 1)

    # 4) Forward pass
    ref_emb      = model(get_batch(ref_idx))
    neg_emb      = model(get_batch(neg_idx))
    beh_pos_emb  = model(get_batch(beh_pos_idx))
    time_pos_emb = model(get_batch(time_pos_idx))

    # 5) Two InfoNCE losses
    beh_loss, beh_align, beh_uniform   = criterion(ref_emb, beh_pos_emb, neg_emb)
    time_loss, time_align, time_uniform = criterion(ref_emb, time_pos_emb, neg_emb)
    loss = beh_loss + time_loss

    # 6) Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 200 == 0:
        pbar.set_postfix(
            beh=f"{beh_loss.item():.2f}",
            time=f"{time_loss.item():.2f}",
            total=f"{loss.item():.2f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
        )

# ── Compute train embeddings ──────────────────────────────────────
print("Computing train embeddings...")
model.eval()
train_embeddings = []
with torch.no_grad():
    for i in range(0, N, 4096):
        idx = torch.arange(i, min(i + 4096, N), device=device)
        emb = model(get_batch(idx))
        train_embeddings.append(emb.cpu().numpy())
X_train_emb = np.concatenate(train_embeddings, axis=0)

# ── Compute test embeddings ───────────────────────────────────────
print("Computing test embeddings...")
d_test = np.load(f'{DATA_DIR}/cebra_prep_test.npz', allow_pickle=True)
X_test = d_test['X'].astype(np.float32)
pat_starts_test = d_test['pat_starts']
pat_ends_test   = d_test['pat_ends']
N_test = len(X_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
pat_starts_test_t = torch.tensor(pat_starts_test, dtype=torch.long, device=device)
pat_ends_test_t   = torch.tensor(pat_ends_test, dtype=torch.long, device=device)

def expand_index_test(idx):
    offsets = torch.arange(-offset.left, offset.right, device=idx.device)
    expanded = idx[:, None] + offsets[None, :]
    lo = pat_starts_test_t[idx][:, None]
    hi = pat_ends_test_t[idx][:, None] - 1
    return expanded.clamp(min=lo, max=hi)

def get_batch_test(idx):
    windows = expand_index_test(idx)
    return X_test_tensor[windows].transpose(2, 1)

test_embeddings = []
with torch.no_grad():
    for i in range(0, N_test, 4096):
        idx = torch.arange(i, min(i + 4096, N_test), device=device)
        emb = model(get_batch_test(idx))
        test_embeddings.append(emb.cpu().numpy())
X_test_emb = np.concatenate(test_embeddings, axis=0)

# ── Save ──────────────────────────────────────────────────────────
print("Saving...")
np.savez(f'{DATA_DIR}/cebra_embeddings_train.npz', embedding=X_train_emb)
np.savez(f'{DATA_DIR}/cebra_embeddings_test.npz', embedding=X_test_emb)
torch.save(model.state_dict(), f'{DATA_DIR}/cebra_hybrid_model.pt')

print(f"Done. Train embedding: {X_train_emb.shape}, Test embedding: {X_test_emb.shape}")
