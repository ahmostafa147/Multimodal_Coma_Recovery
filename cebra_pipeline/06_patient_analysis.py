"""
06_patient_analysis.py — full CEBRA analysis for one patient.

Edit RUN_TAG and PID at the top.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from collections import Counter

from _constants import (CLASS_NAMES, CLASS_COLORS, CLASS_COLOR_LIST,
                        DISPLAY_ORDER, DISPLAY_NAMES, N_CLASSES, BIN_SEC,
                        OUTCOME_GOOD_RGB, OUTCOME_BAD_RGB, OUT_DIR)

# ── CONFIG ──────────────────────────────────────────────────────────
RUN_TAG  = 'combo_i'
PID      = 'ICARE_0647'           # change here
WINDOW   = 15                     # bins per SLERP waypoint
SLERP_N  = 10
TOP_K    = 50                     # samples per prototype landmark

OUT_ROOT = f'{OUT_DIR}/patient_analysis_{RUN_TAG}'
OUT      = os.path.join(OUT_ROOT, PID)
os.makedirs(OUT, exist_ok=True)

STATE_COLORS = CLASS_COLOR_LIST   # data-encoding order

# ── Load splits ─────────────────────────────────────────────────────
def load(split):
    p = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_{split}.npz', allow_pickle=True)
    e = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_{split}.npz')['embedding']
    return dict(
        X=p['X'], y=p['predictions'].astype(int), pid=p['patient_ids'],
        times=p['times'], cpc=p['cpc_scores'], dur=p['bin_durations'],
        emb=e, acts=p['activations'],
    )

train = load('train')
test  = load('test')

split, d = (None, None)
for n, dd in [('train', train), ('test', test)]:
    if PID in dd['pid']:
        split, d = n, dd; break
if split is None:
    print(f"Patient {PID} not found in train or test.")
    raise SystemExit(1)

mask  = d['pid'] == PID
p_emb = d['emb'][mask]
p_y   = d['y'][mask].astype(int)
p_dur = d['dur'][mask].astype(float)
cpc   = float(d['cpc'][mask][0]) if mask.sum() else float('nan')
N     = int(mask.sum())
print(f"Found {PID} in {split} split — {N} bins, CPC={cpc}")

# midpoint time in hours (uniform BIN_SEC == 300 unless dur says otherwise)
t_h = (np.cumsum(p_dur) - p_dur / 2) / 3600.0

# centroids from train
centroids = np.zeros((N_CLASSES, train['emb'].shape[1]))
for c in range(N_CLASSES):
    m = train['y'] == c
    if m.sum() > 0:
        centroids[c] = train['emb'][m].mean(axis=0)

# ── METRICS ─────────────────────────────────────────────────────────
counts = Counter(p_y.tolist())
trans  = int((p_y[1:] != p_y[:-1]).sum()) if N > 1 else 0
sojourns = {c: [] for c in range(N_CLASSES)}
if N > 0:
    s, run = p_y[0], 1
    for i in range(1, N):
        if p_y[i] == s:
            run += 1
        else:
            sojourns[s].append(run); s, run = p_y[i], 1
    sojourns[s].append(run)

outcome = ('Good (CPC<=2)' if cpc <= 2 else
           'Poor (CPC>=3)' if cpc <= 5 else 'Unknown')

lines = [
    f"Patient: {PID}",
    f"Run tag: {RUN_TAG}",
    f"Split: {split}",
    f"CPC: {cpc}    Outcome: {outcome}",
    f"Bins: {N}    Duration: {p_dur.sum()/3600:.2f} h",
    f"Dominant state: {CLASS_NAMES[max(counts, key=counts.get)] if counts else 'N/A'}",
    f"Transitions: {trans}    Rate: {trans/max(N,1):.3f}/bin",
    "",
    "State distribution:",
]
for c in DISPLAY_ORDER:
    pct = 100 * counts.get(c, 0) / max(N, 1)
    lines.append(f"  {CLASS_NAMES[c]:<14s} {counts.get(c,0):5d}  ({pct:5.1f}%)")
lines += ["", "Max sojourn per state:"]
for c in DISPLAY_ORDER:
    mx = max(sojourns[c]) if sojourns[c] else 0
    lines.append(f"  {CLASS_NAMES[c]:<14s} {mx:4d} bins  (~{mx*BIN_SEC/60:5.1f} min)")

with open(os.path.join(OUT, 'metrics.txt'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved metrics.txt')

# ── helpers ─────────────────────────────────────────────────────────
def slerp(p0, p1, t):
    """Spherical linear interpolation between two 3D vectors."""
    n0 = np.linalg.norm(p0); n1 = np.linalg.norm(p1)
    if n0 < 1e-12 or n1 < 1e-12:
        return p0 * (1 - t) + p1 * t
    u0, u1 = p0 / n0, p1 / n1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-6:
        return p0 * (1 - t) + p1 * t
    s = np.sin(omega)
    a = np.sin((1 - t) * omega) / s
    b = np.sin(t * omega) / s
    r = (n0 * (1 - t)) + (n1 * t)
    return (a * u0 + b * u1) * r

def small_circle_on_sphere(points, n_pts=120, percentile=68, cap=np.pi*0.35):
    c     = points.mean(axis=0)
    c_dir = c / (np.linalg.norm(c) + 1e-12)
    r     = np.linalg.norm(points, axis=1).mean()
    pts_u = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-12)
    ang   = np.arccos(np.clip(pts_u @ c_dir, -1, 1))
    alpha = min(np.percentile(ang, percentile), cap)
    t = np.array([1.0, 0, 0]) if abs(c_dir[0]) < 0.9 else np.array([0, 1.0, 0])
    x = t - (t @ c_dir) * c_dir; x /= np.linalg.norm(x) + 1e-12
    y = np.cross(c_dir, x)
    th = np.linspace(0, 2*np.pi, n_pts)
    circ = ((np.cos(th)[:,None]*x + np.sin(th)[:,None]*y) * np.sin(alpha)
            + np.cos(alpha) * c_dir) * r
    label_pos = c_dir * r * 1.32
    return circ, label_pos

def outcome_scale(outcome):
    if outcome == 'good':
        return [[0,'rgb(15,35,85)'], [1, OUTCOME_GOOD_RGB]]
    if outcome == 'bad':
        return [[0,'rgb(70,25,0)'], [1, OUTCOME_BAD_RGB]]
    return [[0,'rgb(40,40,40)'], [1,'rgb(200,200,200)']]

# ── prototype landmarks ─────────────────────────────────────────────
acts = train['acts']
n_proto = acts.shape[1]
proto_emb   = np.zeros((n_proto, 3))
proto_state = np.zeros(n_proto, dtype=int)
for p in range(n_proto):
    top = np.argpartition(acts[:, p], -TOP_K)[-TOP_K:]
    proto_emb[p]   = train['emb'][top].mean(axis=0)
    proto_state[p] = Counter(train['y'][top].astype(int).tolist()).most_common(1)[0][0]

# ── train CPC outcome subsampling ───────────────────────────────────
cpc_tr = train['cpc'].astype(float)
good_tr = cpc_tr <= 2
bad_tr  = (cpc_tr >= 3) & (cpc_tr <= 5)
def subsample(idx_mask, n=8000):
    idx = np.where(idx_mask)[0]
    if len(idx) > n:
        idx = np.random.RandomState(0).choice(idx, n, replace=False)
    return idx
g_idx, b_idx = subsample(good_tr), subsample(bad_tr)

# ── SLERP trajectory of patient ─────────────────────────────────────
out_key = 'good' if cpc <= 2 else 'bad' if cpc <= 5 else 'unknown'
rep_p, rep_t, rep_l, rep_state = [], [], [], []
smooth_p, smooth_t = np.empty((0, 3)), np.empty(0)
if N >= 2:
    for ws in range(0, N, WINDOW):
        we = min(ws + WINDOW, N)
        wp, wc = p_y[ws:we], p_emb[ws:we]
        cls, cnt = np.unique(wp, return_counts=True)
        win = int(cls[np.argmax(cnt)])
        last = np.where(wp == win)[0][-1]
        rep_p.append(wc[last]); rep_t.append((ws + last)/max(N-1, 1))
        rep_l.append(CLASS_NAMES[win]); rep_state.append(win)
    rep_p, rep_t = np.array(rep_p), np.array(rep_t)

    smooth_p_list, smooth_t_list = [], []
    for i in range(len(rep_p) - 1):
        for s in range(SLERP_N):
            tt = s / SLERP_N
            smooth_p_list.append(slerp(rep_p[i], rep_p[i+1], tt))
            smooth_t_list.append(rep_t[i] + tt*(rep_t[i+1]-rep_t[i]))
    smooth_p_list.append(rep_p[-1]); smooth_t_list.append(rep_t[-1])
    smooth_p, smooth_t = np.array(smooth_p_list), np.array(smooth_t_list)

# ── 1. UNIFIED 3D PLOT ──────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=train['emb'][g_idx,0], y=train['emb'][g_idx,1], z=train['emb'][g_idx,2],
    mode='markers',
    marker=dict(size=1.6, color=OUTCOME_GOOD_RGB, opacity=0.30),
    name='Good outcome (CPC<=2)'))
fig.add_trace(go.Scatter3d(
    x=train['emb'][b_idx,0], y=train['emb'][b_idx,1], z=train['emb'][b_idx,2],
    mode='markers',
    marker=dict(size=1.6, color=OUTCOME_BAD_RGB, opacity=0.30),
    name='Poor outcome (CPC>=3)'))

# state boundaries — iterate display order so legend reads in spec order
for c in DISPLAY_ORDER:
    mask_c = train['y'] == c
    if mask_c.sum() < 20:
        continue
    circ, lab_pos = small_circle_on_sphere(train['emb'][mask_c])
    fig.add_trace(go.Scatter3d(
        x=circ[:,0], y=circ[:,1], z=circ[:,2], mode='lines',
        line=dict(color=CLASS_COLORS[c], width=6),
        name=f'{CLASS_NAMES[c]} boundary',
        legendgroup=CLASS_NAMES[c]))
    fig.add_trace(go.Scatter3d(
        x=[lab_pos[0]], y=[lab_pos[1]], z=[lab_pos[2]],
        mode='text', text=[f'<b>{CLASS_NAMES[c]}</b>'],
        textfont=dict(size=15, color=CLASS_COLORS[c]),
        legendgroup=CLASS_NAMES[c], showlegend=False, hoverinfo='skip'))

for c in DISPLAY_ORDER:
    pm = proto_state == c
    if pm.sum() == 0:
        continue
    fig.add_trace(go.Scatter3d(
        x=proto_emb[pm,0], y=proto_emb[pm,1], z=proto_emb[pm,2],
        mode='markers',
        marker=dict(size=8, color=CLASS_COLORS[c], symbol='cross',
                    line=dict(color='white', width=1)),
        name=f'Proto: {CLASS_NAMES[c]}',
        legendgroup=CLASS_NAMES[c], showlegend=True,
        hovertemplate=f'{CLASS_NAMES[c]} prototype<extra></extra>'))

if N >= 2:
    fig.add_trace(go.Scatter3d(
        x=smooth_p[:,0], y=smooth_p[:,1], z=smooth_p[:,2],
        mode='lines', line=dict(color='black', width=14),
        name=f'{PID} (halo)', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(
        x=smooth_p[:,0], y=smooth_p[:,1], z=smooth_p[:,2],
        mode='lines',
        line=dict(color=smooth_t, colorscale=outcome_scale(out_key),
                  width=10, cmin=0, cmax=1),
        name=f'{PID} trajectory ({out_key})'))
    fig.add_trace(go.Scatter3d(
        x=rep_p[:,0], y=rep_p[:,1], z=rep_p[:,2],
        mode='markers',
        marker=dict(size=7,
                    color=[CLASS_COLORS[s] for s in rep_state],
                    line=dict(color='black', width=2)),
        text=rep_l,
        hovertemplate='%{text}<extra></extra>',
        name=f'{PID} waypoints'))
    fig.add_trace(go.Scatter3d(
        x=[rep_p[0,0]], y=[rep_p[0,1]], z=[rep_p[0,2]],
        mode='markers+text',
        marker=dict(size=12, color='lime', symbol='diamond',
                    line=dict(color='black', width=2)),
        text=['START'], textposition='top center',
        textfont=dict(size=13, color='lime'),
        name='Start'))
    fig.add_trace(go.Scatter3d(
        x=[rep_p[-1,0]], y=[rep_p[-1,1]], z=[rep_p[-1,2]],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='diamond',
                    line=dict(color='black', width=2)),
        text=['END'], textposition='top center',
        textfont=dict(size=13, color='red'),
        name='End'))

# symmetric range over everything we drew
extent_pts = [train['emb'][g_idx], train['emb'][b_idx], proto_emb]
for c in DISPLAY_ORDER:
    mask_c = train['y'] == c
    if mask_c.sum() < 20:
        continue
    _, lp = small_circle_on_sphere(train['emb'][mask_c])
    extent_pts.append(lp[None, :])
if N >= 2:
    extent_pts.append(smooth_p)
all_pts = np.concatenate(extent_pts, axis=0)
half = np.abs(all_pts).max() * 1.10
rng_sym = (-half, half)

axis_style = dict(gridcolor='rgb(140,140,160)',
                  zerolinecolor='rgb(160,160,180)',
                  showbackground=True, backgroundcolor='rgb(28,32,48)',
                  color='white')
fig.update_layout(
    title=f'CEBRA overview — {PID} (CPC={cpc}, {out_key}) [{RUN_TAG}]',
    scene=dict(
        bgcolor='rgb(20,25,40)', aspectmode='cube',
        xaxis=dict(title='Dim 1', range=rng_sym, **axis_style),
        yaxis=dict(title='Dim 2', range=rng_sym, **axis_style),
        zaxis=dict(title='Dim 3', range=rng_sym, **axis_style),
    ),
    paper_bgcolor='rgb(20,25,40)', font_color='white',
    legend=dict(bgcolor='rgba(0,0,0,0.55)', font=dict(size=11),
                itemsizing='constant', traceorder='normal'),
)
fig.write_html(os.path.join(OUT, '01_unified_overview.html'))
print('Saved 01_unified_overview.html')

# ── 4. state timeline ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
for ypos, c in enumerate(DISPLAY_ORDER):
    m = p_y == c
    if m.sum() == 0:
        continue
    ax.scatter(t_h[m], np.full(m.sum(), ypos), c=CLASS_COLORS[c],
               s=20, marker='|', label=CLASS_NAMES[c])
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(DISPLAY_NAMES)
ax.invert_yaxis(); ax.set_xlabel('Time (hours)')
ax.set_title(f'{PID} — state timeline')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '04_state_timeline.png'), dpi=150)
plt.close()
print('Saved 04_state_timeline.png')

# ── 5. state distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(DISPLAY_NAMES,
       [counts.get(c, 0) for c in DISPLAY_ORDER],
       color=[CLASS_COLORS[c] for c in DISPLAY_ORDER])
ax.set_ylabel('# bins'); ax.set_title(f'{PID} — state distribution')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(OUT, '05_state_distribution.png'), dpi=150)
plt.close()
print('Saved 05_state_distribution.png')

# ── 6. centroid similarity heatmap (display order on y-axis) ───────
pn = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-12)
cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
sims = pn @ cn.T                          # (N, 8) data order
sims_disp = sims[:, DISPLAY_ORDER]

fig = plt.figure(figsize=(14, 5.5))
gs  = GridSpec(2, 2, height_ratios=[6, 0.5], width_ratios=[40, 1],
               hspace=0.08, wspace=0.02)
ax_h  = fig.add_subplot(gs[0, 0])
ax_cb = fig.add_subplot(gs[0, 1])
ax_b  = fig.add_subplot(gs[1, 0], sharex=ax_h)

im = ax_h.imshow(sims_disp.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                 extent=[t_h[0], t_h[-1], N_CLASSES - 0.5, -0.5])
ax_h.set_yticks(range(N_CLASSES)); ax_h.set_yticklabels(DISPLAY_NAMES)
ax_h.set_title(f'{PID} — cosine similarity to state centroids')
ax_h.tick_params(labelbottom=False)
plt.colorbar(im, cax=ax_cb, label='cosine sim')

strip = np.array([to_rgb(STATE_COLORS[v]) for v in p_y]).reshape(1, -1, 3)
ax_b.imshow(strip, aspect='auto', extent=[t_h[0], t_h[-1], 0, 1])
ax_b.set_yticks([0.5]); ax_b.set_yticklabels(['True label'])
ax_b.set_xlabel('Time (hours)')

patches = [Patch(facecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in DISPLAY_ORDER]
ax_b.legend(handles=patches, loc='upper center',
            bbox_to_anchor=(0.5, -1.6), ncol=N_CLASSES,
            frameon=False, fontsize=9, handlelength=1.2)

plt.savefig(os.path.join(OUT, '06_centroid_similarity.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved 06_centroid_similarity.png')

# ── 7. transition matrix (display order) ──────────────────────────
T = np.zeros((N_CLASSES, N_CLASSES))
for i in range(N - 1):
    T[p_y[i], p_y[i+1]] += 1
row_sum = T.sum(axis=1, keepdims=True)
Tn = np.divide(T, row_sum, out=np.zeros_like(T), where=row_sum > 0)
Tn_disp = Tn[np.ix_(DISPLAY_ORDER, DISPLAY_ORDER)]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(Tn_disp, cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set_xticks(np.arange(N_CLASSES)); ax.set_yticks(np.arange(N_CLASSES))
ax.set_xticklabels(DISPLAY_NAMES); ax.set_yticklabels(DISPLAY_NAMES)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, f'{Tn_disp[i, j]:.2f}', ha='center', va='center',
                color='white' if Tn_disp[i, j] > 0.5 else 'black')
ax.set_xlabel('Next'); ax.set_ylabel('Current')
ax.set_title(f'{PID} — transition probabilities')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '07_transition_matrix.png'), dpi=150)
plt.close()
print('Saved 07_transition_matrix.png')

# ── 8. sojourn boxplot (display order) ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
data = [sojourns[c] if sojourns[c] else [0] for c in DISPLAY_ORDER]
bp = ax.boxplot(data, labels=DISPLAY_NAMES, patch_artist=True)
for patch, c in zip(bp['boxes'], DISPLAY_ORDER):
    patch.set_facecolor(CLASS_COLORS[c])
ax.set_ylabel('Sojourn (bins)'); ax.set_title(f'{PID} — sojourn per state')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(OUT, '08_sojourn.png'), dpi=150)
plt.close()
print('Saved 08_sojourn.png')

print(f"\nDone. {len(os.listdir(OUT))} files in {OUT}")
