"""
08_combo_grid.py — 5x2 (combos x train/test) grid of CEBRA spheres.

Produces 3 HTMLs, each colored by a different feature:
  1. EEG label
  2. Binary CPC (good vs poor)
  3. Time (hours from each patient's recording start)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _constants import (CLASS_NAMES, CLASS_COLORS, DISPLAY_ORDER,
                        OUTCOME_GOOD_RGB, OUTCOME_BAD_RGB, OUT_DIR)

COMBOS         = ['combo_i', 'combo_ii', 'combo_iii', 'combo_iv', 'combo_v']
SUBSAMPLE      = 15000     # per cell; None = all
NORMALIZE_TIME = True      # True: 0-1 fractional progress per patient
                           # False: absolute hours from each patient's start
N_ROWS    = len(COMBOS)
N_COLS    = 2              # train, test

# ── Load all combos ────────────────────────────────────────────────
def load(tag, split):
    p = np.load(f'{OUT_DIR}/cebra_prep_{tag}_{split}.npz', allow_pickle=True)
    e = np.load(f'{OUT_DIR}/cebra_embeddings_{tag}_{split}.npz')['embedding']
    return dict(emb=e,
                y=p['predictions'].astype(int),
                cpc_b=p['cpc_binary'].astype(int),
                pid=p['patient_ids'],
                times=p['times'].astype(float))

def per_patient_time(times, pids, normalize=False):
    out = np.zeros_like(times, dtype=np.float64)
    for pid in np.unique(pids):
        m = pids == pid
        rel = times[m] - times[m].min()
        if normalize:
            span = max(rel.max(), 1.0)            # avoid div-by-zero
            out[m] = rel / span
        else:
            out[m] = rel / 3600.0                 # absolute hours
    return out

def maybe_subsample(d):
    n = len(d['emb'])
    if SUBSAMPLE is None or n <= SUBSAMPLE:
        return d
    rs = np.random.RandomState(0)
    idx = rs.choice(n, SUBSAMPLE, replace=False)
    return {k: (v[idx] if hasattr(v, '__len__') else v) for k, v in d.items()}

print('Loading combos...')
data = {}
for tag in COMBOS:
    try:
        tr, te = load(tag, 'train'), load(tag, 'test')
    except FileNotFoundError as e:
        print(f"  [skip] {tag}: {e}")
        continue
    tr['hours'] = per_patient_time(tr['times'], tr['pid'], normalize=NORMALIZE_TIME)
    te['hours'] = per_patient_time(te['times'], te['pid'], normalize=NORMALIZE_TIME)
    data[tag] = (maybe_subsample(tr), maybe_subsample(te))
    print(f"  loaded {tag}")

if not data:
    raise SystemExit("No combos loaded; check OUT_DIR and that 02 has been run.")

# ── Subplot scaffold ───────────────────────────────────────────────
def make_grid(title):
    specs = [[{'type': 'scatter3d'}, {'type': 'scatter3d'}] for _ in COMBOS]
    titles = []
    for tag in COMBOS:
        titles += [f'{tag} — Train', f'{tag} — Test']
    fig = make_subplots(rows=N_ROWS, cols=N_COLS, specs=specs,
                        subplot_titles=titles,
                        horizontal_spacing=0.02, vertical_spacing=0.04)
    axis = dict(showbackground=False, showticklabels=False, title='',
                showgrid=False, zeroline=False)
    scene_kw = dict(bgcolor='white', xaxis=axis, yaxis=axis, zaxis=axis)
    n_scenes = N_ROWS * N_COLS
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        paper_bgcolor='white', font=dict(color='black', family='Arial'),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(size=11),
                    bordercolor='lightgray', borderwidth=1,
                    x=1.01, y=0.99, xanchor='left', yanchor='top',
                    itemsizing='constant', traceorder='normal'),
        margin=dict(l=0, r=180, t=60, b=0),
        height=380 * N_ROWS, width=1500,
    )
    for k in range(n_scenes):
        key = 'scene' if k == 0 else f'scene{k+1}'
        fig.layout[key].update(**scene_kw)
    return fig

# ═══════════════════════════════════════════════════════════════════
# View 1: EEG label
# ═══════════════════════════════════════════════════════════════════
fig = make_grid('CEBRA Embedding by EEG state — combos x train/test')
shown = set()
for r, tag in enumerate(COMBOS, start=1):
    if tag not in data:
        continue
    tr, te = data[tag]
    for col, d in enumerate([tr, te], start=1):
        for c in DISPLAY_ORDER:
            m = d['y'] == c
            if not m.any():
                continue
            show = CLASS_NAMES[c] not in shown
            fig.add_trace(go.Scatter3d(
                x=d['emb'][m, 0], y=d['emb'][m, 1], z=d['emb'][m, 2],
                mode='markers',
                marker=dict(size=1.4, color=CLASS_COLORS[c], opacity=0.35),
                name=CLASS_NAMES[c], legendgroup=CLASS_NAMES[c],
                showlegend=show, hoverinfo='skip',
            ), row=r, col=col)
            if show:
                shown.add(CLASS_NAMES[c])
out1 = f'{OUT_DIR}/cebra_grid_eeg.html'
fig.write_html(out1); print(f'Saved {out1}')

# ═══════════════════════════════════════════════════════════════════
# View 2: binary CPC
# ═══════════════════════════════════════════════════════════════════
fig = make_grid('CEBRA Embedding by CPC outcome — combos x train/test')
shown = set()
for r, tag in enumerate(COMBOS, start=1):
    if tag not in data:
        continue
    tr, te = data[tag]
    for col, d in enumerate([tr, te], start=1):
        for label, color, name in [(0, OUTCOME_GOOD_RGB, 'Good (CPC<=2)'),
                                    (1, OUTCOME_BAD_RGB,  'Poor (CPC>=3)')]:
            m = d['cpc_b'] == label
            if not m.any():
                continue
            show = name not in shown
            fig.add_trace(go.Scatter3d(
                x=d['emb'][m, 0], y=d['emb'][m, 1], z=d['emb'][m, 2],
                mode='markers',
                marker=dict(size=1.4, color=color, opacity=0.30),
                name=name, legendgroup=name,
                showlegend=show, hoverinfo='skip',
            ), row=r, col=col)
            if show:
                shown.add(name)
out2 = f'{OUT_DIR}/cebra_grid_cpc.html'
fig.write_html(out2); print(f'Saved {out2}')

# ═══════════════════════════════════════════════════════════════════
# View 3: time (hours from patient start)
# ═══════════════════════════════════════════════════════════════════
if NORMALIZE_TIME:
    hmax = 1.0
    cbar_title = 'Fractional progress<br>(0=start, 1=end)'
    title3 = 'CEBRA Embedding by time (normalized per patient) — combos x train/test'
else:
    hmax = max(max(float(tr['hours'].max()), float(te['hours'].max()))
               for tr, te in data.values())
    cbar_title = 'Hours from<br>patient start'
    title3 = 'CEBRA Embedding by time (hours from patient start) — combos x train/test'

fig = make_grid(title3)
first = True
for r, tag in enumerate(COMBOS, start=1):
    if tag not in data:
        continue
    tr, te = data[tag]
    for col, d in enumerate([tr, te], start=1):
        fig.add_trace(go.Scatter3d(
            x=d['emb'][:, 0], y=d['emb'][:, 1], z=d['emb'][:, 2],
            mode='markers',
            marker=dict(size=1.4, color=d['hours'], colorscale='Plasma',
                        cmin=0, cmax=hmax, opacity=0.35,
                        showscale=first,
                        colorbar=dict(title=cbar_title,
                                      x=1.01, len=0.6, thickness=14)),
            showlegend=False, hoverinfo='skip',
        ), row=r, col=col)
        first = False
out3 = f'{OUT_DIR}/cebra_grid_time.html'
fig.write_html(out3); print(f'Saved {out3}')
