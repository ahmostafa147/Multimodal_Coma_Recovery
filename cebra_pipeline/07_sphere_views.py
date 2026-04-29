"""
07_sphere_views.py — three side-by-side train/test CEBRA sphere views:
  1. colored by EEG label (Seizure, LPD, ...)
  2. colored by binary CPC (good vs poor)
  3. colored by hours from each patient's recording start
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _constants import (CLASS_NAMES, CLASS_COLORS, DISPLAY_ORDER,
                        OUTCOME_GOOD_RGB, OUTCOME_BAD_RGB, OUT_DIR)

RUN_TAG        = 'combo_i'
SUBSAMPLE      = 30000     # cap per panel for browser performance; None = all
NORMALIZE_TIME = True      # True: 0-1 fractional progress per patient
                           # False: absolute hours from each patient's start

# ── Load ────────────────────────────────────────────────────────────
def load(split):
    p = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_{split}.npz', allow_pickle=True)
    e = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_{split}.npz')['embedding']
    return dict(emb=e,
                y=p['predictions'].astype(int),
                cpc_b=p['cpc_binary'].astype(int),
                pid=p['patient_ids'],
                times=p['times'].astype(float))

train = load('train')
test  = load('test')

def per_patient_time(times, pids, normalize=False):
    out = np.zeros_like(times, dtype=np.float64)
    for pid in np.unique(pids):
        m = pids == pid
        rel = times[m] - times[m].min()
        if normalize:
            out[m] = rel / max(rel.max(), 1.0)        # 0..1 per patient
        else:
            out[m] = rel / 3600.0                     # absolute hours
    return out

train['hours'] = per_patient_time(train['times'], train['pid'], normalize=NORMALIZE_TIME)
test ['hours'] = per_patient_time(test ['times'], test ['pid'], normalize=NORMALIZE_TIME)

def maybe_subsample(d):
    n = len(d['emb'])
    if SUBSAMPLE is None or n <= SUBSAMPLE:
        return d
    rs = np.random.RandomState(0)
    idx = rs.choice(n, SUBSAMPLE, replace=False)
    return {k: (v[idx] if hasattr(v, '__len__') else v) for k, v in d.items()}

tr = maybe_subsample(train)
te = maybe_subsample(test)

# ── Layout helper ──────────────────────────────────────────────────
def make_fig(title):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['Train', 'Test'],
        horizontal_spacing=0.02,
    )
    axis = dict(showbackground=False, showticklabels=False, title='',
                showgrid=False, zeroline=False)
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        paper_bgcolor='white', font=dict(color='black', family='Arial'),
        scene =dict(bgcolor='white', xaxis=axis, yaxis=axis, zaxis=axis),
        scene2=dict(bgcolor='white', xaxis=axis, yaxis=axis, zaxis=axis),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(size=11),
                    bordercolor='lightgray', borderwidth=1,
                    x=0.01, y=0.99, xanchor='left', yanchor='top',
                    itemsizing='constant', traceorder='normal'),
        margin=dict(l=0, r=0, t=60, b=0),
        height=600, width=1400,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════
# View 1: EEG label
# ═══════════════════════════════════════════════════════════════════
fig = make_fig(f'CEBRA Embedding by EEG state — Train vs Test ({RUN_TAG})')
for col, d in enumerate([tr, te], start=1):
    for c in DISPLAY_ORDER:
        m = d['y'] == c
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=d['emb'][m, 0], y=d['emb'][m, 1], z=d['emb'][m, 2],
            mode='markers',
            marker=dict(size=1.5, color=CLASS_COLORS[c], opacity=0.35),
            name=CLASS_NAMES[c], legendgroup=CLASS_NAMES[c],
            showlegend=(col == 1), hoverinfo='skip',
        ), row=1, col=col)
out1 = f'{OUT_DIR}/cebra_sphere_eeg_{RUN_TAG}.html'
fig.write_html(out1)
print(f'Saved {out1}')

# ═══════════════════════════════════════════════════════════════════
# View 2: binary CPC
# ═══════════════════════════════════════════════════════════════════
fig = make_fig(f'CEBRA Embedding by CPC outcome — Train vs Test ({RUN_TAG})')
for col, d in enumerate([tr, te], start=1):
    for label, color, name in [(0, OUTCOME_GOOD_RGB, 'Good (CPC<=2)'),
                                (1, OUTCOME_BAD_RGB,  'Poor (CPC>=3)')]:
        m = d['cpc_b'] == label
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=d['emb'][m, 0], y=d['emb'][m, 1], z=d['emb'][m, 2],
            mode='markers',
            marker=dict(size=1.5, color=color, opacity=0.30),
            name=name, legendgroup=name,
            showlegend=(col == 1), hoverinfo='skip',
        ), row=1, col=col)
out2 = f'{OUT_DIR}/cebra_sphere_cpc_{RUN_TAG}.html'
fig.write_html(out2)
print(f'Saved {out2}')

# ═══════════════════════════════════════════════════════════════════
# View 3: time (hours from each patient's recording start)
# ═══════════════════════════════════════════════════════════════════
if NORMALIZE_TIME:
    hmax = 1.0
    cbar_title = 'Fractional progress<br>(0=start, 1=end)'
    title3 = f'CEBRA Embedding by time (normalized per patient) — Train vs Test ({RUN_TAG})'
else:
    hmax = max(float(tr['hours'].max()), float(te['hours'].max()))
    cbar_title = 'Hours from<br>patient start'
    title3 = f'CEBRA Embedding by time within patient — Train vs Test ({RUN_TAG})'

fig = make_fig(title3)
for col, d in enumerate([tr, te], start=1):
    fig.add_trace(go.Scatter3d(
        x=d['emb'][:, 0], y=d['emb'][:, 1], z=d['emb'][:, 2],
        mode='markers',
        marker=dict(size=1.5, color=d['hours'], colorscale='Plasma',
                    cmin=0, cmax=hmax, opacity=0.35,
                    showscale=(col == 1),
                    colorbar=dict(title=cbar_title,
                                  x=1.02, len=0.6, thickness=14)),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=col)
out3 = f'{OUT_DIR}/cebra_sphere_time_{RUN_TAG}.html'
fig.write_html(out3)
print(f'Saved {out3}')
