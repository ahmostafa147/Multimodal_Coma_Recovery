"""03_embedding_visualization.py — train vs test 3D embedding plot."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _constants import CLASS_NAMES, CLASS_COLORS, DISPLAY_ORDER, OUT_DIR

RUN_TAG = 'combo_i'

# ── Load ────────────────────────────────────────────────────────────
train_prep = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_train.npz', allow_pickle=True)
test_prep  = np.load(f'{OUT_DIR}/cebra_prep_{RUN_TAG}_test.npz',  allow_pickle=True)
train_emb  = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_train.npz')['embedding']
test_emb   = np.load(f'{OUT_DIR}/cebra_embeddings_{RUN_TAG}_test.npz')['embedding']

y_train = train_prep['predictions'].astype(int)
y_test  = test_prep['predictions'].astype(int)

# ── Build side-by-side figure ──────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=['Train Embedding', 'Test Embedding'],
    horizontal_spacing=0.02,
)

for col, (emb, labels) in enumerate([(train_emb, y_train),
                                      (test_emb,  y_test)], start=1):
    scene = 'scene' if col == 1 else 'scene2'

    # Class clouds — iterate in DISPLAY_ORDER so legend reads top-down per spec
    for cls_id in DISPLAY_ORDER:
        mask = labels == cls_id
        if not mask.any():
            continue
        name = CLASS_NAMES[cls_id]
        fig.add_trace(go.Scatter3d(
            x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
            mode='markers',
            marker=dict(size=1.5, color=CLASS_COLORS[cls_id], opacity=0.3),
            name=name,
            legendgroup=name,
            showlegend=(col == 1),
            hoverinfo='skip',
        ), row=1, col=col)

    # Floating labels at centroids (offset outward from global center)
    center = emb.mean(axis=0)
    for cls_id in DISPLAY_ORDER:
        mask = labels == cls_id
        if not mask.any():
            continue
        centroid = emb[mask].mean(axis=0)
        d = centroid - center
        d = d / (np.linalg.norm(d) + 1e-8)
        pos = centroid + d * 1.5
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='text', text=[f'<b>{CLASS_NAMES[cls_id]}</b>'],
            textfont=dict(size=12, color=CLASS_COLORS[cls_id]),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=col)

    axis = dict(showbackground=False, showticklabels=False,
                title='', showgrid=False, zeroline=False)
    fig.update_layout(**{
        scene: dict(bgcolor='white', xaxis=axis, yaxis=axis, zaxis=axis)
    })

fig.update_layout(
    title=dict(text=f'CEBRA Embeddings — Train vs Test ({RUN_TAG})',
               font=dict(size=20), x=0.5),
    paper_bgcolor='white',
    font=dict(color='black', family='Arial'),
    legend=dict(
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='lightgray', borderwidth=1,
        x=0.01, y=0.99, xanchor='left', yanchor='top',
        itemsizing='constant',
        traceorder='normal',
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=600, width=1400,
)

out_path = f'{OUT_DIR}/cebra_train_vs_test_{RUN_TAG}.html'
fig.write_html(out_path)
print(f"Saved {out_path}")
