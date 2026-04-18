import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model'

# ── Load ────────────────────────────────────────────────────────────
train_prep = np.load(f'{DATA_DIR}/cebra_prep_train.npz', allow_pickle=True)
test_prep  = np.load(f'{DATA_DIR}/cebra_prep_test.npz', allow_pickle=True)
train_emb  = np.load(f'{DATA_DIR}/cebra_embeddings_train.npz')['embedding']
test_emb   = np.load(f'{DATA_DIR}/cebra_embeddings_test.npz')['embedding']

y_train = train_prep['y']  # already 0-7
y_test  = test_prep['y']

class_names = {
    0: 'Seizure', 1: 'LPD', 2: 'GPD', 3: 'LRDA',
    4: 'GRDA', 5: 'Burst Supp.', 6: 'Continuous', 7: 'Discontinuous',
}
class_colors = {
    0: '#E63946',  # Seizure - red
    1: '#2A9D8F',  # LPD - teal
    2: '#E9C46A',  # GPD - gold
    3: '#7209B7',  # LRDA - purple
    4: '#0077B6',  # GRDA - blue
    5: '#F72585',  # Burst Supp. - pink
    6: '#8D99AE',  # Continuous - grey
    7: '#6A4C3C',  # Discontinuous - brown
}

# ── Build side-by-side figure ──────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=['Train Embedding', 'Test Embedding'],
    horizontal_spacing=0.02,
)

for col, (emb, labels, split) in enumerate([
    (train_emb, y_train, 'train'),
    (test_emb, y_test, 'test'),
], start=1):
    scene = 'scene' if col == 1 else 'scene2'

    # Class clouds
    for cls_id in sorted(class_names.keys()):
        mask = labels == cls_id
        if not mask.any():
            continue
        name = class_names[cls_id]
        fig.add_trace(go.Scatter3d(
            x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
            mode='markers',
            marker=dict(size=1.5, color=class_colors[cls_id], opacity=0.3),
            name=name,
            legendgroup=name,
            showlegend=(col == 1),  # legend only from train
            hoverinfo='skip',
        ), row=1, col=col)

    # Floating labels at centroids
    center = emb.mean(axis=0)
    for cls_id, name in class_names.items():
        mask = labels == cls_id
        if not mask.any():
            continue
        centroid = emb[mask].mean(axis=0)
        d = centroid - center
        d = d / (np.linalg.norm(d) + 1e-8)
        pos = centroid + d * 1.5
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='text', text=[f'<b>{name}</b>'],
            textfont=dict(size=12, color=class_colors[cls_id]),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=col)

    # Scene styling
    axis = dict(showbackground=False, showticklabels=False,
                title='', showgrid=False, zeroline=False)
    fig.update_layout(**{
        scene: dict(bgcolor='white', xaxis=axis, yaxis=axis, zaxis=axis)
    })

fig.update_layout(
    title=dict(text='CEBRA Hybrid Embeddings — Train vs Test', font=dict(size=20), x=0.5),
    paper_bgcolor='white',
    font=dict(color='black', family='Arial'),
    legend=dict(
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='lightgray', borderwidth=1,
        x=0.01, y=0.99, xanchor='left', yanchor='top',
        itemsizing='constant',
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    height=600, width=1400,
)

fig.write_html(f'{DATA_DIR}/cebra_train_vs_test.html')
print(f"Saved cebra_train_vs_test.html")
