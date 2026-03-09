"""Plotly 3D/2D visualization for CEBRA embeddings."""
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.interpolate import splprep, splev


def plot_embeddings(emb, labels, title='CEBRA Embedding', max_points=50000, seed=42):
    """Scatter plot of embeddings colored by CPC label."""
    if len(emb) > max_points:
        idx = np.random.RandomState(seed).choice(len(emb), max_points, replace=False)
        emb, labels = emb[idx], labels[idx]

    colors = np.where(labels == 0, 'rgba(46,204,113,0.5)', 'rgba(231,76,60,0.5)').tolist()

    if emb.shape[1] >= 3:
        fig = go.Figure(go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode='markers', marker=dict(size=1.5, color=colors),
            text=[f'{"Good" if l==0 else "Poor"}' for l in labels],
            hoverinfo='text', showlegend=False,
        ))
        fig.update_layout(scene=dict(
            xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'))
    else:
        fig = go.Figure(go.Scatter(
            x=emb[:, 0], y=emb[:, 1],
            mode='markers', marker=dict(size=2, color=colors),
            hoverinfo='skip', showlegend=False,
        ))
        fig.update_layout(xaxis_title='Dim 1', yaxis_title='Dim 2')

    fig.update_layout(title=title, width=900, height=700)
    return fig


def smooth_trajectory(points, n_output=150, smoothing_factor=1.0):
    """Aggregate and smooth trajectory to n_output points via spline."""
    if len(points) < 4:
        return points
    # Subsample to reduce noise before splining
    step = max(1, len(points) // (n_output * 3))
    sub = points[::step]
    if len(sub) < 4:
        return sub
    try:
        coords = [sub[:, i] for i in range(sub.shape[1])]
        tck, _ = splprep(coords, s=smoothing_factor * len(sub), k=3)
        u = np.linspace(0, 1, n_output)
        return np.column_stack(splev(u, tck))
    except Exception:
        return sub


def plot_trajectory(train_emb, train_labels, patient_emb, patient_name,
                    max_bg=50000, seed=42, smoothing=1.0, n_traj_points=150):
    """Plot smoothed test patient trajectory over training embeddings."""
    # Background
    if len(train_emb) > max_bg:
        idx = np.random.RandomState(seed).choice(len(train_emb), max_bg, replace=False)
        bg_emb, bg_lab = train_emb[idx], train_labels[idx]
    else:
        bg_emb, bg_lab = train_emb, train_labels

    is_3d = bg_emb.shape[1] >= 3
    colors = np.where(bg_lab == 0, 'rgba(46,204,113,0.2)', 'rgba(231,76,60,0.2)').tolist()

    fig = go.Figure()

    # Training background
    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=bg_emb[:, 0], y=bg_emb[:, 1], z=bg_emb[:, 2],
            mode='markers', marker=dict(size=1.5, color=colors),
            name='Training', hoverinfo='skip',
        ))
    else:
        fig.add_trace(go.Scatter(
            x=bg_emb[:, 0], y=bg_emb[:, 1],
            mode='markers', marker=dict(size=2, color=colors),
            name='Training', hoverinfo='skip',
        ))

    # Smoothed trajectory
    dims = 3 if is_3d else 2
    traj = smooth_trajectory(patient_emb[:, :dims], n_traj_points, smoothing)

    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode='lines', line=dict(color='blue', width=4),
            name=f'Patient {patient_name}',
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode='markers', marker=dict(size=8, color='cyan', symbol='diamond'),
            name='Start',
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
            mode='markers', marker=dict(size=8, color='yellow', symbol='diamond'),
            name='End',
        ))
        fig.update_layout(scene=dict(
            xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'))
    else:
        fig.add_trace(go.Scatter(
            x=traj[:, 0], y=traj[:, 1],
            mode='lines', line=dict(color='blue', width=2),
            name=f'Patient {patient_name}',
        ))
        fig.add_trace(go.Scatter(
            x=[traj[0, 0]], y=[traj[0, 1]],
            mode='markers', marker=dict(size=10, color='cyan', symbol='diamond'),
            name='Start',
        ))
        fig.add_trace(go.Scatter(
            x=[traj[-1, 0]], y=[traj[-1, 1]],
            mode='markers', marker=dict(size=10, color='yellow', symbol='diamond'),
            name='End',
        ))
        fig.update_layout(xaxis_title='Dim 1', yaxis_title='Dim 2')

    fig.update_layout(
        title=f'Patient {patient_name} Trajectory over Training Embeddings',
        width=1000, height=750,
    )
    return fig


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
    print(f"Saved: {path}")
