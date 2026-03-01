#!/usr/bin/env python3
"""Animate CEBRA embeddings with patient trajectory and CatBoost predictions"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Patch
from scipy.interpolate import splprep, splev
from cebra import CEBRA
from pathlib import Path


def smooth_trajectory(points, smoothing=0.5, num_points=None):
    """Smooth trajectory using spline interpolation"""
    if len(points) < 4:
        return points

    if num_points is None:
        num_points = len(points) * 3

    try:
        n_dims = points.shape[1]
        coords = [points[:, i] for i in range(n_dims)]
        tck, _ = splprep(coords, s=smoothing, k=3)
        u_new = np.linspace(0, 1, num_points)
        smooth = np.column_stack(splev(u_new, tck))
        return smooth
    except Exception:
        return points


def create_animation(embedding, labels, patient_ids, rel_sec, highlight_patient,
                     output_path, duration=120, fps=30, smoothing=0.5,
                     rotation_speed=1.0, catboost_model=None, patient_features=None):
    """
    Create 3D animation of CEBRA embeddings with patient trajectory.

    Args:
        embedding: (n_samples, n_dims) CEBRA embedding (background)
        labels: (n_samples,) cpc_bin labels for coloring
        patient_ids: (n_samples,) patient IDs
        rel_sec: (n_samples,) relative time in seconds
        highlight_patient: patient ID to highlight
        output_path: output MP4 path
        duration: animation duration in seconds
        fps: frames per second
        smoothing: spline smoothing factor
        rotation_speed: degrees per frame for 3D rotation
        catboost_model: trained CatBoost model for predictions (optional)
        patient_features: (n_patient_samples, n_embedding_dims) for CatBoost input (optional)
    """
    use_3d = embedding.shape[1] >= 3

    # Get patient data and sort by time
    patient_mask = patient_ids == highlight_patient
    patient_emb = embedding[patient_mask]
    patient_time = rel_sec[patient_mask]

    sort_idx = np.argsort(patient_time)
    patient_emb = patient_emb[sort_idx]
    patient_time = patient_time[sort_idx]

    if patient_features is not None:
        patient_features = patient_features[sort_idx]

    # Get CatBoost predictions per timestep if available
    predictions = None
    if catboost_model is not None and patient_features is not None:
        predictions = catboost_model.predict_proba(patient_features)[:, 1]

    # Smooth trajectory
    if use_3d:
        smooth_emb = smooth_trajectory(patient_emb[:, :3], smoothing)
    else:
        smooth_emb = smooth_trajectory(patient_emb[:, :2], smoothing)

    n_frames = duration * fps

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    if use_3d:
        ax = fig.add_subplot(111, projection='3d')
        # Background: all points colored by cpc_bin
        colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in labels]
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                   c=colors, s=2, alpha=0.3, zorder=1)

        # Initialize trajectory
        line, = ax.plot([], [], [], 'b-', linewidth=1.5, alpha=0.8, zorder=2)
        point, = ax.plot([], [], [], 'bo', markersize=10,
                         markeredgecolor='white', markeredgewidth=1.5, zorder=3)

        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
    else:
        ax = fig.add_subplot(111)
        colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in labels]
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=colors, s=3, alpha=0.5, zorder=1)

        line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7, zorder=2)
        point, = ax.plot([], [], 'bo', markersize=8,
                         markeredgecolor='white', markeredgewidth=1.5, zorder=3)

        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    ax.set_title(f'CEBRA Embedding - Patient {highlight_patient} Trajectory', fontsize=14)

    # Legend
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.5, label='Good CPC'),
        Patch(facecolor='#e74c3c', alpha=0.5, label='Poor CPC'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label=f'Patient {highlight_patient}')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Info text
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                          verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        if use_3d:
            line.set_data_3d([], [], [])
            point.set_data_3d([], [], [])
        else:
            line.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text

    def animate(frame):
        progress = frame / n_frames
        idx = int(progress * (len(smooth_emb) - 1))

        trail = smooth_emb[:idx + 1]
        if len(trail) > 0:
            if use_3d:
                line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])
                point.set_data_3d([trail[-1, 0]], [trail[-1, 1]], [trail[-1, 2]])

                # Rotate view
                ax.view_init(elev=20, azim=frame * rotation_speed)
            else:
                line.set_data(trail[:, 0], trail[:, 1])
                point.set_data([trail[-1, 0]], [trail[-1, 1]])

        # Time display
        time_val = patient_time[0] + progress * (patient_time[-1] - patient_time[0])
        info = f'Time: {time_val:.1f}s'

        # CatBoost prediction display
        if predictions is not None:
            pred_idx = min(int(progress * (len(predictions) - 1)), len(predictions) - 1)
            prob = predictions[pred_idx]
            pred_label = 'Poor' if prob > 0.5 else 'Good'
            info += f'\nPred: {pred_label} ({prob:.2f})'

        time_text.set_text(info)
        return line, point, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=int(n_frames),
                         interval=1000 / fps, blit=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={'title': 'CEBRA Trajectory'})
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='CEBRA model .pt file')
    parser.add_argument('--data', required=True, help='NPZ file with neural, cpc_bin, patient_names, rel_sec')
    parser.add_argument('--patient', required=True, help='Patient ID to highlight')
    parser.add_argument('--catboost-model', default=None, help='CatBoost .cbm model for predictions')
    parser.add_argument('--output', default='visualizations/trajectory.mp4')
    parser.add_argument('--duration', type=int, default=120)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--smoothing', type=float, default=0.5)
    parser.add_argument('--rotation-speed', type=float, default=1.0)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    model = CEBRA.load(args.model)
    embedding = model.transform(data['neural'])

    # Load CatBoost model if provided
    catboost_clf = None
    patient_features = None
    if args.catboost_model and Path(args.catboost_model).exists():
        from catboost import CatBoostClassifier
        catboost_clf = CatBoostClassifier()
        catboost_clf.load_model(args.catboost_model)

        # Patient embedding features for CatBoost
        patient_mask = data['patient_names'] == args.patient
        patient_features = embedding[patient_mask]

    create_animation(
        embedding=embedding,
        labels=data['cpc_bin'],
        patient_ids=data['patient_names'],
        rel_sec=data['rel_sec'] if 'rel_sec' in data else np.arange(len(embedding)),
        highlight_patient=args.patient,
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        smoothing=args.smoothing,
        rotation_speed=args.rotation_speed,
        catboost_model=catboost_clf,
        patient_features=patient_features
    )
