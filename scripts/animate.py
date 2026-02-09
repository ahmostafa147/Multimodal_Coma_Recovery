#!/usr/bin/env python3
"""Animate CEBRA embeddings with patient trajectory"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
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
        tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothing, k=3)
        u_new = np.linspace(0, 1, num_points)
        smooth = np.column_stack(splev(u_new, tck))
        return smooth
    except:
        return points


def create_animation(embedding, labels, patient_ids, rel_sec, highlight_patient,
                     output_path, duration=120, fps=30, smoothing=0.5):
    """
    Create animation of CEBRA embeddings with patient trajectory

    Args:
        embedding: (n_samples, n_dims) CEBRA embedding
        labels: (n_samples,) cpc_bin labels for coloring
        patient_ids: (n_samples,) patient IDs
        rel_sec: (n_samples,) relative time in seconds
        highlight_patient: Patient ID to highlight
        output_path: Output MP4 path
        duration: Animation duration in seconds
        fps: Frames per second
        smoothing: Spline smoothing factor (0=interpolate, higher=smoother)
    """
    # Get patient mask and sort by time
    patient_mask = patient_ids == highlight_patient
    patient_emb = embedding[patient_mask]
    patient_time = rel_sec[patient_mask]

    # Sort by time
    sort_idx = np.argsort(patient_time)
    patient_emb = patient_emb[sort_idx]
    patient_time = patient_time[sort_idx]

    # Smooth trajectory
    smooth_emb = smooth_trajectory(patient_emb[:, :2], smoothing)
    n_frames = duration * fps

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: all points colored by cpc_bin
    colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in labels]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5, alpha=0.3)

    # Initialize trajectory line and point
    line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.8)
    point, = ax.plot([], [], 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)

    # Labels
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(f'CEBRA Embedding - Patient {highlight_patient} Trajectory', fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.5, label='Good CPC'),
        Patch(facecolor='#e74c3c', alpha=0.5, label='Poor CPC'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label=f'Patient {highlight_patient}')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text

    def animate(frame):
        # Calculate progress through trajectory
        progress = frame / n_frames
        idx = int(progress * (len(smooth_emb) - 1))

        # Growing trail
        trail = smooth_emb[:idx+1]
        if len(trail) > 0:
            line.set_data(trail[:, 0], trail[:, 1])
            point.set_data([trail[-1, 0]], [trail[-1, 1]])

        # Interpolate time for display
        time_progress = patient_time[0] + progress * (patient_time[-1] - patient_time[0])
        time_text.set_text(f'Time: {time_progress:.1f}s')

        return line, point, time_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=int(n_frames),
                        interval=1000/fps, blit=True)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={'title': 'CEBRA Trajectory'})
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True, help='NPZ file with neural, cpc_bin, patient_names, rel_sec')
    parser.add_argument('--patient', required=True, help='Patient ID to highlight')
    parser.add_argument('--output', default='visualizations/trajectory.mp4')
    parser.add_argument('--duration', type=int, default=120, help='Animation duration in seconds')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--smoothing', type=float, default=0.5)
    args = parser.parse_args()

    # Load data
    data = np.load(args.data, allow_pickle=True)
    model = CEBRA.load(args.model)
    embedding = model.transform(data['neural'])

    create_animation(
        embedding=embedding,
        labels=data['cpc_bin'],
        patient_ids=data['patient_names'],
        rel_sec=data['rel_sec'] if 'rel_sec' in data else np.arange(len(embedding)),
        highlight_patient=args.patient,
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        smoothing=args.smoothing
    )
