#!/usr/bin/env python3
"""Run the full CEBRA pipeline end-to-end, or individual stages."""
import argparse
import gc
import json
import logging
import sys
from datetime import datetime
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def _build_labels(data, label_keys):
    """Build label array from config label string (supports comma-separated)."""
    keys = [k.strip() for k in label_keys.split(',')]
    if len(keys) == 1:
        return data[keys[0]]
    return np.column_stack([data[k] for k in keys])


def _paths(run_name):
    """Return all output paths for a given run name."""
    return {
        'train_npz': f'data/{run_name}_train.npz',
        'test_npz': f'data/{run_name}_test.npz',
        'model': f'models/{run_name}_cebra.pt',
        'catboost': f'evaluation/{run_name}_catboost.cbm',
        'results': f'evaluation/{run_name}_results.json',
        'train_plot': f'visualizations/{run_name}_train.png',
        'test_plot': f'visualizations/{run_name}_test.png',
        'comparison_plot': f'visualizations/{run_name}_train_test.png',
    }


def run_prepare(config, paths):
    """Stage 1: Data preparation"""
    log.info("=== 1. Data Preparation ===")
    from scripts.prepare_data import prepare_data_streaming, save_splits

    data_cfg = config['data']
    train_data, test_data = prepare_data_streaming(
        neural_dir=data_cfg['neural_dir'],
        labels_path=data_cfg['labels_path'],
        test_size=data_cfg.get('test_size', 0.2),
        seed=data_cfg.get('seed', 42),
        nan_strategy=data_cfg.get('nan_strategy', 'mean'),
        n_workers=data_cfg.get('n_workers', 4)
    )

    # Save with run name
    Path('data').mkdir(exist_ok=True)
    np.savez_compressed(paths['train_npz'],
                        neural=train_data['neural'], cpc=train_data['cpc'],
                        cpc_bin=train_data['cpc_bin'], patient_ids=train_data['patient_ids'],
                        patient_names=train_data['patient_names'],
                        rel_sec=train_data['rel_sec'], feature_names=train_data['feature_names'])
    np.savez_compressed(paths['test_npz'],
                        neural=test_data['neural'], cpc=test_data['cpc'],
                        cpc_bin=test_data['cpc_bin'], patient_ids=test_data['patient_ids'],
                        patient_names=test_data['patient_names'],
                        rel_sec=test_data['rel_sec'], feature_names=test_data['feature_names'])

    log.info(f"Train: {train_data['neural'].shape}, Test: {test_data['neural'].shape}")
    log.info(f"Saved: {paths['train_npz']}, {paths['test_npz']}")
    del train_data, test_data
    gc.collect()


def run_train(config, paths):
    """Stage 2: CEBRA training"""
    log.info("=== 2. Training CEBRA ===")
    from scripts.train import train_cebra

    train_cfg = config['training']
    label_keys = train_cfg.get('labels', 'cpc_bin')

    data = np.load(paths['train_npz'], allow_pickle=True)
    labels = _build_labels(data, label_keys)

    model = train_cebra(
        data['neural'], labels, paths['model'],
        config=train_cfg,
        seed=train_cfg.get('seed', 42)
    )
    del data, labels, model
    gc.collect()


def run_predict(config, paths):
    """Stage 3: CatBoost prediction"""
    log.info("=== 3. CatBoost Prediction ===")
    from scripts.predict import predict_model

    train_cfg = config['training']
    pred_cfg = config.get('prediction', {})
    label_key = train_cfg.get('labels', 'cpc_bin')

    if ',' in label_key:
        label_key = label_key.split(',')[0].strip()
        log.info(f"Using first label for classification: {label_key}")

    train_npz = np.load(paths['train_npz'], allow_pickle=True)
    test_npz = np.load(paths['test_npz'], allow_pickle=True)

    output_dir = str(Path(paths['catboost']).parent)
    results, clf = predict_model(
        paths['model'], train_npz, test_npz, label_key,
        catboost_config=pred_cfg, output_dir=output_dir
    )

    # Rename catboost model to run-specific name
    default_cbm = f'{output_dir}/catboost_model.cbm'
    if Path(default_cbm).exists() and default_cbm != paths['catboost']:
        Path(default_cbm).rename(paths['catboost'])
    results['catboost_model_path'] = paths['catboost']

    Path(paths['results']).parent.mkdir(exist_ok=True)
    with open(paths['results'], 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Test accuracy: {results['test_accuracy']:.4f}, AUC: {results['test_auc']:.4f}")
    log.info(f"Saved: {paths['results']}, {paths['catboost']}")
    del train_npz, test_npz, clf
    gc.collect()


def run_visualize(config, paths):
    """Stage 4: Visualization"""
    log.info("=== 4. Visualization ===")
    from scripts.visualize import plot_embedding, plot_train_test
    from scripts.train import transform_batched
    from cebra import CEBRA

    train_cfg = config['training']
    label_key = train_cfg.get('labels', 'cpc_bin')
    if ',' in label_key:
        label_key = label_key.split(',')[0].strip()

    model = CEBRA.load(paths['model'])
    train_npz = np.load(paths['train_npz'], allow_pickle=True)
    test_npz = np.load(paths['test_npz'], allow_pickle=True)

    train_emb = transform_batched(model, train_npz['neural'])
    test_emb = transform_batched(model, test_npz['neural'])

    plot_embedding(train_emb, train_npz[label_key],
                   'Train Embedding', paths['train_plot'])
    plot_embedding(test_emb, test_npz[label_key],
                   'Test Embedding', paths['test_plot'])
    plot_train_test(train_emb, test_emb,
                    train_npz[label_key], test_npz[label_key],
                    paths['comparison_plot'])
    del model, train_emb, test_emb, train_npz, test_npz
    gc.collect()


def run_animate(config, paths):
    """Stage 5: Animation"""
    log.info("=== 5. Animation ===")
    from scripts.animate import create_animation
    from scripts.train import transform_batched
    from cebra import CEBRA

    data_cfg = config['data']
    train_cfg = config['training']
    anim_cfg = config.get('animation', {})
    label_key = train_cfg.get('labels', 'cpc_bin')
    if ',' in label_key:
        label_key = label_key.split(',')[0].strip()

    model = CEBRA.load(paths['model'])
    test_npz = np.load(paths['test_npz'], allow_pickle=True)
    test_emb = transform_batched(model, test_npz['neural'])

    # Pick a random test patient
    test_patients = np.unique(test_npz['patient_names'])
    np.random.seed(data_cfg.get('seed', 42))
    patient = np.random.choice(test_patients)
    log.info(f"Animating patient: {patient}")

    # Load CatBoost if available
    catboost_clf = None
    patient_features = None
    if Path(paths['catboost']).exists():
        from catboost import CatBoostClassifier
        catboost_clf = CatBoostClassifier()
        catboost_clf.load_model(paths['catboost'])
        patient_mask = test_npz['patient_names'] == patient
        patient_features = test_emb[patient_mask]

    run_name = Path(paths['model']).stem.replace('_cebra', '')
    output_path = f'visualizations/{run_name}_trajectory_{patient}.mp4'

    create_animation(
        embedding=test_emb,
        labels=test_npz[label_key],
        patient_ids=test_npz['patient_names'],
        rel_sec=test_npz['rel_sec'],
        highlight_patient=patient,
        output_path=output_path,
        duration=anim_cfg.get('duration', 30),
        fps=anim_cfg.get('fps', 15),
        smoothing=anim_cfg.get('smoothing', 0.5),
        rotation_speed=anim_cfg.get('rotation_speed', 1.0),
        max_bg_points=anim_cfg.get('max_bg_points', 50000),
        catboost_model=catboost_clf,
        patient_features=patient_features
    )

    del model, test_emb, test_npz
    gc.collect()


STAGES = {
    'prepare': run_prepare,
    'train': run_train,
    'predict': run_predict,
    'visualize': run_visualize,
    'animate': run_animate,
}


def main(config_path='config.json', stages=None, run_name=None):
    with open(config_path) as f:
        config = json.load(f)

    if run_name is None:
        run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    paths = _paths(run_name)
    log.info(f"Run: {run_name}")
    log.info(f"Config: {config_path}")

    # Save config snapshot for this run
    Path('evaluation').mkdir(exist_ok=True)
    with open(f'evaluation/{run_name}_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    if stages is None:
        stages = list(STAGES.keys())

    for stage in stages:
        if stage not in STAGES:
            log.error(f"Unknown stage: {stage}. Available: {list(STAGES.keys())}")
            sys.exit(1)
        STAGES[stage](config, paths)

    log.info(f"=== Pipeline Complete: {run_name} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CEBRA pipeline (all or specific stages)')
    parser.add_argument('--config', default='config.json', help='Config file')
    parser.add_argument('--stages', nargs='+',
                        choices=list(STAGES.keys()),
                        default=None,
                        help='Stages to run (default: all)')
    parser.add_argument('--run-name', default=None,
                        help='Name for this run (default: run_YYYYMMDD_HHMMSS)')
    args = parser.parse_args()
    main(args.config, args.stages, args.run_name)
