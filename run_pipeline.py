#!/usr/bin/env python3
"""Run the full CEBRA pipeline end-to-end, or individual stages."""
import argparse
import gc
import json
import logging
import sys
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


def run_prepare(config):
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
    save_splits(train_data, test_data)
    log.info(f"Train: {train_data['neural'].shape}, Test: {test_data['neural'].shape}")
    del train_data, test_data
    gc.collect()


def run_train(config):
    """Stage 2: CEBRA training"""
    log.info("=== 2. Training CEBRA ===")
    from scripts.train import train_cebra

    train_cfg = config['training']
    label_keys = train_cfg.get('labels', 'cpc_bin')

    data = np.load('data/train.npz', allow_pickle=True)
    labels = _build_labels(data, label_keys)

    model_path = 'models/cebra_model.pt'
    model = train_cebra(
        data['neural'], labels, model_path,
        config=train_cfg,
        seed=train_cfg.get('seed', 42)
    )
    del data, labels, model
    gc.collect()


def run_predict(config):
    """Stage 3: CatBoost prediction"""
    log.info("=== 3. CatBoost Prediction ===")
    from scripts.predict import predict_model

    train_cfg = config['training']
    pred_cfg = config.get('prediction', {})
    label_key = train_cfg.get('labels', 'cpc_bin')

    # CatBoost needs a single label for classification
    if ',' in label_key:
        label_key = label_key.split(',')[0].strip()
        log.info(f"Using first label for classification: {label_key}")

    train_npz = np.load('data/train.npz', allow_pickle=True)
    test_npz = np.load('data/test.npz', allow_pickle=True)

    results, clf = predict_model(
        'models/cebra_model.pt', train_npz, test_npz, label_key,
        catboost_config=pred_cfg
    )

    results_path = 'evaluation/results.json'
    Path('evaluation').mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Test accuracy: {results['test_accuracy']:.4f}, AUC: {results['test_auc']:.4f}")
    del train_npz, test_npz, clf
    gc.collect()


def run_visualize(config):
    """Stage 4: Visualization"""
    log.info("=== 4. Visualization ===")
    from scripts.visualize import plot_embedding, plot_train_test
    from cebra import CEBRA

    train_cfg = config['training']
    label_key = train_cfg.get('labels', 'cpc_bin')
    if ',' in label_key:
        label_key = label_key.split(',')[0].strip()

    model = CEBRA.load('models/cebra_model.pt')
    train_npz = np.load('data/train.npz', allow_pickle=True)
    test_npz = np.load('data/test.npz', allow_pickle=True)

    from scripts.train import transform_batched
    train_emb = transform_batched(model, train_npz['neural'])
    test_emb = transform_batched(model, test_npz['neural'])

    plot_embedding(train_emb, train_npz[label_key],
                   'Train Embedding', 'visualizations/cebra_model_train.png')
    plot_embedding(test_emb, test_npz[label_key],
                   'Test Embedding', 'visualizations/cebra_model_test.png')
    plot_train_test(train_emb, test_emb,
                    train_npz[label_key], test_npz[label_key],
                    'visualizations/cebra_model_train_test.png')
    del model, train_emb, test_emb, train_npz, test_npz
    gc.collect()


def run_animate(config):
    """Stage 5: Animation"""
    log.info("=== 5. Animation ===")
    from scripts.animate import create_animation
    from cebra import CEBRA

    data_cfg = config['data']
    train_cfg = config['training']
    anim_cfg = config.get('animation', {})
    label_key = train_cfg.get('labels', 'cpc_bin')
    if ',' in label_key:
        label_key = label_key.split(',')[0].strip()

    from scripts.train import transform_batched
    model = CEBRA.load('models/cebra_model.pt')
    test_npz = np.load('data/test.npz', allow_pickle=True)
    test_emb = transform_batched(model, test_npz['neural'])

    # Pick a random test patient
    test_patients = np.unique(test_npz['patient_names'])
    np.random.seed(data_cfg.get('seed', 42))
    patient = np.random.choice(test_patients)
    log.info(f"Animating patient: {patient}")

    # Patient embedding for CatBoost predictions
    catboost_clf = None
    patient_features = None
    catboost_path = 'evaluation/catboost_model.cbm'
    if Path(catboost_path).exists():
        from catboost import CatBoostClassifier
        catboost_clf = CatBoostClassifier()
        catboost_clf.load_model(catboost_path)
        patient_mask = test_npz['patient_names'] == patient
        patient_features = test_emb[patient_mask]

    create_animation(
        embedding=test_emb,
        labels=test_npz[label_key],
        patient_ids=test_npz['patient_names'],
        rel_sec=test_npz['rel_sec'],
        highlight_patient=patient,
        output_path=f'visualizations/trajectory_{patient}.mp4',
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


def main(config_path='config.json', stages=None):
    with open(config_path) as f:
        config = json.load(f)

    if stages is None:
        stages = list(STAGES.keys())

    for stage in stages:
        if stage not in STAGES:
            log.error(f"Unknown stage: {stage}. Available: {list(STAGES.keys())}")
            sys.exit(1)
        STAGES[stage](config)

    log.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CEBRA pipeline (all or specific stages)')
    parser.add_argument('--config', default='config.json', help='Config file')
    parser.add_argument('--stages', nargs='+',
                        choices=list(STAGES.keys()),
                        default=None,
                        help='Stages to run (default: all). e.g. --stages train predict visualize')
    args = parser.parse_args()
    main(args.config, args.stages)
