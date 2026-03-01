#!/usr/bin/env python3
"""Run the full CEBRA pipeline end-to-end"""
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


def main(config_path='config.json'):
    with open(config_path) as f:
        config = json.load(f)

    data_cfg = config['data']
    train_cfg = config['training']
    pred_cfg = config.get('prediction', {})
    anim_cfg = config.get('animation', {})

    # === 1. Data Preparation ===
    log.info("=== 1. Data Preparation ===")
    from scripts.prepare_data import prepare_data_streaming, save_splits

    train_data, test_data = prepare_data_streaming(
        neural_dir=data_cfg['neural_dir'],
        labels_path=data_cfg['labels_path'],
        test_size=data_cfg.get('test_size', 0.2),
        seed=data_cfg.get('seed', 42),
        nan_strategy=data_cfg.get('nan_strategy', 'mean'),
        n_workers=data_cfg.get('n_workers', 4)
    )
    save_splits(train_data, test_data)

    # Keep only what we need
    train_neural = train_data['neural']
    train_labels = train_data['cpc_bin']
    test_neural = test_data['neural']
    test_labels = test_data['cpc_bin']

    log.info(f"Train: {train_neural.shape}, Test: {test_neural.shape}")

    # === 2. Train CEBRA ===
    log.info("=== 2. Training CEBRA ===")
    from scripts.train import train_cebra

    model_path = 'models/cebra_model.pt'
    model, train_embedding = train_cebra(
        train_neural, train_labels, model_path,
        config=train_cfg,
        seed=train_cfg.get('seed', 42)
    )
    del train_neural
    gc.collect()

    # === 3. Predict with CatBoost ===
    log.info("=== 3. CatBoost Prediction ===")
    from scripts.predict import predict_model

    train_npz = np.load('data/train.npz', allow_pickle=True)
    test_npz = np.load('data/test.npz', allow_pickle=True)

    results, catboost_clf = predict_model(
        model_path, train_npz, test_npz, 'cpc_bin',
        catboost_config=pred_cfg
    )

    results_path = 'evaluation/results.json'
    Path('evaluation').mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Test accuracy: {results['test_accuracy']:.4f}, AUC: {results['test_auc']:.4f}")

    # === 4. Visualize ===
    log.info("=== 4. Visualization ===")
    from scripts.visualize import plot_embedding, plot_train_test

    test_emb = model.transform(test_npz['neural'])

    plot_embedding(train_embedding, train_labels,
                   'Train Embedding', 'visualizations/cebra_model_train.png')
    plot_embedding(test_emb, test_npz['cpc_bin'],
                   'Test Embedding', 'visualizations/cebra_model_test.png')
    plot_train_test(train_embedding, test_emb,
                    train_labels, test_npz['cpc_bin'],
                    'visualizations/cebra_model_train_test.png')
    del train_embedding
    gc.collect()

    # === 5. Animate ===
    log.info("=== 5. Animation ===")
    from scripts.animate import create_animation

    # Pick a random test patient
    test_patients = np.unique(test_npz['patient_names'])
    np.random.seed(data_cfg.get('seed', 42))
    patient = np.random.choice(test_patients)
    log.info(f"Animating patient: {patient}")

    # Get patient embedding for CatBoost
    patient_mask = test_npz['patient_names'] == patient
    patient_features = test_emb[patient_mask]

    create_animation(
        embedding=test_emb,
        labels=test_npz['cpc_bin'],
        patient_ids=test_npz['patient_names'],
        rel_sec=test_npz['rel_sec'],
        highlight_patient=patient,
        output_path=f'visualizations/trajectory_{patient}.mp4',
        duration=anim_cfg.get('duration', 120),
        fps=anim_cfg.get('fps', 30),
        smoothing=anim_cfg.get('smoothing', 0.5),
        rotation_speed=anim_cfg.get('rotation_speed', 1.0),
        catboost_model=catboost_clf,
        patient_features=patient_features
    )

    log.info("=== Pipeline Complete ===")
    log.info(f"Model: {model_path}")
    log.info(f"Results: {results_path}")
    log.info(f"CatBoost: {results['catboost_model_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full CEBRA pipeline')
    parser.add_argument('--config', default='config.json', help='Config file')
    args = parser.parse_args()
    main(args.config)
