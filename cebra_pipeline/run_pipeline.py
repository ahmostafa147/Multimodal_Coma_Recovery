"""
run_pipeline.py — drives the full CEBRA pipeline (01..08) for every combo.

All top-level config vars are overridable from this file: edit COMBOS for
per-combo settings, SCRIPT_CONFIG for per-script hyperparams (shared
across combos). The runner regex-rewrites those exact assignments at the
top of each script before exec'ing it.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── PER-COMBO CONFIG ────────────────────────────────────────────────
# Keys here override matching top-level vars in 01 and 02.
COMBOS = {
    'combo_i':   {'FEATURE_KEYS': ['features'],
                  'LABEL_KEYS_DISC': ['predictions'],
                  'LABEL_KEYS_CONT': []},
    'combo_ii':  {'FEATURE_KEYS': ['features'],
                  'LABEL_KEYS_DISC': ['predictions', 'cpc_binary'],
                  'LABEL_KEYS_CONT': []},
    'combo_iii': {'FEATURE_KEYS': ['features', 'cebra_features'],
                  'LABEL_KEYS_DISC': [],
                  'LABEL_KEYS_CONT': ['probabilities']},
    'combo_iv':  {'FEATURE_KEYS': ['features', 'cebra_features'],
                  'LABEL_KEYS_DISC': ['cpc_binary'],
                  'LABEL_KEYS_CONT': ['probabilities']},
    'combo_v':   {'FEATURE_KEYS': ['features', 'cebra_features'],
                  'LABEL_KEYS_DISC': ['predictions', 'cpc_binary'],
                  'LABEL_KEYS_CONT': ['probabilities']},
}

# ── PER-SCRIPT CONFIG (shared across combos) ───────────────────────
# Add/remove keys freely; only keys whose name matches a top-level
# assignment in the target script are rewritten.
SCRIPT_CONFIG = {
    '01_data_preprocessing.py': {
        'PCA_KEY':        'features',
        'PCA_COMPONENTS': 50,
        'SEED':           42,
        'BIN_SEC':        300,
    },
    '02_cebra_hybrid_training.py': {
        'USE_TIME_OBJECTIVE': True,
        'OUTPUT_DIM':         3,
        'TIME_OFFSET':        144,        # 12h at 5-min res
        'BATCH_SIZE':         1024,
        'MAX_ITER':           20000,
        'TEMPERATURE':        0.35,
        'NUM_UNITS':          32,
        'LR':                 3e-4,
        'KNN_NEIGHBORS':      10,
        'SEED':               42,
    },
    '03_embedding_visualization.py':  {},
    '04_classification_evaluation.py':{},
    '05_temporal_analysis.py':        {},
    '06_patient_analysis.py':    {'PID': 'ICARE_0647', 'WINDOW': 15,
                                  'SLERP_N': 10, 'TOP_K': 50},
    '06_patient_analysis_v2.py': {'PID': 'ICARE_0647', 'WINDOW': 15,
                                  'SLERP_N': 10, 'TOP_K': 50},
    '07_sphere_views.py': {'SUBSAMPLE': 30000, 'NORMALIZE_TIME': True},
    '08_combo_grid.py':   {'SUBSAMPLE': 15000, 'NORMALIZE_TIME': True},
}

PER_COMBO_SCRIPTS = [
    '01_data_preprocessing.py',
    '02_cebra_hybrid_training.py',
    '03_embedding_visualization.py',
    '04_classification_evaluation.py',
    '05_temporal_analysis.py',
    '06_patient_analysis.py',
    '06_patient_analysis_v2.py',
    '07_sphere_views.py',
]
GLOBAL_SCRIPTS = ['08_combo_grid.py']

# ── Runner ──────────────────────────────────────────────────────────
def run(script, overrides):
    path = os.path.join(HERE, script)
    src  = open(path).read()
    for key, val in overrides.items():
        src, n = re.subn(rf'(?m)^{re.escape(key)}\s*=.*$',
                         f'{key} = {val!r}', src, count=1)
        if n == 0:
            print(f'  [warn] {script}: no top-level `{key} = ...` to override')
    print(f'\n>>> {script}  {overrides}')
    exec(compile(src, path, 'exec'),
         {'__name__': '__main__', '__file__': path})

for tag, combo_cfg in COMBOS.items():
    for s in PER_COMBO_SCRIPTS:
        overrides = {'RUN_TAG': tag,
                     **SCRIPT_CONFIG.get(s, {}),
                     **(combo_cfg if s.startswith(('01_', '02_')) else {})}
        run(s, overrides)

for s in GLOBAL_SCRIPTS:
    overrides = {'COMBOS': list(COMBOS.keys()),
                 **SCRIPT_CONFIG.get(s, {})}
    run(s, overrides)

print('\nAll done.')
