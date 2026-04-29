"""Shared constants for cebra_pipeline plotting/analysis scripts."""

# Data encoding (matches `predictions` values 0..7)
CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA',
               'BurstSupp', 'Continuous', 'Discontinuous']
N_CLASSES = len(CLASS_NAMES)

# Display order: Seizure > GPD > LPD > LRDA > GRDA > BurstSupp > Discontinuous > Continuous
DISPLAY_ORDER = [0, 2, 1, 3, 4, 5, 7, 6]
DISPLAY_NAMES = [CLASS_NAMES[i] for i in DISPLAY_ORDER]

# Class colors (indexed by data encoding)
CLASS_COLORS = {
    0: '#FF4040',   # Seizure
    1: '#35D2BA',   # LPD
    2: '#449C7C',   # GPD
    3: '#F9EBB2',   # LRDA
    4: '#F8D759',   # GRDA
    5: '#7C6494',   # BurstSupp
    6: '#B3CDF5',   # Continuous
    7: '#C3B2EC',   # Discontinuous
}
CLASS_COLOR_LIST = [CLASS_COLORS[i] for i in range(N_CLASSES)]

# Outcome (binary CPC)
OUTCOME_GOOD     = '#3AA4F3'        # rgb(58,164,243)  CPC <= 2
OUTCOME_BAD      = '#FFA219'        # rgb(255,162,25)  CPC >= 3
OUTCOME_GOOD_RGB = 'rgb(58,164,243)'
OUTCOME_BAD_RGB  = 'rgb(255,162,25)'

# Uniform bin resolution
BIN_SEC = 300

# Paths
DATA_DIR = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model'
OUT_DIR  = '/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model/finalized'
