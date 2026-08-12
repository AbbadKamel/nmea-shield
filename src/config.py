"""
N2KShield Configuration
=======================
Central configuration for training and evaluation.
"""

# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================

# 15 monitored signals (ordered by hierarchical clustering)
SIGNALS = [
    'heading',
    'rate_of_turn',
    'yaw',
    'pitch',
    'roll',
    'magnetic_variation',
    'rudder_position',
    'rudder_order',
    'latitude',
    'longitude',
    'cog',
    'sog',
    'wind_speed',
    'wind_angle',
    'depth',
]


# =============================================================================
# FEATURE AGGREGATION
# =============================================================================

# Statistics computed per signal per window
AGGREGATION_STATS = ['mean', 'std', 'min', 'max']

# Total features = 15 signals × 4 stats = 60
NUM_SIGNALS = len(SIGNALS)
NUM_STATS = len(AGGREGATION_STATS)
NUM_FEATURES = NUM_SIGNALS * NUM_STATS  # 60

# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================

# Window sizes evaluated (seconds)
WINDOW_SIZES = [50, 75, 100]

# Default window size (best performance)
DEFAULT_WINDOW_SIZE = 75

# Stride for training (maximize diversity)
TRAIN_STRIDE = 1

# Stride for validation/test (reduce redundancy)
EVAL_STRIDE = 10

# Resampling frequency
RESAMPLE_FREQ = '1S'  # 1 second

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

MODEL_CONFIG = {
    # Encoder configuration
    'encoder': [
        {'filters': 32, 'kernel_size': (5, 5)},
        {'filters': 16, 'kernel_size': (5, 5)},
        {'filters': 16, 'kernel_size': (3, 3)},
    ],
    # Decoder mirrors encoder
    'decoder': [
        {'filters': 16, 'kernel_size': (3, 3)},
        {'filters': 16, 'kernel_size': (5, 5)},
        {'filters': 32, 'kernel_size': (5, 5)},
    ],
    # Activation
    'leaky_relu_alpha': 0.2,
    # Output activation
    'output_activation': 'linear',
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

TRAINING_CONFIG = {
    'learning_rate': 0.0002,
    'beta_1': 0.5,
    'beta_2': 0.99,
    'batch_size': 128,
    'epochs': 100,
    'early_stopping_patience': 10,
    'loss': 'mse',
}

# =============================================================================
# THRESHOLD SELECTION
# =============================================================================

# Percentile of validation MSE for threshold
THRESHOLD_PERCENTILE = 99

# =============================================================================
# ATTACK TYPES
# =============================================================================

ATTACK_TYPES = [
    'spike',     # Sudden value jump
    'noise',     # Random interference
    'drift',     # Gradual change
    'scaling',   # Multiplicative change
    'constant',  # Fixed value forcing
]

# Number of attack samples per type
NUM_ATTACK_SAMPLES = 50

# Attack intensity levels
ATTACK_INTENSITIES = {
    'low': 0.3,
    'medium': 0.6,
    'high': 1.0,
}

# =============================================================================
# DATA SPLIT
# =============================================================================

# Chronological split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# =============================================================================
# PATHS
# =============================================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
