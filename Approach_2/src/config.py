"""
Configuration for Approach 2: Intermediate Fusion with Cross-Modal Attention (Enhanced)
"""
from pathlib import Path
import torch

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'

# Data files
TRAIN_PROCESSED = DATA_DIR / 'train_processed.csv'
VAL_PROCESSED = DATA_DIR / 'val_processed.csv'
TEST_PROCESSED = DATA_DIR / 'test_processed.csv'
FINAL_TEST_PROCESSED = DATA_DIR / 'final_test_processed.csv'

# Create directories
for dir_path in [MODELS_DIR, REPORTS_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Image backbone
IMAGE_CONFIG = {
    'backbone': 'efficientnet_b1',
    'pretrained': True,
    'freeze_ratio': 0.90,           # Freeze 90%, train only 10%
    'img_size': 224,
    'extract_features': True,
}

# Tabular Transformer (ENHANCED - replaces MLP)
TRANSFORMER_CONFIG = {
    'd_model': 128,            # Transformer hidden dimension
    'nhead': 4,                # Number of attention heads
    'num_layers': 2,           # Number of transformer encoder layers
    'dim_feedforward': 256,    # FFN dimension
    'dropout': 0.1,            # Transformer dropout
    'output_dim': 32,          # Final output dimension for tabular context
}

# Cross-modal attention
ATTENTION_CONFIG = {
    'spatial_features': 49,        # 7x7 feature maps from CNN
    'feature_dim': 1280,           # EfficientNet-B1 feature dimension
    'tabular_context_dim': 32,     # Output from tabular transformer
    'attention_heads': 4,          # Multi-head attention
    'attention_dropout': 0.1,
}

# Fusion network
FUSION_CONFIG = {
    'fusion_dims': [256, 128, 64],
    'dropout': 0.3,
    'output_dim': 1,
}

# ============================================================================
# TRAINING
# ============================================================================

TRAINING_CONFIG = {
    'num_epochs': 25,              # Increased for transformer
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 12,  # Increased patience
    'grad_clip': 1.0,
    
    # Learning rate schedule
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'min_lr': 1e-7,
    
    # Warmup
    'warmup_epochs': 3,
    'warmup_lr': 1e-6,
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.3,
    'rotate_90': 0.5,
    'shift_scale_rotate': {
        'shift_limit': 0.1,
        'scale_limit': 0.1,
        'rotate_limit': 15,
        'p': 0.5
    },
    'brightness_contrast': {
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'p': 0.5
    },
    'gauss_noise': {
        'var_limit': (10.0, 50.0),
        'p': 0.3
    },
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Categorical features to encode
CATEGORICAL_FEATURES = [
    'waterfront', 'view', 'condition', 'grade',
    'has_basement', 'was_renovated', 'is_luxury',
    'has_view',
]

# Numerical features to normalize
NUMERICAL_FEATURES = [
    'bedrooms', 'bathrooms', 'floors',
    'property_age', 'years_since_renovation',
    'total_rooms',
    'distance_from_seattle',
]

# Log-transformed features (already in log scale in the CSV)
LOG_FEATURES = [
    'sqft_living_log', 'sqft_lot_log', 'sqft_above_log',
    'sqft_basement_log', 'sqft_living15_log', 'sqft_lot15_log'
]

# Features to exclude
EXCLUDE_FEATURES = [
    'id', 'price', 'price_log', 'image_path', 'image_exists',
    'lat', 'long',  # Exclude lat/long as they're spatial info, not features
]

# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'num_samples': 20,           # Number of validation samples to visualize
    'save_dir': REPORTS_DIR / 'attention_visualizations',
    'analyze_patterns': True,    # Whether to analyze attention patterns
    'num_analysis_samples': 100, # Samples for pattern analysis
}

# ============================================================================
# DEVICE & REPRODUCIBILITY
# ============================================================================

# Device detection (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
RANDOM_STATE = 42
NUM_WORKERS = 4

# ============================================================================
# MLFLOW TRACKING
# ============================================================================

MLFLOW_CONFIG = {
    'tracking_uri': str(REPORTS_DIR / 'mlruns'),
    'experiment_name': 'Approach_2_Attention_Fusion_Enhanced',
    'run_name': 'transformer_attention_v1',
}

print(f"✅ Configuration loaded (Enhanced with Tabular Transformer)")
print(f"   Device: {DEVICE}")
print(f"   Models dir: {MODELS_DIR}")
print(f"   Data dir: {DATA_DIR}")
print(f"   Transformer: d_model={TRANSFORMER_CONFIG['d_model']}, heads={TRANSFORMER_CONFIG['nhead']}, layers={TRANSFORMER_CONFIG['num_layers']}")
