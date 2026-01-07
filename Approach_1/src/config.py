"""
Configuration file for Satellite Property Valuation Project
"""
import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, 
                  MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== DATA FILES ====================
TRAIN_FILE = RAW_DATA_DIR / "train.xlsx"
TEST_FILE = RAW_DATA_DIR / "test.xlsx"
IMAGES_DIR = RAW_DATA_DIR / "satellite_images_19"

# ==================== IMAGE MATCHING ====================
def format_latlong_for_filename(lat, long):
    """
    Convert lat/long to image filename format.
    Images are named: img_{seq}_{lat}_{long}.png
    where lat/long have underscores instead of decimal points
    
    Example:
    lat=47.4362, long=-122.187 -> '47_436200', '-122_187000'
    
    The trick: we format with 6 decimals, then replace '.' with '_'
    """
    # Format to 6 decimal places
    lat_formatted = f"{lat:.6f}"
    long_formatted = f"{long:.6f}"
    
    # Replace decimal point with underscore
    lat_str = lat_formatted
    long_str = long_formatted
    
    return lat_str, long_str

# ==================== FEATURE ENGINEERING ====================
TARGET_COL = "price"
ID_COL = "id"
DATE_COL = "date"
LAT_COL = "lat"
LONG_COL = "long"

DROP_COLS = []

NUMERICAL_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", 
    "floors", "waterfront", "view", "condition", "grade",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
    "lat", "long", "sqft_living15", "sqft_lot15"
]

CATEGORICAL_FEATURES = ["zipcode"]

# ==================== MODEL HYPERPARAMETERS ====================

IMAGE_CONFIG = {
    "model_name": "efficientnet_b1",
    "pretrained": True,
    "img_size": 224,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "dropout": 0.3,
    "hidden_dims": [256, 128],
    "grad_cam_layer": "features",
}

XGBOOST_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

LIGHTGBM_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

CATBOOST_PARAMS = {
    "depth": 8,
    "learning_rate": 0.05,
    "iterations": 1000,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_state": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
}

# ==================== TRAIN/VAL/TEST SPLIT ====================
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# ==================== FUSION CONFIGURATION ====================
FUSION_METHODS = ["weighted_average", "meta_learner", "stacking"]
FUSION_WEIGHTS = {
    "image_model": 0.4,
    "tabular_model": 0.6,
}

# ==================== MLFLOW CONFIGURATION ====================
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT_NAME = "Satellite_Property_Valuation"

# ==================== EVALUATION METRICS ====================
METRICS = ["rmse", "mae", "r2", "mape"]

# ==================== IMAGE AUGMENTATION ====================
TRAIN_AUGMENTATION = {
    "HorizontalFlip": {"p": 0.5},
    "RandomRotate90": {"p": 0.5},
    "ShiftScaleRotate": {
        "shift_limit": 0.1,
        "scale_limit": 0.1,
        "rotate_limit": 15,
        "p": 0.5
    },
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5
    },
    "GaussNoise": {"var_limit": (10.0, 50.0), "p": 0.3},
}

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== REPRODUCIBILITY ====================
SEED = 42