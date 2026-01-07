"""
Utility functions for the project
"""
import numpy as np
import random
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import pandas as pd

from src.config import *

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(title)
    
    # Add metrics
    metrics = calculate_metrics(y_true, y_pred)
    textstr = f"RMSE: ${metrics['rmse']:,.0f}\nR²: {metrics['r2']:.4f}\nMAE: ${metrics['mae']:,.0f}"
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def check_data_files(train_file, test_file, images_dir):
    """Check if required data files exist"""
    missing = []
    
    if not Path(train_file).exists():
        missing.append(str(train_file))
    if not Path(test_file).exists():
        missing.append(str(test_file))
    if not Path(images_dir).exists():
        missing.append(str(images_dir))
    
    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    return True

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def save_predictions(ids, predictions, filename):
    """Save predictions to CSV"""
    df = pd.DataFrame({
        'id': ids,
        'predicted_price': predictions
    })
    df.to_csv(filename, index=False)
    print(f"✅ Predictions saved to: {filename}")

def load_image_safely(image_path, target_size=(224, 224)):
    """Load image with error handling"""
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
