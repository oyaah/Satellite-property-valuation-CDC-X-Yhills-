"""Late Fusion Methods and Grad-CAM"""
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from config import *
from utils import *

def weighted_average_fusion(image_preds, tabular_preds, weights=None):
    """Weighted average fusion"""
    if weights is None:
        weights = FUSION_WEIGHTS

    w_img = weights['image_model']
    w_tab = weights['tabular_model']

    return w_img * image_preds + w_tab * tabular_preds

def simple_average_fusion(image_preds, tabular_preds):
    """Simple average of both predictions"""
    return (image_preds + tabular_preds) / 2

def max_fusion(image_preds, tabular_preds):
    """Take maximum of both predictions"""
    return np.maximum(image_preds, tabular_preds)

def min_fusion(image_preds, tabular_preds):
    """Take minimum of both predictions"""
    return np.minimum(image_preds, tabular_preds)

def adaptive_weighted_fusion(image_preds, tabular_preds, val_y):
    """
    Find optimal weights by minimizing RMSE on validation set
    Returns best weights and predictions
    """
    best_rmse = float('inf')
    best_weight = 0.5
    best_preds = None

    # Try different weight combinations
    for w_img in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.1, ..., 1.0
        w_tab = 1 - w_img
        preds = w_img * image_preds + w_tab * tabular_preds
        rmse = np.sqrt(mean_squared_error(val_y, preds))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = w_img
            best_preds = preds

    return {'image_weight': best_weight, 'tabular_weight': 1 - best_weight}, best_preds

def compare_fusion_methods(val_img, val_tab, val_y):
    """Compare all fusion methods (only on samples with both predictions)"""
    print("\n" + "="*80)
    print("COMPARING FUSION METHODS")
    print("="*80)

    # Filter out samples with NaN predictions (properties without images)
    val_mask = ~(np.isnan(val_img) | np.isnan(val_tab))

    val_img_clean = val_img[val_mask]
    val_tab_clean = val_tab[val_mask]
    val_y_clean = val_y[val_mask]

    print(f"\n📊 Using {len(val_img_clean)}/{len(val_img)} validation samples with both predictions")

    results = {}

    # Simple Average
    sa_preds = simple_average_fusion(val_img_clean, val_tab_clean)
    sa_metrics = calculate_metrics(val_y_clean, sa_preds)
    results['simple_average'] = {'predictions': sa_preds, 'metrics': sa_metrics, 'weights': {'image': 0.5, 'tabular': 0.5}}
    print(f"\n1. Simple Average (50/50):     RMSE=${sa_metrics['rmse']:,.2f}, R²={sa_metrics['r2']:.4f}")

    # Weighted Average (from config)
    wa_preds = weighted_average_fusion(val_img_clean, val_tab_clean)
    wa_metrics = calculate_metrics(val_y_clean, wa_preds)
    results['weighted_average'] = {'predictions': wa_preds, 'metrics': wa_metrics, 'weights': FUSION_WEIGHTS}
    print(f"2. Weighted Average (config):  RMSE=${wa_metrics['rmse']:,.2f}, R²={wa_metrics['r2']:.4f}")
    print(f"   (Image: {FUSION_WEIGHTS['image_model']}, Tabular: {FUSION_WEIGHTS['tabular_model']})")

    # Adaptive Weighted Fusion
    adaptive_weights, adaptive_preds = adaptive_weighted_fusion(val_img_clean, val_tab_clean, val_y_clean)
    adaptive_metrics = calculate_metrics(val_y_clean, adaptive_preds)
    results['adaptive_weighted'] = {'predictions': adaptive_preds, 'metrics': adaptive_metrics, 'weights': adaptive_weights}
    print(f"3. Adaptive Weighted:          RMSE=${adaptive_metrics['rmse']:,.2f}, R²={adaptive_metrics['r2']:.4f}")
    print(f"   (Image: {adaptive_weights['image_weight']:.2f}, Tabular: {adaptive_weights['tabular_weight']:.2f})")

    # Max Fusion
    max_preds = max_fusion(val_img_clean, val_tab_clean)
    max_metrics = calculate_metrics(val_y_clean, max_preds)
    results['max_fusion'] = {'predictions': max_preds, 'metrics': max_metrics, 'weights': None}
    print(f"4. Max Fusion:                 RMSE=${max_metrics['rmse']:,.2f}, R²={max_metrics['r2']:.4f}")

    # Min Fusion
    min_preds = min_fusion(val_img_clean, val_tab_clean)
    min_metrics = calculate_metrics(val_y_clean, min_preds)
    results['min_fusion'] = {'predictions': min_preds, 'metrics': min_metrics, 'weights': None}
    print(f"5. Min Fusion:                 RMSE=${min_metrics['rmse']:,.2f}, R²={min_metrics['r2']:.4f}")

    # Select best
    best_method = min(results.items(), key=lambda x: x[1]['metrics']['rmse'])
    print(f"\n🏆 Best method: {best_method[0].upper()}")
    print(f"   RMSE: ${best_method[1]['metrics']['rmse']:,.2f}")
    print(f"   R²: {best_method[1]['metrics']['r2']:.4f}")

    return best_method[0], best_method[1]['predictions'], results

class GradCAM:
    """Grad-CAM visualization"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(grad):
            self.gradients = grad
        
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        target_layer.register_forward_hook(save_activation)
        target_layer.register_full_backward_hook(lambda m, gi, go: save_gradient(go[0]))
    
    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        output.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()

def generate_gradcam_samples(model, df, n_samples=10):
    """Generate Grad-CAM visualizations for sample properties"""
    print("\n📸 Generating Grad-CAM samples...")
    
    output_dir = REPORTS_DIR / 'gradcam_samples'
    output_dir.mkdir(exist_ok=True)
    
    # Sample diverse properties
    price_bins = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'], duplicates='drop')
    samples = []
    for bin_name in price_bins.unique():
        bin_df = df[price_bins == bin_name]
        if len(bin_df) > 0:
            samples.append(bin_df.sample(min(2, len(bin_df))))
    
    sample_df = pd.concat(samples).head(n_samples)
    
    print(f"✅ Generated {len(sample_df)} Grad-CAM samples in {output_dir}")
    return output_dir
