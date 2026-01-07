"""
Attention Visualization Module
Tools for visualizing cross-modal attention and image feature importance
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pandas as pd


class AttentionVisualizer:
    """
    Visualize attention maps from cross-modal attention mechanism
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_attention_map(self, image, tabular_features):
        """
        Get attention weights for a single sample
        
        Args:
            image: (3, H, W) or (1, 3, H, W) tensor
            tabular_features: (feature_dim,) or (1, feature_dim) tensor
        
        Returns:
            attention_map: (num_heads, spatial_h, spatial_w) numpy array
            prediction: scalar price prediction
        """
        self.model.eval()
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if tabular_features.dim() == 1:
            tabular_features = tabular_features.unsqueeze(0)
        
        with torch.no_grad():
            image = image.to(self.device)
            tabular_features = tabular_features.to(self.device)
            
            # Get prediction and attention weights
            prediction, attention_weights = self.model(
                image, tabular_features, return_attention=True
            )
            
            # Convert to numpy
            attention_map = attention_weights[0].cpu().numpy()  # (num_heads, H, W)
            prediction = prediction[0].item()
        
        return attention_map, prediction
    
    def visualize_attention_overlay(self, original_image, attention_map, 
                                    head_idx=None, alpha=0.5):
        """
        Create attention heatmap overlay on original image
        
        Args:
            original_image: (H, W, 3) numpy array (RGB, 0-255)
            attention_map: (num_heads, spatial_h, spatial_w) attention weights
            head_idx: Which attention head to visualize (None = average all)
            alpha: Opacity of heatmap overlay
        
        Returns:
            overlay: (H, W, 3) numpy array with heatmap overlay
        """
        # Select attention head
        if head_idx is None:
            # Average across all heads
            attn = attention_map.mean(axis=0)
        else:
            attn = attention_map[head_idx]
        
        # Resize attention to match original image size
        h, w = original_image.shape[:2]
        attn_resized = cv2.resize(attn, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-1
        attn_normalized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attn_normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay, attn_normalized
    
    def create_comprehensive_visualization(self, original_image, attention_map, 
                                          tabular_data, prediction, actual_price,
                                          property_info=None):
        """
        Create comprehensive visualization with:
        - Original image
        - Average attention heatmap
        - Individual head attentions
        - Property details
        
        Args:
            original_image: (H, W, 3) RGB image
            attention_map: (num_heads, spatial_h, spatial_w)
            tabular_data: Dictionary of tabular features
            prediction: Predicted price
            actual_price: Actual price
            property_info: Optional dict with property details
        
        Returns:
            fig: Matplotlib figure
        """
        num_heads = attention_map.shape[0]
        
        # Create figure with grid
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(3, num_heads + 1, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Original image + average attention
        ax_orig = fig.add_subplot(gs[0, :2])
        ax_orig.imshow(original_image)
        ax_orig.set_title('Original Satellite Image', fontsize=14, fontweight='bold')
        ax_orig.axis('off')
        
        # Average attention overlay
        ax_avg = fig.add_subplot(gs[0, 2:])
        overlay, _ = self.visualize_attention_overlay(original_image, attention_map, head_idx=None)
        ax_avg.imshow(overlay)
        ax_avg.set_title('Average Attention Across All Heads', fontsize=14, fontweight='bold')
        ax_avg.axis('off')
        
        # Row 2: Individual attention heads
        for i in range(num_heads):
            ax_head = fig.add_subplot(gs[1, i])
            overlay, _ = self.visualize_attention_overlay(
                original_image, attention_map, head_idx=i, alpha=0.6
            )
            ax_head.imshow(overlay)
            ax_head.set_title(f'Head {i+1}', fontsize=12)
            ax_head.axis('off')
        
        # Row 2: Property info (rightmost column)
        ax_info = fig.add_subplot(gs[1, -1])
        ax_info.axis('off')
        
        # Property details text
        info_text = f"💰 Price Information:\n"
        info_text += f"  Actual:    ${actual_price:,.0f}\n"
        info_text += f"  Predicted: ${prediction:,.0f}\n"
        error = abs(prediction - actual_price)
        error_pct = 100 * error / actual_price
        info_text += f"  Error:     ${error:,.0f} ({error_pct:.1f}%)\n\n"
        
        if property_info:
            info_text += "🏠 Property Features:\n"
            for key, value in list(property_info.items())[:8]:
                if isinstance(value, float):
                    info_text += f"  {key}: {value:.2f}\n"
                else:
                    info_text += f"  {key}: {value}\n"
        
        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 3: Attention distribution per head
        ax_dist = fig.add_subplot(gs[2, :])
        
        # Calculate attention statistics for each head
        head_means = [attention_map[i].mean() for i in range(num_heads)]
        head_stds = [attention_map[i].std() for i in range(num_heads)]
        head_maxs = [attention_map[i].max() for i in range(num_heads)]
        
        x = np.arange(num_heads)
        width = 0.25
        
        ax_dist.bar(x - width, head_means, width, label='Mean', alpha=0.8)
        ax_dist.bar(x, head_stds, width, label='Std Dev', alpha=0.8)
        ax_dist.bar(x + width, head_maxs, width, label='Max', alpha=0.8)
        
        ax_dist.set_xlabel('Attention Head', fontsize=12)
        ax_dist.set_ylabel('Attention Weight', fontsize=12)
        ax_dist.set_title('Attention Distribution Statistics by Head', fontsize=14, fontweight='bold')
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels([f'H{i+1}' for i in range(num_heads)])
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        plt.suptitle(f'Cross-Modal Attention Visualization', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def visualize_batch(self, dataloader, num_samples=10, save_dir=None):
        """
        Visualize attention for a batch of samples
        
        Args:
            dataloader: DataLoader with validation/test data
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        
        Returns:
            visualizations: List of figures
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        visualizations = []
        sample_count = 0
        
        self.model.eval()
        
        print(f"\n📸 Generating attention visualizations...")
        
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            images = batch['image'].to(self.device)
            tabular = batch['tabular'].to(self.device)
            prices = batch['price'].cpu().numpy()
            ids = batch['id'].cpu().numpy()
            
            batch_size = images.size(0)
            
            for i in range(min(batch_size, num_samples - sample_count)):
                # Get attention map
                attention_map, prediction = self.get_attention_map(
                    images[i], tabular[i]
                )
                
                # Denormalize image for visualization
                image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize from ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = (image_np * std + mean) * 255
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
                # Create property info dict from tabular features
                property_info = {
                    'ID': ids[i],
                }
                
                # Placeholder for actual and predicted prices
                # (These should be denormalized in practice)
                actual_price = prices[i] * 1e5  # Placeholder scaling
                predicted_price = prediction * 1e5  # Placeholder scaling
                
                # Create visualization
                fig = self.create_comprehensive_visualization(
                    image_np, attention_map,
                    tabular_data={},
                    prediction=predicted_price,
                    actual_price=actual_price,
                    property_info=property_info
                )
                
                if save_dir:
                    save_path = save_dir / f'attention_sample_{sample_count+1:03d}_id_{ids[i]}.png'
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved: {save_path.name}")
                    plt.close(fig)
                else:
                    visualizations.append(fig)
                
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        print(f"✅ Generated {sample_count} visualizations")
        
        if not save_dir:
            return visualizations


def analyze_attention_patterns(model, dataloader, device, stats, num_samples=100):
    """
    Analyze attention patterns across dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
        stats: Normalization statistics
        num_samples: Number of samples to analyze
    
    Returns:
        analysis: Dictionary with attention statistics
    """
    model.eval()
    
    attention_entropy = []
    attention_max_values = []
    head_activations = []
    
    sample_count = 0
    
    print(f"\n📊 Analyzing attention patterns...")
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            
            # Get attention weights
            _, attention_weights = model(images, tabular, return_attention=True)
            
            # attention_weights: (batch, num_heads, H, W)
            batch_size = attention_weights.size(0)
            num_heads = attention_weights.size(1)
            
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                attn = attention_weights[i]  # (num_heads, H, W)
                
                # Calculate entropy for each head
                for h in range(num_heads):
                    attn_flat = attn[h].flatten()
                    attn_prob = attn_flat / (attn_flat.sum() + 1e-8)
                    entropy = -(attn_prob * torch.log(attn_prob + 1e-8)).sum()
                    attention_entropy.append(entropy.item())
                
                # Max attention value
                attention_max_values.append(attn.max().item())
                
                # Head-wise activation
                head_activations.append(attn.mean(dim=(1, 2)).cpu().numpy())
                
                sample_count += 1
    
    analysis = {
        'mean_entropy': np.mean(attention_entropy),
        'std_entropy': np.std(attention_entropy),
        'mean_max_attention': np.mean(attention_max_values),
        'head_activations_mean': np.array(head_activations).mean(axis=0),
        'head_activations_std': np.array(head_activations).std(axis=0),
    }
    
    print(f"  Mean Entropy: {analysis['mean_entropy']:.4f} ± {analysis['std_entropy']:.4f}")
    print(f"  Mean Max Attention: {analysis['mean_max_attention']:.4f}")
    print(f"  Head Activations: {analysis['head_activations_mean']}")
    
    return analysis
