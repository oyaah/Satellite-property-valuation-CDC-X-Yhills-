"""
Cross-Modal Attention Fusion Model (Enhanced with Tabular Transformer)
Learns to attend to relevant image regions based on tabular context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from tabular_transformer import TabularEmbeddingNetwork


class MultiHeadCrossModalAttention(nn.Module):
    """
    Multi-head cross-modal attention mechanism
    Uses tabular context to attend to relevant spatial regions in image features
    """
    def __init__(self, feature_dim, tabular_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.tabular_dim = tabular_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Project tabular context to match feature dimension
        self.tabular_proj = nn.Linear(tabular_dim, feature_dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(tabular_dim, feature_dim)  # Query from tabular
        self.k_proj = nn.Linear(feature_dim, feature_dim)   # Key from image
        self.v_proj = nn.Linear(feature_dim, feature_dim)   # Value from image
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, image_features, tabular_context):
        """
        Args:
            image_features: (batch, spatial_locations, feature_dim)
            tabular_context: (batch, tabular_dim)
        Returns:
            attended_features: (batch, feature_dim)
            attention_weights: (batch, num_heads, spatial_locations)
        """
        batch_size = image_features.size(0)
        spatial_locations = image_features.size(1)
        
        # Expand tabular context for query
        tabular_expanded = tabular_context.unsqueeze(1)  # (batch, 1, tabular_dim)
        
        # Project to Q, K, V
        Q = self.q_proj(tabular_expanded)  # (batch, 1, feature_dim)
        K = self.k_proj(image_features)    # (batch, spatial_locations, feature_dim)
        V = self.v_proj(image_features)    # (batch, spatial_locations, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, 1, head_dim)
        
        K = K.view(batch_size, spatial_locations, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, spatial_locations, head_dim)
        
        V = V.view(batch_size, spatial_locations, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, spatial_locations, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, num_heads, 1, spatial_locations)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (batch, num_heads, 1, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.feature_dim)
        attended = attended.squeeze(1)  # (batch, feature_dim)
        
        # Output projection
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        # Return attention weights for visualization
        attn_weights = attn_weights.squeeze(2)  # (batch, num_heads, spatial_locations)
        
        return output, attn_weights


class AttentionFusionModel(nn.Module):
    """
    Complete cross-modal attention fusion model with Tabular Transformer
    
    Architecture:
    1. Image CNN → Spatial feature maps (7x7x1280)
    2. Tabular Transformer → Context vector (32-dim) [ENHANCED]
    3. Cross-modal attention → Attended image features
    4. Concatenate attended image + tabular → Fusion network → Price
    """
    def __init__(self, tabular_input_dim, config):
        super().__init__()
        
        # ====== IMAGE BRANCH ======
        # Load pretrained CNN backbone
        self.image_backbone = timm.create_model(
            config['IMAGE_CONFIG']['backbone'],
            pretrained=config['IMAGE_CONFIG']['pretrained'],
            features_only=True,  # Extract feature maps
            out_indices=[4]  # Last conv block
        )
        
        # Freeze backbone layers
        freeze_ratio = config['IMAGE_CONFIG']['freeze_ratio']
        total_params = list(self.image_backbone.parameters())
        total_param_count = sum(p.numel() for p in total_params)
        freeze_threshold = total_param_count * freeze_ratio
        
        cumsum = 0
        for param in self.image_backbone.parameters():
            cumsum += param.numel()
            param.requires_grad = (cumsum > freeze_threshold)
        
        # Get feature dimension (1280 for EfficientNet-B1)
        self.feature_dim = self.image_backbone.feature_info[-1]['num_chs']
        
        # ====== TABULAR BRANCH (TRANSFORMER) ======
        transformer_config = config.get('TRANSFORMER_CONFIG', {})
        
        self.tabular_embedding = TabularEmbeddingNetwork(
            input_dim=tabular_input_dim,
            d_model=transformer_config.get('d_model', 128),
            nhead=transformer_config.get('nhead', 4),
            num_layers=transformer_config.get('num_layers', 2),
            dim_feedforward=transformer_config.get('dim_feedforward', 256),
            dropout=transformer_config.get('dropout', 0.1),
            output_dim=transformer_config.get('output_dim', 32)
        )
        self.tabular_context_dim = self.tabular_embedding.output_dim
        
        # ====== CROSS-MODAL ATTENTION ======
        self.cross_attention = MultiHeadCrossModalAttention(
            feature_dim=self.feature_dim,
            tabular_dim=self.tabular_context_dim,
            num_heads=config['ATTENTION_CONFIG']['attention_heads'],
            dropout=config['ATTENTION_CONFIG']['attention_dropout']
        )
        
        # ====== FUSION NETWORK ======
        fusion_input_dim = self.feature_dim + self.tabular_context_dim
        
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in config['FUSION_CONFIG']['fusion_dims']:
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            fusion_layers.append(nn.ReLU(inplace=True))
            fusion_layers.append(nn.Dropout(config['FUSION_CONFIG']['dropout']))
            prev_dim = hidden_dim
        
        # Final prediction layer
        fusion_layers.append(nn.Linear(prev_dim, config['FUSION_CONFIG']['output_dim']))
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
    def forward(self, images, tabular_features, return_attention=False):
        """
        Forward pass
        
        Args:
            images: (batch, 3, H, W)
            tabular_features: (batch, tabular_dim)
            return_attention: If True, return attention weights for visualization
        
        Returns:
            predictions: (batch,) price predictions
            attention_weights: (batch, num_heads, spatial_locations) if return_attention=True
        """
        # Extract image feature maps
        image_feature_maps = self.image_backbone(images)[-1]
        # (batch, feature_dim, H, W) e.g., (batch, 1280, 7, 7)
        
        # Reshape to (batch, spatial_locations, feature_dim)
        batch_size = image_feature_maps.size(0)
        feature_dim = image_feature_maps.size(1)
        spatial_h = image_feature_maps.size(2)
        spatial_w = image_feature_maps.size(3)
        
        image_features = image_feature_maps.view(batch_size, feature_dim, -1)
        image_features = image_features.transpose(1, 2)  # (batch, spatial_locations, feature_dim)
        
        # Process tabular features with TRANSFORMER
        tabular_context = self.tabular_embedding(tabular_features)
        # (batch, tabular_context_dim)
        
        # Apply cross-modal attention
        attended_image_features, attention_weights = self.cross_attention(
            image_features, tabular_context
        )
        # attended_image_features: (batch, feature_dim)
        # attention_weights: (batch, num_heads, spatial_locations)
        
        # Concatenate attended image features with tabular context
        fused_features = torch.cat([attended_image_features, tabular_context], dim=1)
        # (batch, feature_dim + tabular_context_dim)
        
        # Final prediction
        predictions = self.fusion_network(fused_features).squeeze(1)
        # (batch,)
        
        if return_attention:
            # Reshape attention for visualization (batch, num_heads, H, W)
            attention_weights_2d = attention_weights.view(
                batch_size, -1, spatial_h, spatial_w
            )
            return predictions, attention_weights_2d
        
        return predictions
    
    def get_attention_maps(self, images, tabular_features):
        """
        Helper function to get attention maps for visualization
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(
                images, tabular_features, return_attention=True
            )
        return attention_weights


def count_parameters(model):
    """Count trainable and total parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by component
    components = {}
    
    if hasattr(model, 'image_backbone'):
        components['image_backbone'] = sum(p.numel() for p in model.image_backbone.parameters())
        components['image_backbone_trainable'] = sum(
            p.numel() for p in model.image_backbone.parameters() if p.requires_grad
        )
    
    if hasattr(model, 'tabular_embedding'):
        components['tabular_transformer'] = sum(p.numel() for p in model.tabular_embedding.parameters())
    
    if hasattr(model, 'cross_attention'):
        components['cross_attention'] = sum(p.numel() for p in model.cross_attention.parameters())
    
    if hasattr(model, 'fusion_network'):
        components['fusion_network'] = sum(p.numel() for p in model.fusion_network.parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_pct': 100 * trainable / total,
        'components': components
    }
