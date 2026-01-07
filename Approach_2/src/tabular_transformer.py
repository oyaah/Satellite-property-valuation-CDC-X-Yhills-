"""
Tabular Transformer Network
Self-attention based architecture for tabular feature processing
"""
import torch
import torch.nn as nn
import math


class TabularTransformerEncoder(nn.Module):
    """
    Transformer encoder for tabular data
    Treats each feature as a token and applies self-attention
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1):
        """
        Args:
            input_dim: Number of input features
            d_model: Dimension of the model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature embedding - project each feature to d_model dimensions
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Learnable positional encoding for features
        self.positional_encoding = nn.Parameter(
            torch.randn(input_dim, d_model) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_dim = d_model
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.feature_embedding.weight)
        nn.init.zeros_(self.feature_embedding.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - tabular features
        
        Returns:
            output: (batch_size, d_model) - aggregated representation
            attention_weights: Optional attention weights for visualization
        """
        batch_size = x.size(0)
        
        # Reshape each feature as a separate token: (batch, features, 1)
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        # Embed each feature independently
        # (batch_size, input_dim, 1) -> (batch_size, input_dim, d_model)
        feature_embeds = self.feature_embedding(x)
        
        # Add positional encoding
        feature_embeds = feature_embeds + self.positional_encoding.unsqueeze(0)
        
        # Apply transformer encoder
        # (batch_size, input_dim, d_model)
        transformed = self.transformer_encoder(feature_embeds)
        
        # Aggregate across features - use mean pooling
        # (batch_size, d_model)
        output = transformed.mean(dim=1)
        
        return output


class TabularEmbeddingNetwork(nn.Module):
    """
    Enhanced Tabular Embedding using Transformer architecture
    Replaces the simple MLP with self-attention mechanism
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, output_dim=32):
        super().__init__()
        
        # Transformer encoder for feature interactions
        self.transformer = TabularTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Final projection to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) - tabular features
        
        Returns:
            output: (batch_size, output_dim) - embedded tabular context
        """
        # Transform features with self-attention
        transformed = self.transformer(x)
        
        # Project to output dimension
        output = self.projection(transformed)
        
        return output


def count_transformer_parameters(model):
    """Count parameters in transformer model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_pct': 100 * trainable / total if total > 0 else 0
    }
