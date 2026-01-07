"""
CNN Image Model Training - Optimized Transfer Learning
- Freezes 80% of backbone layers (only trains last 20%)
- 10 epochs for faster training
- Learning rate: 3e-4
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm
import mlflow
from sklearn.metrics import mean_squared_error, r2_score

from config import *
from utils import *

class PropertyImageDataset(Dataset):
    """Dataset for property images with log-transformed prices"""
    def __init__(self, df, transform=None, price_log_mean=None, price_log_std=None):
        # Only keep rows with images
        self.df = df[df['image_exists']].reset_index(drop=True)
        self.transform = transform

        # Calculate or use provided normalization stats for log(price)
        if price_log_mean is None and 'price_log' in self.df.columns:
            self.price_log_mean = self.df['price_log'].mean()
            self.price_log_std = self.df['price_log'].std()
        else:
            self.price_log_mean = price_log_mean if price_log_mean is not None else 0
            self.price_log_std = price_log_std if price_log_std is not None else 1
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image = cv2.imread(row['image_path'])
        if image is None:
            raise RuntimeError(f"Failed to load image: {row['image_path']}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']

        # Get log(price) and normalize
        if 'price_log' in row:
            price_log = (row['price_log'] - self.price_log_mean) / self.price_log_std
        else:
            price_log = 0.0

        return image, torch.tensor(price_log, dtype=torch.float32), row['id']

def get_train_transforms():
    """Training data augmentation"""
    return A.Compose([
        A.Resize(IMAGE_CONFIG['img_size'], IMAGE_CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    """Validation/test transforms (no augmentation)"""
    return A.Compose([
        A.Resize(IMAGE_CONFIG['img_size'], IMAGE_CONFIG['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class ImageRegressionModel(nn.Module):
    """
    EfficientNet-B1 with frozen backbone (80% of layers)
    Only trains last 20% of backbone + custom regression head
    """
    def __init__(self, freeze_ratio=0.8):
        super().__init__()
        
        # Load pretrained EfficientNet-B1
        self.backbone = timm.create_model(
            IMAGE_CONFIG['model_name'],  # 'efficientnet_b1'
            pretrained=IMAGE_CONFIG['pretrained'],  # True
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Freeze first 80% of backbone layers
        # Count total parameters first
        total_backbone_params = sum(p.numel() for p in self.backbone.parameters())
        freeze_threshold = total_backbone_params * freeze_ratio
        
        frozen = 0
        trainable = 0
        cumsum = 0
        
        # Freeze layers until we reach 80% threshold
        for param in self.backbone.parameters():
            cumsum += param.numel()
            if cumsum <= freeze_threshold:
                param.requires_grad = False  # Freeze
                frozen += param.numel()
            else:
                param.requires_grad = True   # Train
                trainable += param.numel()
        
        total = frozen + trainable
        print(f"\n📊 Backbone Parameters:")
        print(f"  Total:     {total:>12,}")
        print(f"  Frozen:    {frozen:>12,} ({100*frozen/total:>5.1f}%)")
        print(f"  Trainable: {trainable:>12,} ({100*trainable/total:>5.1f}%)")
        
        # Custom regression head
        in_features = self.backbone.num_features  # 1280 for EfficientNet-B1
        self.regression_head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(IMAGE_CONFIG['dropout']),  # 0.3
            nn.Linear(in_features, IMAGE_CONFIG['hidden_dims'][0]),  # 1280 → 256
            nn.ReLU(),
            nn.BatchNorm1d(IMAGE_CONFIG['hidden_dims'][0]),
            nn.Dropout(IMAGE_CONFIG['dropout']),
            nn.Linear(IMAGE_CONFIG['hidden_dims'][0], IMAGE_CONFIG['hidden_dims'][1]),  # 256 → 128
            nn.ReLU(),
            nn.Linear(IMAGE_CONFIG['hidden_dims'][1], 1)  # 128 → 1 (price)
        )
        
        # Count head parameters
        head_params = sum(p.numel() for p in self.regression_head.parameters())
        print(f"\n📊 Regression Head Parameters: {head_params:,}")
        print(f"\n📊 Total Trainable: {trainable + head_params:,}")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features).squeeze()

def train_image_model(train_df, val_df, num_epochs=10, learning_rate=3e-4):
    """
    Train image regression model with optimized settings
    
    Args:
        train_df: Training dataframe with image_path and price columns
        val_df: Validation dataframe
        num_epochs: Number of training epochs (default: 10)
        learning_rate: Learning rate (default: 3e-4)
    
    Returns:
        model: Trained model
        metrics: Dictionary with validation metrics
    """
    print("\n" + "="*80)
    print("TRAINING IMAGE MODEL - TRANSFER LEARNING")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"  Model: {IMAGE_CONFIG['model_name']}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {IMAGE_CONFIG['batch_size']}")
    print(f"  Image Size: {IMAGE_CONFIG['img_size']}x{IMAGE_CONFIG['img_size']}")
    print(f"  Frozen Layers: 80%")
    
    set_seed(RANDOM_STATE)

    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n🖥️  Device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🖥️  Device: {device}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"\n🖥️  Device: {device}")
    
    # Calculate log(price) normalization stats from training data
    price_log_mean = train_df['price_log'].mean()
    price_log_std = train_df['price_log'].std()

    print(f"\n📊 Log(Price) Normalization:")
    print(f"  Mean:  {price_log_mean:>8.4f}")
    print(f"  Std:   {price_log_std:>8.4f}")
    print(f"  Range: {train_df['price_log'].min():>8.4f} - {train_df['price_log'].max():>8.4f}")
    print(f"\n📝 Training Strategy:")
    print(f"  1. Train model to predict log(price)")
    print(f"  2. At inference: y_pred = exp(y_pred_log)")
    print(f"  3. Evaluate on actual price (not log)")

    # Create datasets with normalization
    train_dataset = PropertyImageDataset(train_df, get_train_transforms(), price_log_mean, price_log_std)
    val_dataset = PropertyImageDataset(val_df, get_val_transforms(), price_log_mean, price_log_std)
    
    print(f"\n📊 Dataset Sizes:")
    print(f"  Training:   {len(train_dataset):>6,} images")
    print(f"  Validation: {len(val_dataset):>6,} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=IMAGE_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm error with batch_size=1
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=IMAGE_CONFIG['batch_size'],
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Initialize model
    print(f"\n🔧 Initializing model...")
    model = ImageRegressionModel(freeze_ratio=0.8).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=IMAGE_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=IMAGE_CONFIG['early_stopping_patience'], 
        mode='min'
    )
    
    best_val_rmse = float('inf')
    best_val_r2 = -float('inf')
    
    print(f"\n🚀 Starting training...")
    print("="*80)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # ============ TRAINING ============
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        train_pbar = tqdm(train_loader, desc=f"Training  ", leave=False)
        for images, prices, _ in train_pbar:
            images, prices = images.to(device), prices.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, prices)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_true.extend(prices.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.2f}'})
        
        avg_train_loss = train_loss / len(train_loader)

        # Denormalize log predictions and convert to actual prices
        train_preds_log = np.array(train_preds) * price_log_std + price_log_mean
        train_true_log = np.array(train_true) * price_log_std + price_log_mean

        # Convert from log space to actual prices
        train_preds_price = np.exp(train_preds_log)
        train_true_price = np.exp(train_true_log)

        train_rmse = np.sqrt(mean_squared_error(train_true_price, train_preds_price))
        train_r2 = r2_score(train_true_price, train_preds_price)
        
        # ============ VALIDATION ============
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
        with torch.no_grad():
            for images, prices, _ in val_pbar:
                images, prices = images.to(device), prices.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, prices)
                
                # Track metrics
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(prices.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.2f}'})
        
        avg_val_loss = val_loss / len(val_loader)

        # Denormalize log predictions and convert to actual prices
        val_preds_log = np.array(val_preds) * price_log_std + price_log_mean
        val_true_log = np.array(val_true) * price_log_std + price_log_mean

        # Convert from log space to actual prices
        val_preds_price = np.exp(val_preds_log)
        val_true_price = np.exp(val_true_log)

        val_rmse = np.sqrt(mean_squared_error(val_true_price, val_preds_price))
        val_r2 = r2_score(val_true_price, val_preds_price)
        
        # Update learning rate
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:>8.2f} | Train RMSE: ${train_rmse:>12,.2f} | Train R²: {train_r2:>6.4f}")
        print(f"  Val Loss:   {avg_val_loss:>8.2f} | Val RMSE:   ${val_rmse:>12,.2f} | Val R²:   {val_r2:>6.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'price_log_mean': price_log_mean,
                'price_log_std': price_log_std,
            }, MODELS_DIR / 'best_image_model.pth')
            
            print(f"  ✅ Best model saved! (Val RMSE: ${val_rmse:,.2f})")
        
        # Early stopping check
        if early_stopping(val_rmse):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED")
    print("="*80)
    print(f"\n🏆 Best Results:")
    print(f"  Val RMSE: ${best_val_rmse:>12,.2f}")
    print(f"  Val R²:   {best_val_r2:>6.4f}")
    print(f"\n💾 Model saved to: {MODELS_DIR / 'best_image_model.pth'}")
    
    # Load best model
    checkpoint = torch.load(MODELS_DIR / 'best_image_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, {
        'val_rmse': best_val_rmse,
        'val_r2': best_val_r2,
        'train_rmse': checkpoint['train_rmse'],
        'train_r2': checkpoint['train_r2']
    }

def predict_with_image_model(model, df, price_log_mean=None, price_log_std=None):
    """
    Generate predictions using trained image model

    Args:
        model: Trained PyTorch model
        df: Dataframe with image_path and image_exists columns
        price_log_mean: Mean log(price) for denormalization (if None, loads from checkpoint)
        price_log_std: Std log(price) for denormalization (if None, loads from checkpoint)

    Returns:
        ids: List of property IDs
        predictions: List of predicted prices (in actual price scale, not log)
    """
    print(f"\n🔮 Generating predictions...")

    # Load normalization stats if not provided
    if price_log_mean is None or price_log_std is None:
        checkpoint_path = MODELS_DIR / 'best_image_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            price_log_mean = checkpoint.get('price_log_mean', 0)
            price_log_std = checkpoint.get('price_log_std', 1)
            print(f"  Loaded normalization: log_mean={price_log_mean:.4f}, log_std={price_log_std:.4f}")
        else:
            price_log_mean = 0
            price_log_std = 1
            print("  ⚠️  No checkpoint found, using log_mean=0, log_std=1")

    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    model.eval()

    # Create dataset and loader
    dataset = PropertyImageDataset(df, get_val_transforms(), price_log_mean, price_log_std)
    loader = DataLoader(
        dataset,
        batch_size=IMAGE_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print(f"  Total images: {len(dataset):,}")
    print(f"  Batches: {len(loader)}")

    predictions = []
    ids = []

    with torch.no_grad():
        for images, _, batch_ids in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)

            # Denormalize log predictions and convert to actual prices
            log_preds = outputs.cpu().numpy() * price_log_std + price_log_mean
            price_preds = np.exp(log_preds)

            predictions.extend(price_preds)
            ids.extend(batch_ids.numpy())

    print(f"✅ Generated {len(predictions):,} predictions")
    print(f"  Price range: ${min(predictions):,.2f} - ${max(predictions):,.2f}")

    return ids, predictions

def load_trained_model(checkpoint_path=None):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file (default: best_image_model.pth)
    
    Returns:
        model: Loaded model
        metrics: Dictionary with saved metrics
    """
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / 'best_image_model.pth'
    
    print(f"\n📂 Loading model from: {checkpoint_path}")
    
    # Initialize model
    model = ImageRegressionModel(freeze_ratio=0.8)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics = {
        'epoch': checkpoint['epoch'],
        'val_rmse': checkpoint['val_rmse'],
        'val_r2': checkpoint['val_r2'],
        'train_rmse': checkpoint.get('train_rmse', None),
        'train_r2': checkpoint.get('train_r2', None),
        'price_log_mean': checkpoint.get('price_log_mean', 0),
        'price_log_std': checkpoint.get('price_log_std', 1),
    }
    
    print(f"✅ Model loaded successfully")
    print(f"  Epoch: {metrics['epoch']}")
    print(f"  Val RMSE: ${metrics['val_rmse']:,.2f}")
    print(f"  Val R²: {metrics['val_r2']:.4f}")
    
    return model, metrics