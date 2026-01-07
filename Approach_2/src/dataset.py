"""
Dataset and DataLoader for Cross-Modal Attention Model
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class MultimodalPropertyDataset(Dataset):
    """
    Dataset that loads both images and tabular features (uses log-transformed prices)
    """
    def __init__(self, df, tabular_features, image_transform=None,
                 price_log_mean=None, price_log_std=None,
                 tabular_mean=None, tabular_std=None):
        """
        Args:
            df: DataFrame with image_path, price_log, and tabular features
            tabular_features: List of column names for tabular features
            image_transform: Albumentations transform for images
            price_log_mean, price_log_std: For log(price) normalization
            tabular_mean, tabular_std: For tabular feature normalization
        """
        # Only keep rows with images
        self.df = df[df['image_exists']].reset_index(drop=True)
        self.tabular_features = tabular_features
        self.image_transform = image_transform

        # Log(price) normalization
        if price_log_mean is None and 'price_log' in self.df.columns:
            self.price_log_mean = self.df['price_log'].mean()
            self.price_log_std = self.df['price_log'].std()
        else:
            self.price_log_mean = price_log_mean if price_log_mean is not None else 0
            self.price_log_std = price_log_std if price_log_std is not None else 1
        
        # Tabular feature normalization
        tabular_data = self.df[tabular_features].values
        if tabular_mean is None:
            self.tabular_mean = tabular_data.mean(axis=0)
            self.tabular_std = tabular_data.std(axis=0) + 1e-8  # Avoid division by zero
        else:
            self.tabular_mean = tabular_mean
            self.tabular_std = tabular_std
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and transform image
        image = cv2.imread(row['image_path'])
        if image is None:
            raise RuntimeError(f"Failed to load image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.image_transform:
            image = self.image_transform(image=image)['image']
        
        # Get tabular features and normalize
        tabular_data = row[self.tabular_features].values.astype(np.float32)
        tabular_normalized = (tabular_data - self.tabular_mean) / self.tabular_std
        tabular_tensor = torch.tensor(tabular_normalized, dtype=torch.float32)
        
        # Get and normalize log(price)
        if 'price_log' in row:
            price_log = (row['price_log'] - self.price_log_mean) / self.price_log_std
        else:
            price_log = 0.0

        price_tensor = torch.tensor(price_log, dtype=torch.float32)
        
        return {
            'image': image,
            'tabular': tabular_tensor,
            'price': price_tensor,
            'id': row['id']
        }


def get_train_transforms(img_size, config):
    """Training augmentation pipeline"""
    aug_config = config['AUGMENTATION_CONFIG']
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=aug_config['horizontal_flip']),
        A.VerticalFlip(p=aug_config.get('vertical_flip', 0.3)),
        A.RandomRotate90(p=aug_config['rotate_90']),
        A.ShiftScaleRotate(**aug_config['shift_scale_rotate']),
        A.RandomBrightnessContrast(**aug_config['brightness_contrast']),
        A.GaussNoise(**aug_config['gauss_noise']),
        A.Normalize(**aug_config['normalize']),
        ToTensorV2()
    ])


def get_val_transforms(img_size, config):
    """Validation/test transform (no augmentation)"""
    aug_config = config['AUGMENTATION_CONFIG']
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(**aug_config['normalize']),
        ToTensorV2()
    ])


def create_dataloaders(train_df, val_df, test_df, tabular_features, config):
    """
    Create train, validation, and test dataloaders (uses log-transformed prices)

    Returns:
        dataloaders: dict with 'train', 'val', 'test' DataLoaders
        stats: dict with normalization statistics
    """
    img_size = config['IMAGE_CONFIG']['img_size']
    batch_size = config['TRAINING_CONFIG']['batch_size']
    num_workers = config['NUM_WORKERS']

    # Calculate normalization stats from training data (using log-transformed prices)
    price_log_mean = train_df['price_log'].mean()
    price_log_std = train_df['price_log'].std()

    tabular_data = train_df[tabular_features].values
    tabular_mean = tabular_data.mean(axis=0)
    tabular_std = tabular_data.std(axis=0) + 1e-8

    # Create datasets
    train_dataset = MultimodalPropertyDataset(
        train_df, tabular_features,
        image_transform=get_train_transforms(img_size, config),
        price_log_mean=price_log_mean, price_log_std=price_log_std,
        tabular_mean=tabular_mean, tabular_std=tabular_std
    )

    val_dataset = MultimodalPropertyDataset(
        val_df, tabular_features,
        image_transform=get_val_transforms(img_size, config),
        price_log_mean=price_log_mean, price_log_std=price_log_std,
        tabular_mean=tabular_mean, tabular_std=tabular_std
    )

    test_dataset = MultimodalPropertyDataset(
        test_df, tabular_features,
        image_transform=get_val_transforms(img_size, config),
        price_log_mean=price_log_mean, price_log_std=price_log_std,
        tabular_mean=tabular_mean, tabular_std=tabular_std
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    stats = {
        'price_log_mean': price_log_mean,
        'price_log_std': price_log_std,
        'tabular_mean': tabular_mean,
        'tabular_std': tabular_std
    }

    print(f"\n📊 Dataset Sizes:")
    print(f"  Train:      {len(train_dataset):>6,} samples, {len(train_loader):>4} batches")
    print(f"  Validation: {len(val_dataset):>6,} samples, {len(val_loader):>4} batches")
    print(f"  Test:       {len(test_dataset):>6,} samples, {len(test_loader):>4} batches")
    print(f"\n📊 Normalization Stats:")
    print(f"  Log(Price) mean: {price_log_mean:>8.4f}")
    print(f"  Log(Price) std:  {price_log_std:>8.4f}")
    print(f"  Tabular features: {len(tabular_features)}")
    print(f"\n📝 Training Strategy:")
    print(f"  1. Train model to predict log(price)")
    print(f"  2. At inference: y_pred = exp(y_pred_log)")
    print(f"  3. Evaluate on actual price (not log)")
    
    return dataloaders, stats


def prepare_features(df, config):
    """
    Prepare tabular features from DataFrame
    
    Returns:
        feature_columns: List of feature column names
        df: DataFrame with processed features
    """
    # Get feature lists from config
    numerical_features = config.get('NUMERICAL_FEATURES', [])
    categorical_features = config.get('CATEGORICAL_FEATURES', [])
    log_features = config.get('LOG_FEATURES', [])
    exclude_features = config.get('EXCLUDE_FEATURES', [])
    
    # Combine all feature types
    all_features = []
    
    # Add numerical features that exist
    for feat in numerical_features:
        if feat in df.columns:
            all_features.append(feat)
    
    # Add log features that exist
    for feat in log_features:
        if feat in df.columns:
            all_features.append(feat)
    
    # One-hot encode categorical features
    for feat in categorical_features:
        if feat in df.columns:
            # Simple binary encoding for now
            all_features.append(feat)
    
    # Remove excluded features
    all_features = [f for f in all_features if f not in exclude_features]
    
    # Remove duplicates while preserving order
    seen = set()
    feature_columns = []
    for feat in all_features:
        if feat not in seen and feat in df.columns:
            seen.add(feat)
            feature_columns.append(feat)
    
    print(f"\n📊 Tabular Features Selected: {len(feature_columns)}")
    print(f"  Sample features: {feature_columns[:10]}")
    
    return feature_columns, df
