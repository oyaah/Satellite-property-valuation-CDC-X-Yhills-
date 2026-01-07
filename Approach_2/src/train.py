"""
Training Script for Cross-Modal Attention Fusion Model (Enhanced)
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.pytorch
from pathlib import Path
import matplotlib.pyplot as plt

from attention_model import AttentionFusionModel, count_parameters
from attention_visualizer import AttentionVisualizer, analyze_attention_patterns
from config import *


class WarmupScheduler:
    """Learning rate warmup scheduler"""
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                 (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=10, mode='min', min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        tabular = batch['tabular'].to(device)
        targets = batch['price'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images, tabular)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['TRAINING_CONFIG'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['TRAINING_CONFIG']['grad_clip']
            )
        
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_targets)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            images = batch['image'].to(device)
            tabular = batch['tabular'].to(device)
            targets = batch['price'].to(device)
            ids = batch['id']
            
            # Forward pass
            predictions = model(images, tabular)
            loss = criterion(predictions, targets)
            
            # Track metrics
            running_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_ids.extend(ids.numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_targets), np.array(all_ids)


def compute_metrics(preds, targets, price_log_mean, price_log_std):
    """Compute denormalized metrics (converts from log to actual prices)"""
    # Denormalize log predictions
    preds_log = preds * price_log_std + price_log_mean
    targets_log = targets * price_log_std + price_log_mean

    # Convert from log space to actual prices
    preds_price = np.exp(preds_log)
    targets_price = np.exp(targets_log)

    # Compute metrics on actual prices
    rmse = np.sqrt(mean_squared_error(targets_price, preds_price))
    mae = mean_absolute_error(targets_price, preds_price)
    r2 = r2_score(targets_price, preds_price)

    # MAPE
    mape = np.mean(np.abs((targets_price - preds_price) / targets_price)) * 100

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def train_attention_model(model, dataloaders, stats, config, device):
    """
    Complete training loop for attention fusion model
    
    Args:
        model: AttentionFusionModel
        dataloaders: dict with 'train', 'val', 'test' DataLoaders
        stats: dict with normalization statistics
        config: configuration dict
        device: torch device
    
    Returns:
        model: trained model
        history: training history
        best_metrics: best validation metrics
    """
    print("\n" + "="*80)
    print("TRAINING CROSS-MODAL ATTENTION FUSION MODEL")
    print("="*80)
    
    # Print model info
    param_info = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"  Total:     {param_info['total']:>12,}")
    print(f"  Trainable: {param_info['trainable']:>12,} ({param_info['trainable_pct']:.1f}%)")
    print(f"  Frozen:    {param_info['frozen']:>12,}")
    
    if 'components' in param_info:
        print(f"\n📊 Component Breakdown:")
        for comp_name, comp_params in param_info['components'].items():
            print(f"  {comp_name:25s}: {comp_params:>12,}")
    
    # Training configuration
    train_config = config['TRAINING_CONFIG']
    num_epochs = train_config['num_epochs']
    learning_rate = train_config['learning_rate']
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Weight decay: {train_config['weight_decay']:.2e}")
    print(f"  Early stopping patience: {train_config['early_stopping_patience']}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config['scheduler_factor'],
        patience=train_config['scheduler_patience'],
        min_lr=train_config['min_lr']
    )
    
    print(f"  LR Scheduler: ReduceLROnPlateau (factor={train_config['scheduler_factor']}, patience={train_config['scheduler_patience']})")
    
    # Warmup scheduler
    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=train_config['warmup_epochs'],
        initial_lr=train_config['warmup_lr'],
        target_lr=learning_rate
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config['early_stopping_patience'],
        mode='min'
    )
    
    # MLflow logging
    mlflow.set_tracking_uri(config['MLFLOW_CONFIG']['tracking_uri'])
    mlflow.set_experiment(config['MLFLOW_CONFIG']['experiment_name'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_r2': [],
        'val_r2': [],
        'learning_rate': []
    }
    
    best_val_rmse = float('inf')
    best_epoch = 0
    best_metrics = None
    
    print(f"\n🚀 Starting training...")
    print("="*80)
    
    with mlflow.start_run(run_name=config['MLFLOW_CONFIG']['run_name']):
        # Log hyperparameters
        mlflow.log_params({
            'model': 'cross_modal_attention',
            'backbone': config['IMAGE_CONFIG']['backbone'],
            'freeze_ratio': config['IMAGE_CONFIG']['freeze_ratio'],
            'batch_size': train_config['batch_size'],
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'attention_heads': config['ATTENTION_CONFIG']['attention_heads'],
        })
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)
            
            # Warmup
            if epoch < train_config['warmup_epochs']:
                warmup_scheduler.step()
            
            # Training
            train_loss, train_preds, train_targets = train_one_epoch(
                model, dataloaders['train'], criterion, optimizer, 
                device, config, epoch
            )
            
            # Validation
            val_loss, val_preds, val_targets, val_ids = validate(
                model, dataloaders['val'], criterion, device
            )
            
            # Compute metrics
            train_metrics = compute_metrics(
                train_preds, train_targets,
                stats['price_log_mean'], stats['price_log_std']
            )
            val_metrics = compute_metrics(
                val_preds, val_targets,
                stats['price_log_mean'], stats['price_log_std']
            )
            
            # Update scheduler (after warmup)
            if epoch >= train_config['warmup_epochs']:
                scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_metrics['rmse'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['train_r2'].append(train_metrics['r2'])
            history['val_r2'].append(val_metrics['r2'])
            history['learning_rate'].append(current_lr)
            
            # Print results
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:>8.4f} | Val Loss: {val_loss:>8.4f}")
            print(f"  Train RMSE: ${train_metrics['rmse']:>12,.2f} | Val RMSE: ${val_metrics['rmse']:>12,.2f}")
            print(f"  Train R²:   {train_metrics['r2']:>8.4f} | Val R²:   {val_metrics['r2']:>8.4f}")
            print(f"  Train MAE:  ${train_metrics['mae']:>12,.2f} | Val MAE:  ${val_metrics['mae']:>12,.2f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'learning_rate': current_lr
            }, step=epoch)
            
            # Save best model
            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                best_epoch = epoch
                best_metrics = val_metrics.copy()

                # Extract only serializable config values (filter out modules and functions)
                serializable_config = {
                    k: v for k, v in config.items()
                    if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))
                }

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2'],
                    'val_mae': val_metrics['mae'],
                    'train_rmse': train_metrics['rmse'],
                    'train_r2': train_metrics['r2'],
                    'price_log_mean': stats['price_log_mean'],
                    'price_log_std': stats['price_log_std'],
                    'tabular_mean': stats['tabular_mean'],
                    'tabular_std': stats['tabular_std'],
                    'config': serializable_config
                }

                torch.save(checkpoint, MODELS_DIR / 'best_attention_model.pth', _use_new_zipfile_serialization=True)
                mlflow.pytorch.log_model(model, "model")
                
                print(f"  ✅ Best model saved! (Val RMSE: ${val_metrics['rmse']:,.2f})")
            
            # Early stopping
            if early_stopping(val_metrics['rmse']):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch was {best_epoch+1}")
                break
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED")
        print("="*80)
        print(f"\n🏆 Best Results (Epoch {best_epoch+1}):")
        print(f"  Val RMSE: ${best_metrics['rmse']:>12,.2f}")
        print(f"  Val R²:   {best_metrics['r2']:>8.4f}")
        print(f"  Val MAE:  ${best_metrics['mae']:>12,.2f}")
        print(f"  Val MAPE: {best_metrics['mape']:>7.2f}%")
        print(f"\n💾 Model saved to: {MODELS_DIR / 'best_attention_model.pth'}")
        
        # Log best metrics
        mlflow.log_metrics({
            'best_val_rmse': best_metrics['rmse'],
            'best_val_r2': best_metrics['r2'],
            'best_val_mae': best_metrics['mae'],
            'best_epoch': best_epoch
        })
        
        # Generate attention visualizations
        print(f"\n🎨 Generating attention visualizations...")
        viz_config = config.get('VISUALIZATION_CONFIG', {})
        if viz_config.get('num_samples', 0) > 0:
            visualizer = AttentionVisualizer(model, device)
            
            save_dir = viz_config.get('save_dir', REPORTS_DIR / 'attention_visualizations')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            visualizer.visualize_batch(
                dataloaders['val'],
                num_samples=viz_config.get('num_samples', 10),
                save_dir=save_dir
            )
            
            # Analyze attention patterns
            if viz_config.get('analyze_patterns', False):
                print(f"\n📊 Analyzing attention patterns...")
                attention_analysis = analyze_attention_patterns(
                    model, dataloaders['val'], device, stats,
                    num_samples=viz_config.get('num_analysis_samples', 100)
                )
                
                # Log analysis to MLflow
                mlflow.log_metrics({
                    'attention_mean_entropy': attention_analysis['mean_entropy'],
                    'attention_std_entropy': attention_analysis['std_entropy'],
                    'attention_mean_max': attention_analysis['mean_max_attention'],
                })
    
    return model, history, best_metrics


def load_checkpoint(checkpoint_path=None):
    """Load trained model from checkpoint"""
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / 'best_attention_model.pth'

    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Reconstruct config (should be saved in checkpoint)
    config = checkpoint.get('config', {})

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val RMSE: ${checkpoint['val_rmse']:,.2f}")
    print(f"  Val R²: {checkpoint['val_r2']:.4f}")

    return checkpoint