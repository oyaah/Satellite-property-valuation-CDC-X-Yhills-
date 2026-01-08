# Satellite Imagery-Based Property Price Prediction

**CDC X Yhills Competition 2025-26**

Predicting real estate prices using satellite imagery and tabular property features through multimodal deep learning.

---

## Overview

This project combines two data modalities to predict property prices:
- **Satellite Imagery**: Zoom-20 images capturing neighborhood context, building density, greenery, waterfront access
- **Tabular Features**: 21 property attributes (bedrooms, bathrooms, sqft, grade, condition, location, etc.)

Dataset: King County, Washington housing data with ~13,000 training properties and ~3,200 validation properties.

---

## Approaches Summary

| Approach | Method | Val RMSE | Val R² | Notes |
|----------|--------|----------|--------|-------|
| **Approach 4** | EfficientNet-V2-S + FPN + CBAM (Multimodal) | $143,495 | 0.8182 | Best multimodal approach |
| **Approach 1** | Late Fusion (XGBoost/LightGBM/CatBoost ensemble) | $98,019 | 0.9152 | Best overall but not truly multimodal |
| **Approach 3** | ResNet-50 + Dual Zoom (19+20) + Cross-Attention | $134,190 | 0.8210 | Too computationally expensive |
| **Approach 2** | EfficientNet-B1 + Transformer + Cross-Attention | $149,553 | 0.7931 | Underfitting issues |

---

## Approach 4: Final Multimodal Solution

This is the recommended approach for true multimodal learning where both image and tabular features contribute to predictions.

### Architecture

```
INPUT
├── Satellite Image (224×224, Zoom-20)
│   └── EfficientNet-V2-S (BatchNorm unfrozen only, ~0.5% trainable)
│       ├── Stage 3 features (160 channels)
│       └── Stage 4 features (256 channels)
│           └── FPN-Lite (multi-scale fusion) → 128 channels
│               └── CBAM (channel attention)
│                   └── Global Average Pool → 128-dim vector
│
├── Tabular Features (21 features)
│   └── MLP [128 → 128 → 64] with LayerNorm + GELU
│       └── 64-dim vector
│
└── FUSION
    └── Concatenate [128 + 64 = 192]
        └── MLP [256 → 64 → 1] → log(price) prediction
```

### Validation Metrics

```
Log-Scale (optimization target):
  RMSE_log: 0.2184
  R²_log:   0.8197

Real-Scale (actual prices):
  RMSE:     $143,495
  MAE:      $88,874
  R²:       0.8182
  MAPE:     17.28%
```

### Project Structure

```
Approach_4/
├── preprocessing.ipynb      # Data preparation and feature engineering
├── model_training.ipynb     # Model training, evaluation, explainability
├── data_fetcher.py          # Satellite image download utilities
├── data/
│   ├── raw/
│   │   ├── train.xlsx       # Raw training data
│   │   ├── test.xlsx        # Raw test data
│   │   └── train_images_20/ # Zoom-20 satellite images ({id}_z20.jpg)
│   ├── processed/
│   │   ├── train_processed.csv
│   │   ├── val_processed.csv
│   │   └── test_processed.csv
│   └── test_data_images/
│       └── test_images_20/  # Test set images
├── outputs/
│   ├── best_model.pth       # Trained model checkpoint
│   ├── test_predictions.csv # Final predictions
│   ├── training_history.png
│   └── prediction_scatter.png
└── Explainability/
    ├── validation/          # Grad-CAM++ visualizations for val set
    └── test/                # Grad-CAM++ visualizations for test set
```

### How to Run

**Prerequisites**: Python 3.8+, PyTorch 2.0+, 16GB RAM, GPU recommended (MPS/CUDA)

**Step 1: Setup**
```bash
cd /Users/yashbansal/Documents/cdc
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Prepare Data**
```bash
cd Approach_4
jupyter notebook preprocessing.ipynb
```
Run all cells to:
- Load raw Excel files (train.xlsx, test.xlsx)
- Match images by property ID (format: `{id}_z20.jpg`)
- Engineer 21 features from raw data
- Create 80/20 train/val split
- Save processed CSVs to `data/processed/`

**Step 3: Train Model**
```bash
jupyter notebook model_training.ipynb
```
Run all cells to:
- Initialize EfficientNet-V2-S with BatchNorm unfreezing
- Train with Huber loss on log(price)
- Save best model to `outputs/best_model.pth`
- Generate training curves and scatter plots

**Step 4: Generate Outputs**
The notebook automatically creates:
- `outputs/test_predictions.csv` - Final price predictions
- `Explainability/validation/` - Grad-CAM++ heatmaps showing what image regions influence predictions
- `Explainability/test/` - Grad-CAM++ for test set

### Key Design Decisions

- **Zoom-20 only**: Higher resolution captures house-specific features (pools, roof type, landscaping)
- **BatchNorm-only unfreezing**: Cheap domain adaptation without overfitting (~0.5% params trainable)
- **FPN-Lite**: Fuses multi-scale features from backbone stages 3 and 4
- **CBAM**: Channel attention helps model focus on relevant feature channels
- **Log-price training**: Handles skewed price distribution, Huber loss for robustness
- **Image size 224×224**: Balance between detail and training speed on MPS

---

## Approach 1: Late Fusion (Best Metrics, Not Multimodal)

Trains image and tabular models separately, then combines predictions. Achieved best raw metrics but adaptive weighting gave 100% weight to tabular, meaning images weren't used.

### Method
1. **Image Model**: EfficientNet-B0, frozen backbone, trained on log(price)
2. **Tabular Models**: XGBoost, LightGBM, CatBoost ensemble
3. **Fusion**: Tested multiple strategies (simple average, weighted, adaptive, max, min)

### Results
```
Individual Models:
  XGBoost:  RMSE $103,460, R² 0.9055
  LightGBM: RMSE $100,812, R² 0.9103
  CatBoost: RMSE $96,759,  R² 0.9173
  Ensemble: RMSE $98,016,  R² 0.9152

Fusion Methods:
  Simple Average (50/50):    RMSE $155,172, R² 0.7874
  Weighted (40/60):          RMSE $137,085, R² 0.8341
  Adaptive Weighted:         RMSE $98,019,  R² 0.9152  (Image: 0%, Tabular: 100%)
  Max Fusion:                RMSE $148,337, R² 0.8057
  Min Fusion:                RMSE $236,751, R² 0.5052
```

### Why Not Final Solution
The adaptive weighting learned to ignore images entirely (0% weight), making this effectively a tabular-only model. For a true multimodal solution, we need architectures that force both modalities to contribute.

### Files
```
Approach_1/notebooks/
├── 01_eda_and_preprocessing.ipynb
├── 02_image_model_training.ipynb
├── 03_tabular_model_training.ipynb
└── 04_fusion_and_evaluation.ipynb
```

---

## Approach 3: Dual-Zoom Cross-Attention

Used both Zoom-19 and Zoom-20 images with ResNet-50 backbone and cross-modal attention.

### Method
- **Backbone**: ResNet-50 (67% trainable, 33% frozen)
- **Images**: Dual zoom levels (19 and 20) for multi-scale context
- **Fusion**: Cross-attention between tabular queries and spatial image features
- **Parameters**: 26.2M total, 17.7M trainable

### Results
```
Validation:
  RMSE_log: 0.2149
  R²_log:   0.8255
  RMSE:     $134,190
  R²:       0.8410
```

### Why Not Final Solution
Too computationally expensive (dual images, large backbone, high trainable params). Training was slow and gains over single-zoom approaches were marginal.

### Files
```
Approach_3/
├── model_training.ipynb
└── outputs/best_model.pth
```

---

## Approach 2: Transformer Cross-Attention

End-to-end attention model with tabular features attending to image feature maps.

### Method
- **Backbone**: EfficientNet-B1 (feature maps 7×7×1280)
- **Tabular**: Transformer encoder with self-attention
- **Fusion**: Multi-head cross-attention (4 heads)

### Results
```
Best (Epoch 24):
  Val RMSE: $149,553
  Val R²:   0.7931
  Val MAE:  $93,069
 
```

### Why Not Final Solution
Underfitting - the model had insufficient capacity or the attention mechanism didn't learn effectively. Attention entropy was high (3.25), indicating diffuse attention rather than focused regions,training R square plateud at 0.6875

### Files
```
Approach_2/notebooks/
└── train_attention_model.ipynb  # With attention visualizations
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
tqdm>=4.65.0
jupyter>=1.0.0
openpyxl>=3.1.0


xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
```

---

