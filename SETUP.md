# 🛠️ Setup Guide

Complete setup instructions for the Satellite Property Valuation project.

---

## Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU (recommended) or Apple Silicon with MPS support
- **RAM**: 16GB minimum
- **Disk Space**: 10GB for code + models (data not included in repo)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

**Core Dependencies**:
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
opencv-python-headless>=4.8.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
tqdm>=4.65.0
```

### 4. Data Setup

**IMPORTANT**: The data files are NOT included in the repository (too large). You need to obtain them separately.

#### Required Data Files

Place the following files in the appropriate directories:

```
Approach_4/
└── data/
    └── raw/
        ├── train.xlsx              # Training data (download separately)
        ├── test.xlsx               # Test data (download separately)
        └── train_images_19/        # Satellite images (download separately)
            ├── 1000102_z19.jpg
            ├── 1001200050_z19.jpg
            └── ... (19,000+ images)
```

#### Download Instructions

1. **Excel Files**: Contact competition organizers for `train.xlsx` and `test.xlsx`
2. **Satellite Images**: Use the data fetcher or download from provided source
3. **Verify**: Ensure ~21,600 training samples and ~19,000 images

#### Directory Structure After Data Setup

```bash
Approach_4/data/raw/
├── train.xlsx              # 21,613 rows
├── test.xlsx               # 10,806 rows
└── train_images_19/        # 19,452 images
    ├── 1000102_z19.jpg
    ├── 1001200050_z19.jpg
    └── ...
```

---

## Quick Verification

### Test Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import albumentations; print(f'albumentations: {albumentations.__version__}')"
```

### Test GPU/MPS

```bash
# For CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# For Apple Silicon
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Running Approach 4 (Final Solution)

### Step 1: Data Preprocessing

```bash
cd Approach_4
jupyter notebook preprocessing.ipynb
```

**What this does**:
1. Loads `train.xlsx` and `test.xlsx`
2. Matches properties to satellite images using lat/long
3. Engineers 21 features from raw data
4. Applies log-transformation to price
5. Creates stratified train/val/test splits (70/15/15)
6. Saves processed data to `data/processed/`

**Expected Output**:
```
data/processed/
├── train_processed.csv
├── val_processed.csv
├── test_processed.csv
└── final_test_processed.csv
```

**Time**: ~5-10 minutes

### Step 2: Train Model

```bash
jupyter notebook model_training.ipynb
```

**What this does**:
1. Loads processed data
2. Initializes ConvNeXt-Small backbone (pretrained)
3. Trains with cross-attention fusion
4. Generates GradCAM explainability
5. Saves best model and predictions

**Expected Output**:
```
outputs/
├── best_model.pth          # Trained model (~100MB)
├── submission.csv          # Final predictions
└── training_logs.txt

Explainability/
└── test/
    ├── test_*.png          # GradCAM visualizations
    └── ...
```

**Time**: 1-2 hours on GPU (6-8 epochs with early stopping)

### Step 3: Evaluate & Submit

The model automatically:
- Computes metrics on validation set
- Generates predictions for test set
- Saves `submission.csv` in required format

```bash
# Check final metrics
tail outputs/training_logs.txt

# Inspect submission file
head outputs/submission.csv
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1**: Reduce batch size
```python
# In model_training.ipynb, modify:
config.BATCH_SIZE = 16  # or even 8
```

**Solution 2**: Use smaller image size
```python
config.IMG_SIZE = 192  # instead of 224
```

**Solution 3**: Enable gradient checkpointing
```python
model.use_gradient_checkpointing = True
```

### Issue: CUDA/MPS Not Detected

**CUDA (NVIDIA)**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**MPS (Apple Silicon)**:
```bash
# Should work out of the box on macOS 12.3+
# If not, update PyTorch:
pip install --upgrade torch torchvision
```

### Issue: Missing Images

**Error**: `FileNotFoundError: Image not found`

**Solution**:
1. Verify images are in `Approach_4/data/raw/train_images_19/`
2. Check filename format: `{property_id}_z19.jpg`
3. Re-run preprocessing to rebuild image index

### Issue: Slow Training

**Solution 1**: Increase num_workers
```python
config.NUM_WORKERS = 8  # or higher
```

**Solution 2**: Use mixed precision training
```python
# Add to training loop:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images, tabular)
    loss = criterion(outputs, targets)
```

**Solution 3**: Freeze more backbone layers
```python
config.FREEZE_RATIO = 0.75  # instead of 0.67
```

### Issue: Poor Convergence

**Symptom**: Validation loss not decreasing

**Solution 1**: Adjust learning rate
```python
config.LEARNING_RATE = 5e-5  # lower
# or
config.LEARNING_RATE = 2e-4  # higher
```

**Solution 2**: Add warmup
```python
config.WARMUP_EPOCHS = 3
```

**Solution 3**: Check data preprocessing
```bash
# Verify price distribution
import pandas as pd
df = pd.read_csv('data/processed/train_processed.csv')
df['price'].describe()
df['price_log'].describe()  # Should be ~12-16
```

---

## Running Other Approaches

### Approach 1: Late Fusion

```bash
cd Approach_1/notebooks

# Run in sequence:
jupyter notebook 01_eda_and_preprocessing.ipynb
jupyter notebook 02_image_model_training.ipynb
jupyter notebook 03_tabular_model_training.ipynb
jupyter notebook 04_fusion_and_evaluation.ipynb
```

### Approach 2: Cross-Modal Attention

```bash
cd Approach_2
jupyter notebook notebooks/train_attention_model.ipynb

# Optional: Monitor with MLflow
mlflow ui --backend-store-uri reports/mlruns
# Open http://localhost:5000
```

### Approach 3: Spatial Attention

```bash
cd Approach_3
jupyter notebook model_training.ipynb
```

---

## Development Tips

### Jupyter Kernel Selection

Ensure you're using the correct kernel:
```bash
# Install kernel
python -m ipykernel install --user --name=satellite-valuation

# In Jupyter, select: Kernel > Change Kernel > satellite-valuation
```

### Git Workflow

```bash
# Check what will be committed
git status

# The .gitignore excludes:
# - All data files (train.xlsx, images, CSVs)
# - Large explainability folders (4GB validation GradCAMs)
# - MLflow artifacts
# - Virtual environments
# - Checkpoints (except best_model.pth)

# Best models ARE included:
# - Approach_*/outputs/best_model.pth
# - Approach_*/models/best_*.pth

# Add and commit
git add .
git commit -m "Your message"
git push origin main
```

### Reproducing Results

For exact reproducibility:

1. **Set seeds** (already done in code):
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

2. **Use same data split** (saved in `data/processed/`)

3. **Same hyperparameters** (defined in `config.py`)

4. **Same environment**:
```bash
# Save your environment
pip freeze > requirements_exact.txt

# Recreate later
pip install -r requirements_exact.txt
```

---

## Hardware Recommendations

### Minimum Specs
- **CPU**: 4 cores
- **RAM**: 16GB
- **GPU**: 4GB VRAM (or MPS on M1/M2)
- **Storage**: 10GB

### Recommended Specs
- **CPU**: 8+ cores
- **RAM**: 32GB
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 20GB (SSD)

### Training Times

| Hardware | Preprocessing | Training | Total |
|----------|--------------|----------|-------|
| M1 Mac (MPS) | 8 min | 2 hours | ~2h 10m |
| RTX 3070 | 6 min | 1 hour | ~1h 6m |
| RTX 4090 | 5 min | 30 min | ~35m |
| CPU only | 10 min | 8-12 hours | ~12h |

---

## Next Steps

1. ✅ Complete installation
2. ✅ Download and setup data
3. ✅ Run preprocessing
4. ✅ Train Approach 4 model
5. ✅ Generate predictions
6. 📊 Analyze results
7. 🚀 Submit to competition

---

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Check README.md and code comments
- **Updates**: Pull latest changes regularly

```bash
git pull origin main
```

---

**Happy modeling! 🚀**
