"""
Data Preprocessing Module with Lat/Long-Only Image Matching
Images are named: img_{seq}_{lat}_{long}.png
We match ONLY by lat/long coordinates (ignore the seq number)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

from src.config import *
from src.utils import *

def match_image_by_latlong(row, images_dir, image_lookup):
    """
    Match image using ONLY lat/long coordinates (ignore ID in filename)
    
    Args:
        row: DataFrame row with 'lat' and 'long' columns
        images_dir: Directory containing images
        image_lookup: Dictionary mapping (lat_str, long_str) -> image_path
    
    Returns:
        (exists, path): Tuple of (boolean, path_string or None)
    """
    lat = row['lat']
    long = row['long']
    
    # Convert lat/long to filename format
    lat_str, long_str = format_latlong_for_filename(lat, long)
    
    # Look up in pre-built index
    key = (lat_str, long_str)
    if key in image_lookup:
        return True, str(image_lookup[key])
    
    return False, None

def build_image_lookup(images_dir):
    """
    Build a lookup dictionary mapping (lat_str, long_str) -> image_path
    This allows O(1) lookups instead of scanning directory for each property
    """
    print(f"📸 Building image lookup index from: {images_dir}")
    
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"❌ Directory not found: {images_dir}")
        return {}
    
    image_lookup = {}
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    # Get all image files
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} total image files")
    
    if len(image_files) == 0:
        print(f"⚠️  No images found in {images_dir}")
        print(f"    Extensions searched: {extensions}")
        return {}
    
    # Parse filenames to extract lat/long
    # Expected format: img_01957_47_735500_-122_095000.png
    for img_path in image_files:
        filename = img_path.stem  # Without extension
        parts = filename.split('_')
        
        # Format: img _ SEQ _ LAT_PART1 _ LAT_PART2 _ LONG_PART1 _ LONG_PART2
        # Example: img _ 01957 _ 47 _ 735500 _ -122 _ 095000
        if len(parts) == 4:
            try:
                # Reconstruct lat and long strings
                lat = float(parts[2])
                long = float(parts[3])
                lat_str, long_str = format_latlong_for_filename(lat, long)
                image_lookup[(lat_str, long_str)] = img_path
                
               
            except (IndexError, ValueError) as e:
                # Skip malformed filenames
                continue
    
    print(f"✅ Indexed {len(image_lookup)} unique lat/long locations")
    
    if len(image_lookup) > 0:
        # Show sample
        sample_key = list(image_lookup.keys())[0]
        sample_path = image_lookup[sample_key]
        print(f"Sample: {sample_key} -> {sample_path.name}")
    
    return image_lookup

def load_data():
    """Load train and test datasets"""
    print("\n📂 Loading data...")
    
    if not check_data_files(TRAIN_FILE, TEST_FILE, IMAGES_DIR):
        raise FileNotFoundError("Required data files not found! Please check setup instructions.")
    
    train_df = pd.read_excel(TRAIN_FILE)
    test_df = pd.read_excel(TEST_FILE)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def check_images_availability(df, images_dir):
    """Check which properties have satellite images using lat/long matching"""
    
    print(f"\n📸 Matching images using lat/long coordinates...")
    print(f"Image directory: {images_dir}")
    
    # Build image lookup index first (much faster!)
    image_lookup = build_image_lookup(images_dir)
    
    if len(image_lookup) == 0:
        print("❌ No images indexed. Cannot proceed.")
        df['image_exists'] = False
        df['image_path'] = None
        return df
    
    image_exists = []
    image_paths = []
    
    from tqdm import tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching properties"):
        exists, path = match_image_by_latlong(row, images_dir, image_lookup)
        image_exists.append(exists)
        image_paths.append(path)
    
    df['image_exists'] = image_exists
    df['image_path'] = image_paths
    
    n_with_images = sum(image_exists)
    print(f"\n✅ Properties with images: {n_with_images}/{len(df)} ({n_with_images/len(df)*100:.2f}%)")
    
    if n_with_images > 0:
        # Show sample match
        matched_idx = image_exists.index(True)
        sample_row = df.iloc[matched_idx]
        print(f"\n✅ Sample match:")
        print(f"   Property ID: {sample_row['id']}")
        print(f"   Lat/Long: {sample_row['lat']:.6f}, {sample_row['long']:.6f}")
        print(f"   Image: {Path(sample_row['image_path']).name}")
    else:
        print("\n⚠️  No images matched!")
        print(f"Debug info:")
        print(f"  - Images indexed: {len(image_lookup)}")
        print(f"  - Sample from data: lat={df.iloc[0]['lat']:.6f}, long={df.iloc[0]['long']:.6f}")
        lat_str, long_str = format_latlong_for_filename(df.iloc[0]['lat'], df.iloc[0]['long'])
        print(f"  - Looking for pattern: *_{lat_str}_{long_str}.*")
        
        if len(image_lookup) > 0:
            print(f"  - Sample from images: {list(image_lookup.keys())[0]}")
    
    return df

def engineer_features(df):
    """Apply comprehensive feature engineering"""
    df = df.copy()
    
    # Temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S', errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Age features
    current_year = 2025
    df['property_age'] = current_year - df['yr_built']
    df['years_since_renovation'] = current_year - df['yr_renovated']
    df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
    df.loc[df['yr_renovated'] == 0, 'years_since_renovation'] = df.loc[df['yr_renovated'] == 0, 'property_age']
    
    # Size features
    df['price_per_sqft'] = df.get('price', 0) / (df['sqft_living'] + 1)
    df['sqft_living_to_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
    df['sqft_above_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)
    df['sqft_basement_ratio'] = df['sqft_basement'] / (df['sqft_living'] + 1)
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    
    # Room features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['bedroom_to_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
    df['sqft_per_bedroom'] = df['sqft_living'] / (df['bedrooms'] + 1)
    df['sqft_per_bathroom'] = df['sqft_living'] / (df['bathrooms'] + 0.1)
    
    # Quality indicators
    df['is_luxury'] = ((df['grade'] >= 11) | (df.get('price', 0) > 1000000)).astype(int)
    df['high_condition'] = (df['condition'] >= 4).astype(int)
    df['has_view'] = (df['view'] > 0).astype(int)
    
    # Location features
    df['lat_long_interaction'] = df['lat'] * df['long']
    
    # Neighbor comparison
    df['sqft_living_vs_neighbors'] = df['sqft_living'] - df['sqft_living15']
    df['sqft_lot_vs_neighbors'] = df['sqft_lot'] - df['sqft_lot15']
    
    # Log transforms
    df['log_sqft_living'] = np.log1p(df['sqft_living'])
    df['log_sqft_lot'] = np.log1p(df['sqft_lot'])
    df['log_sqft_living15'] = np.log1p(df['sqft_living15'])
    df['log_sqft_lot15'] = np.log1p(df['sqft_lot15'])
    
    return df

def create_splits(train_df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Create train/val/test splits"""
    set_seed(random_state)
    
    # First split: train+val vs test
    train_val_df, test_split = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=pd.cut(train_df['price'], bins=5, labels=False)
    )
    
    # Second split: train vs val
    val_proportion = val_size / (train_size + val_size)
    train_split, val_split = train_test_split(
        train_val_df,
        test_size=val_proportion,
        random_state=random_state,
        stratify=pd.cut(train_val_df['price'], bins=5, labels=False)
    )
    
    print("\n" + "="*80)
    print("DATA SPLITS")
    print("="*80)
    print(f"Training set:   {len(train_split):,} ({len(train_split)/len(train_df)*100:.1f}%)")
    print(f"  With images:  {train_split['image_exists'].sum():,} ({train_split['image_exists'].mean()*100:.1f}%)")
    print(f"\nValidation set: {len(val_split):,} ({len(val_split)/len(train_df)*100:.1f}%)")
    print(f"  With images:  {val_split['image_exists'].sum():,} ({val_split['image_exists'].mean()*100:.1f}%)")
    print(f"\nTest set:       {len(test_split):,} ({len(test_split)/len(train_df)*100:.1f}%)")
    print(f"  With images:  {test_split['image_exists'].sum():,} ({test_split['image_exists'].mean()*100:.1f}%)")
    print("="*80)
    
    return train_split, val_split, test_split

def save_processed_data(train_df, val_df, test_df, final_test_df):
    """Save processed datasets"""
    
    train_df.to_csv(PROCESSED_DATA_DIR / 'train_processed.csv', index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / 'val_processed.csv', index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / 'test_processed.csv', index=False)
    final_test_df.to_csv(PROCESSED_DATA_DIR / 'final_test_processed.csv', index=False)
    
    np.save(PROCESSED_DATA_DIR / 'train_ids_with_images.npy', 
            train_df[train_df['image_exists']]['id'].values)
    np.save(PROCESSED_DATA_DIR / 'val_ids_with_images.npy',
            val_df[val_df['image_exists']]['id'].values)
    np.save(PROCESSED_DATA_DIR / 'test_ids_with_images.npy',
            test_df[test_df['image_exists']]['id'].values)
    
    print("\n✅ Processed data saved to:", PROCESSED_DATA_DIR)

def load_processed_data():
    """Load previously processed data"""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_processed.csv')
    val_df = pd.read_csv(PROCESSED_DATA_DIR / 'val_processed.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_processed.csv')

    # Try to load final_test if it exists, otherwise use test_df
    final_test_path = PROCESSED_DATA_DIR / 'final_test_processed.csv'
    if final_test_path.exists():
        final_test_df = pd.read_csv(final_test_path)
    else:
        print(f"⚠️  Warning: {final_test_path} not found, using test_processed.csv as final_test")
        final_test_df = test_df.copy()

    return train_df, val_df, test_df, final_test_df

def run_complete_preprocessing():
    """Run complete preprocessing pipeline"""
    print("\n" + "="*80)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*80)
    
    train_df, test_df = load_data()
    
    print("\n📸 Checking image availability...")
    train_df = check_images_availability(train_df, IMAGES_DIR)
    test_df = check_images_availability(test_df, IMAGES_DIR)
    
    print("\n⚙️  Engineering features...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    print(f"Total features: {train_df.shape[1]}")
    
    print("\n✂️  Creating data splits...")
    train_split, val_split, test_split = create_splits(
        train_df, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_STATE
    )
    
    print("\n💾 Saving processed data...")
    save_processed_data(train_split, val_split, test_split, test_df)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETED!")
    print("="*80)
    
    return train_split, val_split, test_split, test_df