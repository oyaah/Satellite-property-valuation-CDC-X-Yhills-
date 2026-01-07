"""
Optimized Satellite Image Downloader using Mapbox Static Images API

Parallel downloading of zoom 19 satellite imagery (75m area coverage).
Optimized for fast bulk downloads with configurable worker threads.

Features:
- Parallel downloads with ThreadPoolExecutor
- Automatic resume capability (skips existing images)
- Progress tracking with tqdm
- Thread-safe dataframe updates
"""

import pandas as pd
import requests
from pathlib import Path
import time
from tqdm import tqdm
import hashlib
import os
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class MapboxImageDownloader:
    """Download satellite images from Mapbox Static Images API"""
    
    def __init__(self, access_token, image_size=640):
        """
        Args:
            access_token: Mapbox API access token
            image_size: Image dimension (width = height, max 1280)
        """
        self.access_token = access_token
        self.image_size = image_size
        self.base_url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
        
    def get_image_url(self, longitude, latitude, zoom, retina=True):
        """
        Construct Mapbox Static Image URL
        
        Args:
            longitude: Property longitude
            latitude: Property latitude
            zoom: Zoom level (14-19)
            retina: If True, request @2x resolution
            
        Returns:
            URL string
        """
        retina_suffix = "@2x" if retina else ""
        url = (
            f"{self.base_url}/"
            f"{longitude},{latitude},"
            f"{zoom}/"
            f"{self.image_size}x{self.image_size}{retina_suffix}"
            f"?access_token={self.access_token}"
        )
        return url
    
    def download_image(self, longitude, latitude, zoom, save_path, 
                       max_retries=3, retry_delay=2):
        """
        Download single image from Mapbox
        
        Args:
            longitude: Property longitude
            latitude: Property latitude
            zoom: Zoom level
            save_path: Path to save image
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            
        Returns:
            bool: True if successful, False otherwise
        """
        url = self.get_image_url(longitude, latitude, zoom)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Verify it's actually an image
                    img = Image.open(io.BytesIO(response.content))
                    img.save(save_path)
                    return True
                    
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = retry_delay * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                else:
                    print(f"  Error {response.status_code}: {response.text[:100]}")
                    return False
                    
            except Exception as e:
                print(f"  Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    
        return False
    
    def _download_single_task(self, task):
        """Helper function to download a single image (for parallel processing)"""
        idx, property_id, lon, lat, zoom, save_path = task

        # Skip if already exists
        if save_path.exists():
            return (idx, zoom, str(save_path), True, True)  # idx, zoom, path, success, skipped

        # Download image
        success = self.download_image(lon, lat, zoom, save_path)

        if success:
            return (idx, zoom, str(save_path), True, False)
        else:
            return (idx, zoom, None, False, False)

    def download_multiscale_dataset(self, df, zoom_levels, output_base_dir,
                                   id_column='id', lat_column='lat',
                                   long_column='long', max_workers=10,
                                   max_samples=None):
        """
        Download images at multiple zoom levels for entire dataset (PARALLEL)

        Args:
            df: DataFrame with property data
            zoom_levels: List of zoom levels to download
            output_base_dir: Base directory for images
            id_column: Column name for property ID
            lat_column: Column name for latitude
            long_column: Column name for longitude
            max_workers: Number of parallel download threads (default: 10)
            max_samples: Maximum samples to download (None = all)

        Returns:
            DataFrame with image paths added
        """
        output_base_dir = Path(output_base_dir)

        # Create directories for each zoom level
        zoom_dirs = {}
        for zoom in zoom_levels:
            zoom_dir = output_base_dir / f"zoom_{zoom}"
            zoom_dir.mkdir(parents=True, exist_ok=True)
            zoom_dirs[zoom] = zoom_dir

        # Limit samples if specified
        if max_samples:
            df_download = df.head(max_samples).copy()
        else:
            df_download = df.copy()

        # Add columns for image paths
        for zoom in zoom_levels:
            df_download[f'image_path_z{zoom}'] = None
            df_download[f'image_exists_z{zoom}'] = False

        total_downloads = len(df_download) * len(zoom_levels)
        success_count = 0
        skipped_count = 0

        print(f"\n{'='*80}")
        print(f"STARTING PARALLEL DOWNLOAD")
        print(f"{'='*80}")
        print(f"Properties:      {len(df_download):,}")
        print(f"Zoom level:      {zoom_levels[0]} (75m coverage)")
        print(f"Total images:    {total_downloads:,}")
        print(f"Workers:         {max_workers}")
        print(f"Output dir:      {output_base_dir}")
        print(f"{'='*80}\n")

        # Prepare all download tasks
        tasks = []
        for idx, row in df_download.iterrows():
            property_id = row[id_column]
            lat = row[lat_column]
            lon = row[long_column]

            for zoom in zoom_levels:
                filename = f"{property_id}_z{zoom}.jpg"
                save_path = zoom_dirs[zoom] / filename
                tasks.append((idx, property_id, lon, lat, zoom, save_path))

        # Download in parallel with progress bar
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self._download_single_task, task): task for task in tasks}

            # Process results with progress bar
            with tqdm(total=total_downloads, desc="Downloading images") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, zoom, path, success, skipped = future.result()

                        with lock:
                            if success:
                                df_download.at[idx, f'image_path_z{zoom}'] = path
                                df_download.at[idx, f'image_exists_z{zoom}'] = True
                                success_count += 1
                                if skipped:
                                    skipped_count += 1

                    except Exception as e:
                        print(f"\n  Error in download task: {e}")

                    pbar.update(1)

        print(f"\n{'='*80}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"Success: {success_count}/{total_downloads} ({100*success_count/total_downloads:.1f}%)")
        print(f"Skipped (already exist): {skipped_count}")
        print(f"Downloaded: {success_count - skipped_count}")

        # Print statistics per zoom level
        for zoom in zoom_levels:
            count = df_download[f'image_exists_z{zoom}'].sum()
            print(f"  Zoom {zoom}: {count}/{len(df_download)} images")

        return df_download


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("MAPBOX SATELLITE IMAGE DOWNLOADER (ZOOM 19 - PARALLEL)")
    print("="*80 + "\n")

    # ========== CONFIGURATION ==========

    # IMPORTANT: Set your Mapbox access token here
    MAPBOX_TOKEN = "pk.eyJ1IjoieW9ob29vMTIzNCIsImEiOiJjbWswMjdrYzYwcjJwM2RzYzNodmVjejd3In0.al7RO1XauGezBcV6mOiR6A"

    # Check if token is set
    if MAPBOX_TOKEN == "YOUR_MAPBOX_ACCESS_TOKEN_HERE":
        print("❌ ERROR: Please set your Mapbox access token!")
        print("\n📝 To get a token:")
        print("   1. Go to https://account.mapbox.com/")
        print("   2. Sign up or log in")
        print("   3. Go to 'Access tokens' page")
        print("   4. Copy your default public token or create a new one")
        print("   5. Replace 'YOUR_MAPBOX_ACCESS_TOKEN_HERE' in this script\n")
        return

    # Zoom level configuration (optimized for zoom 19)
    ZOOM_LEVELS = [19]  # 75m area coverage - property details
    
    # Image settings
    IMAGE_SIZE = 1280  # 640x640 pixels (can go up to 1280)
    
    # Download settings
    MAX_SAMPLES = None  # Download all properties (set to number for testing)
    MAX_WORKERS = 30  # Parallel download threads (10-50 recommended, 30 is optimal)

    # Paths
    INPUT_FILE = '/Users/yashbansal/Documents/cdc/Approach_3/data/raw/train.xlsx'
    OUTPUT_DIR = '/Users/yashbansal/Documents/cdc/Approach_3/data/raw/train_images_19'

    # ========== LOAD DATA ==========

    print("📂 Loading dataset...")
    df = pd.read_excel(INPUT_FILE)
    print(f"   Loaded {len(df)} properties")
    print(f"   Columns: {df.columns.tolist()}\n")

    # ========== DOWNLOAD IMAGES (PARALLEL) ==========

    downloader = MapboxImageDownloader(
        access_token=MAPBOX_TOKEN,
        image_size=IMAGE_SIZE
    )

    df_with_images = downloader.download_multiscale_dataset(
        df=df,
        zoom_levels=ZOOM_LEVELS,
        output_base_dir=OUTPUT_DIR,
        max_workers=MAX_WORKERS,
        max_samples=MAX_SAMPLES
    )
    
    # ========== SAVE RESULTS ==========

    output_csv = Path(OUTPUT_DIR) / 'properties_with_images.csv'
    df_with_images.to_csv(output_csv, index=False)
    print(f"\n💾 Saved dataset with image paths to: {output_csv}")

    # ========== SUMMARY ==========

    print("\n" + "="*80)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()