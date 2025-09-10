#!/usr/bin/env python3
"""
Dataset setup script for collaborators
Automatically downloads and sets up the plant disease dataset
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

def download_with_progress(url, filename):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
        print(f"\n✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def setup_dataset():
    """Setup the plant disease dataset"""
    print("🚀 Plant Disease Dataset Setup")
    print("=" * 50)
    
    # Check if dataset already exists
    if os.path.exists("train") and os.path.exists("valid"):
        train_count = len(list(Path("train").rglob("*.jpg")))
        valid_count = len(list(Path("valid").rglob("*.jpg")))
        
        if train_count > 50000 and valid_count > 10000:
            print(f"✅ Dataset already exists!")
            print(f"   - Training images: {train_count:,}")
            print(f"   - Validation images: {valid_count:,}")
            return True
    
    print("📥 Dataset not found. Setting up...")
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. This will download ~2GB of data")
    print("2. You need a Kaggle account")
    print("3. You need Kaggle API token (kaggle.json)")
    print("\n📋 Setup Instructions:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Place it in ~/.kaggle/ folder")
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle API found")
    except ImportError:
        print("❌ Kaggle API not installed")
        print("Installing kaggle...")
        os.system("pip install kaggle")
        
    # Try to download using Kaggle API
    try:
        print("\n🔄 Downloading dataset from Kaggle...")
        os.system("kaggle datasets download -d vipoooool/new-plant-diseases-dataset")
        
        if os.path.exists("new-plant-diseases-dataset.zip"):
            print("🔄 Extracting dataset...")
            with zipfile.ZipFile("new-plant-diseases-dataset.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove("new-plant-diseases-dataset.zip")
            print("✅ Dataset setup complete!")
            return True
        else:
            raise Exception("Download failed")
            
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\n💡 Manual Setup Required:")
        print("1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("2. Click 'Download' button")
        print("3. Extract the zip file in this directory")
        print("4. Ensure you have 'train/' and 'valid/' folders")
        return False

def verify_dataset():
    """Verify dataset is properly set up"""
    print("\n🔍 Verifying dataset...")
    
    required_folders = ["train", "valid"]
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"❌ Missing folder: {folder}")
            return False
        
        image_count = len(list(Path(folder).rglob("*.jpg")))
        print(f"✅ {folder}/: {image_count:,} images")
    
    print("✅ Dataset verification complete!")
    return True

if __name__ == "__main__":
    print("🌱 Plant Disease Detection - Dataset Setup")
    print("=" * 60)
    
    if setup_dataset():
        verify_dataset()
        print("\n🎉 Setup Complete!")
        print("You can now run: streamlit run main.py")
    else:
        print("\n❌ Setup Failed!")
        print("Please follow manual setup instructions in README.md")
        sys.exit(1)
