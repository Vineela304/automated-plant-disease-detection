#!/usr/bin/env python3
"""
Download pre-trained model for the Plant Disease Detection project
This allows collaborators to use the app without retraining
"""

import os
import requests
from pathlib import Path

def download_model():
    """Download the pre-trained model"""
    print("🤖 Downloading pre-trained model...")
    
    # You would replace this with your actual model download link
    # For now, I'll show the structure
    
    model_info = {
        "h5_model": {
            "url": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID",
            "filename": "trained_plant_disease_model.h5",
            "size": "~50MB"
        }
    }
    
    print("📋 Available pre-trained models:")
    print(f"   - {model_info['h5_model']['filename']} ({model_info['h5_model']['size']})")
    
    # Check if model already exists
    if os.path.exists(model_info['h5_model']['filename']):
        print(f"✅ Model already exists: {model_info['h5_model']['filename']}")
        return True
    
    print("\n💡 To use pre-trained model:")
    print("1. Contact the project owner for the model file")
    print("2. Or use the Google Drive link in README.md")
    print("3. Place the model file in this directory")
    print("\n🏃‍♂️ Alternative: Train your own model (2-4 hours)")
    print("   Run: python Train_plant_disease.ipynb")
    
    return False

def create_demo_model():
    """Create a demo model for testing the interface"""
    print("\n🎭 Creating demo model for interface testing...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create a simple model with the same structure
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(38, activation='softmax')  # 38 plant disease classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save as demo model
        model.save("demo_plant_disease_model.h5")
        print("✅ Demo model created: demo_plant_disease_model.h5")
        print("⚠️  NOTE: This is untrained - predictions will be random!")
        print("💡 Good for testing the interface, not for real predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create demo model: {e}")
        return False

if __name__ == "__main__":
    print("🌱 Plant Disease Detection - Model Setup")
    print("=" * 50)
    
    if not download_model():
        print("\n🤔 No pre-trained model found.")
        response = input("Create demo model for interface testing? (y/n): ")
        
        if response.lower() == 'y':
            create_demo_model()
            print("\n🎉 Demo setup complete!")
            print("📝 Update main.py to use 'demo_plant_disease_model.h5'")
        else:
            print("\n💡 You'll need to either:")
            print("1. Get the pre-trained model from project owner")
            print("2. Train your own model using the training notebook")
