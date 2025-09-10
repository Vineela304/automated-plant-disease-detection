# Quick Model Retrain Script
# Run this if you need to quickly retrain and save your model

import tensorflow as tf
import os

def check_and_retrain_model():
    """
    Check if model exists, if not provide instructions to retrain
    """
    model_files = [
        "trained_plant_disease_model.h5",
        "trained_plant_disease_model.keras"
    ]
    
    model_exists = any(os.path.exists(f) for f in model_files)
    
    if model_exists:
        for file in model_files:
            if os.path.exists(file):
                print(f"✅ Found model: {file}")
                try:
                    model = tf.keras.models.load_model(file)
                    print(f"✅ Model {file} loads successfully!")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
    else:
        print("❌ No trained model found!")
        print("\n📋 To fix this issue:")
        print("1. Open Train_plant_disease.ipynb")
        print("2. Run all cells in the notebook to train the model")
        print("3. The model will be saved automatically in both .h5 and .keras formats")
        print("4. After training, run your main.py again")
        print("\n💡 Make sure you have your dataset folders (train, valid, test) in the correct location!")

if __name__ == "__main__":
    check_and_retrain_model()
