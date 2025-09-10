#!/usr/bin/env python3
"""
Script to recover trained weights from existing model files
and transfer them to a working model
"""

import tensorflow as tf
import os
import numpy as np

def test_model_loading(model_path):
    """Test if a model can be loaded successfully"""
    try:
        print(f"🔍 Testing: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ SUCCESS: {model_path}")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Total params: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ FAILED: {model_path}")
        print(f"   - Error: {str(e)[:100]}...")
        return None

def create_compatible_model():
    """Create a model with the exact same architecture as the original"""
    print("🔄 Creating compatible model architecture...")
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1500, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(38, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def recover_weights():
    """Try to recover weights from existing models"""
    print("🚀 Starting weight recovery process...\n")
    
    # List of model files to try
    model_files = [
        "trained_model.keras",
        "trained_plant_disease_model.h5", 
        "trained_plant_disease_model.keras"
    ]
    
    working_model = None
    
    # Test each model file
    for model_file in model_files:
        if os.path.exists(model_file):
            model = test_model_loading(model_file)
            if model is not None:
                working_model = model
                working_file = model_file
                break
        else:
            print(f"⚠️ File not found: {model_file}")
    
    if working_model is None:
        print("\n❌ No working model found!")
        return False
    
    print(f"\n🎉 Found working model: {working_file}")
    
    # Create a fresh compatible model
    fresh_model = create_compatible_model()
    
    # Try to transfer weights
    try:
        print("🔄 Transferring weights...")
        fresh_model.set_weights(working_model.get_weights())
        print("✅ Weights transferred successfully!")
        
        # Save the recovered model
        fresh_model.save("trained_plant_disease_model_recovered.h5")
        print("✅ Saved recovered model as: trained_plant_disease_model_recovered.h5")
        
        # Test the recovered model
        print("\n🧪 Testing recovered model...")
        test_input = np.random.random((1, 128, 128, 3))
        prediction = fresh_model.predict(test_input, verbose=0)
        print(f"✅ Model prediction test passed! Output shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight transfer failed: {e}")
        
        # Try partial weight recovery
        print("🔄 Attempting partial weight recovery...")
        try:
            old_weights = working_model.get_weights()
            fresh_weights = fresh_model.get_weights()
            
            print(f"Original model has {len(old_weights)} weight arrays")
            print(f"Fresh model has {len(fresh_weights)} weight arrays")
            
            # Transfer compatible weights
            for i, (old_w, fresh_w) in enumerate(zip(old_weights, fresh_weights)):
                if old_w.shape == fresh_w.shape:
                    fresh_weights[i] = old_w
                    print(f"✅ Transferred weight {i}: {old_w.shape}")
                else:
                    print(f"⚠️ Skipped weight {i}: shape mismatch {old_w.shape} vs {fresh_w.shape}")
            
            fresh_model.set_weights(fresh_weights)
            fresh_model.save("trained_plant_disease_model_partial_recovery.h5")
            print("✅ Saved partially recovered model")
            return True
            
        except Exception as e2:
            print(f"❌ Partial recovery also failed: {e2}")
            return False

if __name__ == "__main__":
    success = recover_weights()
    
    if success:
        print("\n🎉 Weight recovery completed!")
        print("📝 The recovered model should now work in your Streamlit app.")
        print("💡 Update main.py to use 'trained_plant_disease_model_recovered.h5'")
    else:
        print("\n❌ Weight recovery failed!")
        print("💡 You may need to retrain the model or use the dummy model for testing.")
