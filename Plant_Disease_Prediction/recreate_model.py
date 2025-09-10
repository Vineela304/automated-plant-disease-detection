#!/usr/bin/env python3
"""
Script to recreate the model using saved training history and architecture
This creates a fresh model file to avoid corruption issues
"""

import tensorflow as tf
import json
import os

def recreate_model():
    """Recreate the CNN model with the same architecture"""
    print("🔄 Recreating CNN model...")
    
    # Build the same CNN architecture as in the notebook
    cnn = tf.keras.models.Sequential()
    
    # Convolutional layers
    cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
    cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    
    cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    
    cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    
    cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    
    cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    
    cnn.add(tf.keras.layers.Dropout(0.25))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.4))
    
    # Output layer
    cnn.add(tf.keras.layers.Dense(units=38,activation='softmax'))
    
    # Compile the model
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    print("✅ Model architecture recreated")
    print(f"Model summary:")
    cnn.summary()
    
    return cnn

def save_fresh_model():
    """Create and save a fresh model"""
    try:
        # Create the model
        model = recreate_model()
        
        # Try to load existing weights if available
        if os.path.exists("trained_plant_disease_model.h5"):
            try:
                print("🔄 Attempting to load existing weights...")
                old_model = tf.keras.models.load_model("trained_plant_disease_model.h5")
                model.set_weights(old_model.get_weights())
                print("✅ Successfully transferred weights from existing model")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}")
                print("ℹ️ Creating model with random weights (will need retraining)")
        
        # Save in both formats with error handling
        try:
            # Save as H5 (more stable)
            model.save("trained_plant_disease_model_fresh.h5")
            print("✅ Saved fresh model as H5 format")
        except Exception as e:
            print(f"❌ Error saving H5: {e}")
        
        try:
            # Save as native Keras format
            model.save("trained_plant_disease_model_fresh.keras")
            print("✅ Saved fresh model as Keras format")
        except Exception as e:
            print(f"❌ Error saving Keras: {e}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating fresh model: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting model recreation process...")
    model = save_fresh_model()
    
    if model:
        print("\n🎉 Model recreation completed!")
        print("📝 Next steps:")
        print("1. Rename the fresh model files to replace the corrupted ones")
        print("2. Or update main.py to use the fresh model files")
        print("3. If weights weren't loaded, you'll need to retrain the model")
    else:
        print("\n❌ Model recreation failed!")
