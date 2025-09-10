# Create a dummy model for testing purposes
# This is just for testing the app functionality - not for actual predictions

import tensorflow as tf
import numpy as np

def create_dummy_model():
    """
    Create a simple dummy model that matches the expected structure
    This is just for testing the app - predictions won't be accurate
    """
    
    print("Creating dummy model for testing...")
    
    # Create a simple CNN model with the same structure expected by your app
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(38, activation='softmax')  # 38 classes as per your app
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy training data to initialize weights properly
    dummy_x = np.random.random((10, 128, 128, 3))
    dummy_y = tf.keras.utils.to_categorical(np.random.randint(0, 38, 10), 38)
    
    # Train for just 1 epoch to initialize weights
    print("Initializing model weights...")
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    
    # Save the model in both formats
    try:
        model.save('trained_plant_disease_model.h5')
        print("✅ Dummy model saved as trained_plant_disease_model.h5")
    except Exception as e:
        print(f"❌ Error saving H5 format: {e}")
    
    try:
        model.save('trained_plant_disease_model.keras')
        print("✅ Dummy model saved as trained_plant_disease_model.keras")
    except Exception as e:
        print(f"❌ Error saving Keras format: {e}")
    
    print("\n⚠️  IMPORTANT: This is a dummy model for testing only!")
    print("For real plant disease detection, you need to:")
    print("1. Download the actual dataset from Kaggle")
    print("2. Train the model properly using Train_plant_disease.ipynb")
    print("3. Replace this dummy model with the properly trained one")

if __name__ == "__main__":
    create_dummy_model()
