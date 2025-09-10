# Model Format Converter
# Converts between .keras and .h5 formats for Windows compatibility

import tensorflow as tf
import os

def convert_model_format():
    """
    Convert model between .keras and .h5 formats
    """
    keras_file = "trained_plant_disease_model.keras"
    h5_file = "trained_plant_disease_model.h5"
    
    if os.path.exists(keras_file) and not os.path.exists(h5_file):
        try:
            print(f"Converting {keras_file} to {h5_file}...")
            model = tf.keras.models.load_model(keras_file)
            model.save(h5_file)
            print(f"✅ Successfully converted to {h5_file}")
        except Exception as e:
            print(f"❌ Error converting model: {e}")
            
    elif os.path.exists(h5_file) and not os.path.exists(keras_file):
        try:
            print(f"Converting {h5_file} to {keras_file}...")
            model = tf.keras.models.load_model(h5_file)
            model.save(keras_file)
            print(f"✅ Successfully converted to {keras_file}")
        except Exception as e:
            print(f"❌ Error converting model: {e}")
            
    elif os.path.exists(keras_file) and os.path.exists(h5_file):
        print("✅ Both model formats already exist!")
        
    else:
        print("❌ No model file found to convert!")
        print("Please train your model first using Train_plant_disease.ipynb")

if __name__ == "__main__":
    convert_model_format()
