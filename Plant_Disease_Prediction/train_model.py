#!/usr/bin/env python3
"""
Plant Disease Detection Model Training Script
Extracted from Train_plant_disease.ipynb for direct execution
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os

print("Starting Plant Disease Detection Model Training...")
print("=" * 50)

# Check if dataset exists
if not os.path.exists('train') or not os.path.exists('valid'):
    print("❌ Dataset folders 'train' and 'valid' not found!")
    print("Please make sure you have downloaded the dataset from:")
    print("https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    exit(1)

print("✅ Dataset folders found!")

# Data Preprocessing
print("\n📂 Loading training dataset...")
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

print("📂 Loading validation dataset...")
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

print(f"✅ Training set: Found {len(training_set)} batches")
print(f"✅ Validation set: Found {len(validation_set)} batches")

# Building Model
print("\n🏗️  Building CNN Model...")
cnn = tf.keras.models.Sequential()

# Convolutional Layers
print("Adding convolutional layers...")
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

# Dense Layers
print("Adding dense layers...")
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4)) # To avoid overfitting

# Output Layer
cnn.add(tf.keras.layers.Dense(units=38,activation='softmax'))

# Compile Model
print("🔧 Compiling model...")
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
           loss='categorical_crossentropy',
           metrics=['accuracy'])

# Model Summary
print("\n📊 Model Summary:")
cnn.summary()

# Training
print("\n🚀 Starting Training...")
print("This may take a while depending on your hardware...")
training_history = cnn.fit(
    x=training_set,
    validation_data=validation_set,
    epochs=10
)

print("✅ Training completed!")

# Evaluate Model
print("\n📈 Evaluating Model...")
train_loss, train_acc = cnn.evaluate(training_set)
print(f'Training accuracy: {train_acc:.4f}')

val_loss, val_acc = cnn.evaluate(validation_set)
print(f'Validation accuracy: {val_acc:.4f}')

# Save Model
print("\n💾 Saving Model...")
cnn.save('trained_plant_disease_model.keras')
print("✅ Model saved as 'trained_plant_disease_model.keras'")

# Save Training History
print("💾 Saving training history...")
with open('training_hist.json','w') as f:
    json.dump(training_history.history, f)
print("✅ Training history saved as 'training_hist.json'")

# Plot Training Results
print("\n📊 Generating training plots...")
epochs = [i for i in range(1,11)]
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, training_history.history['loss'], color='red', label='Training Loss')
plt.plot(epochs, training_history.history['val_loss'], color='blue', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 Training Complete!")
print("=" * 50)
print("✅ Model saved: trained_plant_disease_model.keras")
print("✅ History saved: training_hist.json") 
print("✅ Plot saved: training_results.png")
print("\nYour model is ready to use with the Streamlit app!")
