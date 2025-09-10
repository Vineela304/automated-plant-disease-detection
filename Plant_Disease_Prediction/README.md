# Plant Disease Detection Project

## Setup Instructions for Collaborators

### 1. Clone the Repository
```bash
git clone https://github.com/Vineela304/automated-plant-disease-detection.git
cd automated-plant-disease-detection
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Download the Dataset
**⚠️ IMPORTANT: The dataset is NOT included in this repository due to size constraints.**

Download the dataset from Kaggle:
- **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Size**: ~2GB
- **Classes**: 38 plant disease categories

#### Option A: Kaggle CLI (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle credentials (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip
```

#### Option B: Manual Download
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Click "Download" (requires Kaggle account)
3. Extract the zip file
4. Organize folders as shown below

### 4. Dataset Structure
After downloading, organize your project folder like this:
```
Plant_Disease_Prediction/
├── train/              # Training images (70,295 images)
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── ... (38 total classes)
├── valid/              # Validation images (17,572 images)
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── ... (38 total classes)
├── test/               # Test images (33 images)
├── main.py             # Streamlit app
├── Train_plant_disease.ipynb  # Training notebook
└── requirement.txt     # Dependencies
```

### 5. Pre-trained Model
The trained model files are also excluded due to size. You can either:

#### Option A: Use Provided Model (if available)
- Download from [Google Drive/OneDrive link] (add your link here)

#### Option B: Train Your Own Model
```bash
# Run the training notebook
jupyter notebook Train_plant_disease.ipynb
# Or train via Python script
python train_model.py
```

### 6. Run the Application
```bash
streamlit run main.py
```

## Model Information
- **Architecture**: CNN with 5 convolutional blocks
- **Input Size**: 128×128×3 (RGB images)
- **Output**: 38 classes (plant diseases)
- **Training**: 10 epochs with data augmentation
- **Accuracy**: ~95% validation accuracy

## Plant Disease Classes
The model can detect 38 different plant diseases including:
- Apple: Apple scab, Black rot, Cedar apple rust, Healthy
- Corn: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- Grape: Black rot, Esca, Leaf blight, Healthy
- Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, etc.
- And many more...

## File Structure
```
├── main.py                 # Streamlit web application
├── Train_plant_disease.ipynb  # Model training notebook
├── Test_plant_disease.ipynb   # Model testing notebook
├── requirement.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Notes for Collaborators
- The dataset (~87K images) is intentionally excluded from git
- Model files (*.h5, *.keras) are excluded due to size
- Always download fresh dataset for best results
- Training takes 2-4 hours depending on hardware
