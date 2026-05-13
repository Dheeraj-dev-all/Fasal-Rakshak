# Fasal Rakshak — Crop Disease Detector & Farmer Advisory System

B.Tech CSE Minor Project | JECRC University, Jaipur | 2025-26

## What it does
Upload a crop leaf photo → AI detects the disease → Get treatment advisory in English & Hindi.

## How to run locally

### Step 1: Install Python 3.10+
Download from https://python.org

### Step 2: Install dependencies
```
pip install -r requirements.txt
```

### Step 3: Run the app
```
python app.py
```

### Step 4: Open in browser
Go to: http://localhost:5000

## Project Structure
```
crop_disease_app/
├── app.py               # Flask backend
├── advisory_db.json     # Disease advisory database (38 diseases)
├── requirements.txt     # Python dependencies
├── Procfile             # For Render deployment
├── templates/
│   └── index.html       # Frontend UI
└── model/
    └── crop_disease_model/   # Trained TensorFlow model (add after training)
```

## Add the trained model
1. Train MobileNetV2 on PlantVillage dataset (see training notebook)
2. Save model: `model.save('model/crop_disease_model')`
3. Place the saved model folder here: `model/crop_disease_model/`
4. Restart the app — it will automatically load the real model

## Tech Stack
- Python, Flask, TensorFlow/Keras
- MobileNetV2 (Transfer Learning)
- PlantVillage Dataset (38 disease classes)
- HTML5, CSS3, JavaScript (no frameworks)

## Dataset
PlantVillage — available free on Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease
