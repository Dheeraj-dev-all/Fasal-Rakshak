import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64

app = Flask(__name__)

# ── Load advisory database ──────────────────────────────────────────────────
with open('advisory_db.json', 'r', encoding='utf-8') as f:
    ADVISORY_DB = json.load(f)

# ── Class names (PlantVillage dataset order) ────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ── Load model (lazy, on first request) ────────────────────────────────────
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        try:
            import tensorflow as tf
            MODEL = tf.keras.models.load_model('model/crop_disease_model')
            print("✅ Real model loaded successfully")
        except Exception as e:
            print(f"⚠️  Model not found or TF not installed — using demo mode. Error: {e}")
            MODEL = "demo"
    return MODEL

def preprocess_image(image_bytes):
    """Resize and normalize image for MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0          # MobileNetV2 preprocessing: scale to [-1, 1]
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension → (1, 224, 224, 3)
    return arr

def demo_predict(image_bytes):
    """
    Demo prediction used when the real model isn't loaded.
    Returns a varied result based on image content so the UI looks realistic.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_small = img.resize((8, 8))
    pixels = np.array(img_small).mean(axis=(0, 1))  # average RGB
    r, g, b = pixels
    # Simple heuristic: green-heavy → lean healthy; red/yellow heavy → disease
    if g > r + 20 and g > b + 20:
        idx = CLASS_NAMES.index('Tomato___healthy')
        confidence = 0.88
    elif r > g + 30:
        idx = CLASS_NAMES.index('Tomato___Early_blight')
        confidence = 0.82
    else:
        idx = CLASS_NAMES.index('Tomato___Late_blight')
        confidence = 0.76
    return idx, confidence

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed)}'}), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({'error': 'Empty image file'}), 400

    try:
        model = load_model()

        if model == "demo":
            class_idx, confidence = demo_predict(image_bytes)
        else:
            import tensorflow as tf
            tensor = preprocess_image(image_bytes)
            preds = model.predict(tensor, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))

        predicted_class = CLASS_NAMES[class_idx]
        advisory = ADVISORY_DB.get(predicted_class, {
            "crop": "Unknown", "disease": "Unknown",
            "cause": "Could not identify the disease.",
            "symptoms": "Unclear",
            "organic_remedy": "Please consult a local agricultural expert.",
            "chemical_remedy": "Please consult a local agricultural expert.",
            "urgency": "Medium",
            "hindi": "रोग की पहचान नहीं हो सकी। कृपया स्थानीय कृषि विशेषज्ञ से सलाह लें।"
        })

        lang = request.form.get('lang', 'en')
        low_confidence = confidence < 0.60

        return jsonify({
            'success': True,
            'class': predicted_class,
            'confidence': round(confidence * 100, 1),
            'low_confidence': low_confidence,
            'crop': advisory['crop'],
            'disease': advisory['disease'],
            'cause': advisory['cause'],
            'symptoms': advisory['symptoms'],
            'organic_remedy': advisory['organic_remedy'],
            'chemical_remedy': advisory['chemical_remedy'],
            'urgency': advisory['urgency'],
            'hindi': advisory['hindi'],
            'lang': lang,
            'demo_mode': (model == "demo")
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'classes': len(CLASS_NAMES)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
