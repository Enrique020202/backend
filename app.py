from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Load trained model
MODEL_PATH = "corn_disease_models_updated.keras"
model = load_model(MODEL_PATH)

# Class names must match your model's training
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy', 'Unknown']

# Expected input size for the model
target_size = (224, 224)

# Disease info: Symptoms and Remedies
disease_info = {
    'Blight': {
        'symptoms': [
            "Water-soaked lesions on leaves",
            "Brown or tan streaks",
            "Leaf blighting and wilting"
        ],
        'remedies': [
            "Use blight-resistant seeds",
            "Apply appropriate fungicide",
            "Remove and destroy infected plants"
        ]
    },
    'Common Rust': {
        'symptoms': [
            "Reddish-brown pustules on leaves",
            "Yellow spots that turn brown",
            "Reduced photosynthesis"
        ],
        'remedies': [
            "Plant rust-resistant hybrids",
            "Use crop rotation",
            "Apply fungicides like Mancozeb"
        ]
    },
    'Gray Leaf Spot': {
        'symptoms': [
            "Rectangular gray to tan lesions",
            "Leaf tissue death",
            "Early leaf drop"
        ],
        'remedies': [
            "Improve air circulation between plants",
            "Use resistant corn varieties",
            "Apply fungicides at early stage"
        ]
    },
    'Healthy': {
        'symptoms': ["No visible disease symptoms"],
        'remedies': ["Maintain good agricultural practices"]
    },
    'Unknown': {
        'symptoms': ["Unrecognized pattern in image"],
        'remedies': ["Retake image or consult an expert"]
    }
}

# Preprocess image for prediction
def preprocess_image(image):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Flask route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Read image directly from memory (not saved to disk)
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # Preprocess + Predict
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    # Lookup disease info
    info = disease_info.get(predicted_class, {
        "symptoms": ["No data available"],
        "remedies": ["Consult an agronomist"]
    })

    # Build JSON response
    result = {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "symptoms": info["symptoms"],
        "remedies": info["remedies"]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
