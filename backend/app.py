import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load Pretrained MobileNetV2 Model
base_model = MobileNetV2(weights="imagenet")
print("✅ Pretrained Model Loaded Successfully")

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize to MobileNetV2 input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream)
    processed_img = preprocess_image(img)

    preds = base_model.predict(processed_img)
    decoded_preds = decode_predictions(preds, top=1)[0]  # Get top prediction

    species_name = decoded_preds[0][1]  # Class label
    confidence = float(decoded_preds[0][2])  # Confidence score

    return jsonify({"species": species_name, "confidence": confidence})

# **New Function to Predict from Local Image Path**
def predict_from_path(image_path):
    processed_img = preprocess_image(image_path)
    preds = base_model.predict(processed_img)
    decoded_preds = decode_predictions(preds, top=1)[0]

    species_name = decoded_preds[0][1]  # Class label
    confidence = float(decoded_preds[0][2])  # Confidence score

    return {"species": species_name, "confidence": confidence}

if __name__ == "__main__":
    app.run(debug=True)
