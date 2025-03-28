# # !pip install streamlit
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image


# model = tf.keras.models.load_model("skin_disease_model_ISIC_densenet.h5")


# CLASS_NAMES = ["Actinic keratosis", "Atopic Dermatitis", "Benign keratosis", 
#                "Dermatofibroma", "Melanocytic nevus", "Melanoma", 
#                "Squamous cell carcinoma", "Tinea Ringworm Candidiasis", "Vascular lesion"]


# def preprocess_image(image):
#     image = np.array(image)
#     image = cv2.resize(image, (240, 240))  
#     image = image / 255.0  
#     image = np.expand_dims(image, axis=0)  
#     return image


# st.title("🩺 Skin Disease Classifier")
# st.write("Upload an image of a skin lesion, and the model will predict the disease.")


# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
    
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

    
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     predicted_class = np.argmax(prediction)
#     confidence = np.max(prediction) * 100  

    
#     st.subheader("🔍 Prediction")
#     st.write(f"**Class:** {CLASS_NAMES[predicted_class]}")
#     st.write(f"**Confidence:** {confidence:.2f}%")
import flask
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("skin_disease_model_ISIC_densenet.h5")

# Define disease classes
CLASS_NAMES = [
    "Actinic keratosis", "Atopic Dermatitis", "Benign keratosis", 
    "Dermatofibroma", "Melanocytic nevus", "Melanoma", 
    "Squamous cell carcinoma", "Tinea Ringworm Candidiasis", "Vascular lesion"
]

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (240, 240))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# API endpoint to handle image prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100  

    return jsonify({
        "disease": CLASS_NAMES[predicted_class],
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
