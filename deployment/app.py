import streamlit as st
import torch
from PIL import Image
from model import load_model, predict

# Define class labels (replace with your actual labels)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Load trained model
import os
from model import load_model  # Ensure correct import

# Get absolute path of the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_disease_detection.pth")

# Check if file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load model
model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))


# Streamlit UI
st.set_page_config(page_title="Crop Disease Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🌱 Crop Disease Detection 🌾</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Upload an image of a plant leaf to detect the disease.</h3>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction, confidence = predict(image, model, CLASS_NAMES)

    # Display results
    st.markdown(
        f"<h2 style='text-align: center; color: #E65100;'>Prediction: {prediction}</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h3 style='text-align: center;'>Confidence: {confidence:.2%}</h3>",
        unsafe_allow_html=True
    )

    # Apply some styling
    st.markdown(
        """
        <style>
            div.stButton > button:first-child {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 10px;
                width: 100%;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
