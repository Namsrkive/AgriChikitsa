import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json

# Load the model
def load_model():
    model = torch.load("crop_disease_detection.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def predict(image, model):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_labels[str(predicted.item())]

# Streamlit UI
st.set_page_config(page_title="Crop Disease Detection", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #F0F2F6;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .sidebar .sidebar-content {
            background-color: #E8F5E9;
        }
        .result-box {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Crop Disease Detection 🌿</div>", unsafe_allow_html=True)

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    model = load_model()
    prediction = predict(image, model)
    
    st.markdown(f"<div class='result-box'>Prediction: {prediction}</div>", unsafe_allow_html=True)
