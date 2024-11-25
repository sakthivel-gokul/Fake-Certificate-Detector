import os
import cv2
import pytesseract
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from groq import Groq
import streamlit as st

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the Groq client
client = Groq(api_key='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')  # Replace with your actual Groq API key

# Load the pre-trained model for feature extraction
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Function to load and preprocess image for MobileNetV2 model
def load_and_preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Image not found or unable to open")
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    return image_preprocessed

# Function to extract text from an image using Tesseract OCR
def extract_text(img_path):
    image = Image.open(img_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to classify certificate authenticity using MobileNetV2 model
def classify_certificate(image_path):
    image = load_and_preprocess_image(image_path)
    features = model.predict(image)
    threshold = np.mean(features) * 0.8
    classification = "Original" if np.mean(features) > threshold else "Duplicate"
    return classification

# Function to perform certificate check
def check_certificate(image_path):
    text_data = extract_text(image_path)
    authenticity = classify_certificate(image_path)
    
    flag = 1 if authenticity == "Original" else 0

    if "Sample" in text_data or authenticity == "Duplicate":
        return "Duplicate certificate detected!", text_data, 0
    elif "certified" in text_data or "Certificate" in text_data:
        return "Certificate is likely original.", text_data, 1
    else:
        return "Certificate authenticity could not be confirmed.", text_data, flag

# Function to interact with Groq AI based on extracted certificate details
def chat_with_groq(text_data):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the AI Certificate Detector. "
                    "Analyze the text data extracted from the certificate."
                    "Provide insights on the certificate authenticity and any other relevant details."
                    "Determine if the certificate is original or duplicate."
                    "Respond only with 'Original' or 'Duplicate' in the final output."
                )
            },
            {
                "role": "user",
                "content": text_data
            }
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    
    return response.strip()

# Main function to handle certificate verification process
def verify_certificate(image_path):
    authenticity_result, extracted_text, flag = check_certificate(image_path)
    groq_response = chat_with_groq(extracted_text)
    
    if "original" in groq_response.lower() and not any(
        term in groq_response.lower() for term in ["fake", "modified", "concerns about authenticity", "not original", "further verification", "duplicate"]
    ):
        flag = 1
    else:
        flag = 0

    return "Original" if flag == 1 else "Duplicate"

# Streamlit app code with custom CSS styling for output display
st.title("Certificate Authenticity Checker")

uploaded_file = st.file_uploader("Upload a certificate image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily to verify authenticity
    with open("temp_certificate_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Perform certificate verification
    result = verify_certificate("temp_certificate_image.jpg")
    
    # Display the final result with CSS styling
    if result == "Original":
        st.markdown(
            f'<div style="padding: 20px; border-radius: 10px; background-color: #d1f7d1; border: 2px solid #4CAF50; text-align: center;">'
            f'<h2 style="color:green;">Original</h2>'
            f'<p style="color:green;">The certificate is authentic.</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="padding: 20px; border-radius: 10px; background-color: #f7d1d1; border: 2px solid #FF5733; text-align: center;">'
            f'<h2 style="color:red;">Duplicate</h2>'
            f'<p style="color:red;">The certificate is a duplicate.</p>'
            f'</div>',
            unsafe_allow_html=True
        )
