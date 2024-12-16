import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained model
@st.cache_resource
def load_skin_cancer_model():
    return load_model('my_model.keras')

model = load_skin_cancer_model()

# Define class names
class_names = ['Dermatofibroma', 'Melanocyticnevus', 'Melanoma']

# Function to classify an image
def classify_image(uploaded_image):
    # Load the image
    image = Image.open(uploaded_image)
    image_resized = image.resize((224, 224))  # Resize the image to match model input size

    # Convert image to numpy array and preprocess
    img = np.array(image_resized)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    # Predict the label
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return predicted_class, predictions

# Streamlit UI
st.title("Skin Cancer Classifier")

st.write("Upload an image of a skin lesion to classify it as one of the following:")
st.write(", ".join(class_names))

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Classify the image
    with st.spinner("Classifying the image..."):
        predicted_class, predictions = classify_image(uploaded_file)

    # Display the results
    st.write("### Prediction")
    st.write(f"The model predicts: **{predicted_class}**")

    # Show prediction probabilities
    st.write("### Prediction Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
