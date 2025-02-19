import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np

# Page setup
st.set_page_config(page_title="Skin Lesion Analyzer", page_icon="🔍")

# Simple, clean styling with more prominent disclaimer
st.markdown("""
<style>
    .header {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .benign {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .malignant {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .disclaimer {
        background-color: #fff4e5;
        border: 1px solid #ffcc80;
        color: #e65100;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model efficiently
@st.cache_resource
def load_classifier_model():
    try:
        return load_model('my_model.keras')
    except Exception:
        return None

# Clear header
st.markdown("<h1 class='header'>Skin Lesion Analyzer</h1>", unsafe_allow_html=True)

# PROMINENT DISCLAIMER - Now placed at the top for visibility
st.markdown("""
<div class="disclaimer">
    ⚠️ IMPORTANT: This tool is for educational purposes only and not a substitute for 
    professional medical diagnosis. Always consult a healthcare provider for medical concerns.
</div>
""", unsafe_allow_html=True)

# Simplified class information
conditions = {
    'Dermatofibroma': {
        'type': 'Benign',
        'description': 'A common benign skin growth appearing as a small, firm bump.',
        'action': 'Generally harmless. No immediate action required.'
    },
    'Melanocytic nevus': {
        'type': 'Benign',
        'description': 'Common mole - a benign growth of pigment cells.',
        'action': 'Monitor periodically for changes in appearance.'
    },
    'Melanoma': {
        'type': 'Malignant',
        'description': 'A serious form of skin cancer that develops from pigment cells.',
        'action': 'Seek immediate medical attention from a dermatologist.'
    }
}

# Simple image classification function
def analyze_skin_image(image_file):
    model = load_classifier_model()
    if not model:
        st.error("Could not load the analysis model. Please check if 'my_model.keras' exists.")
        return None, None
    
    # Process image
    image = Image.open(image_file)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    class_index = np.argmax(predictions[0])
    class_name = list(conditions.keys())[class_index]
    
    return class_name, predictions[0]

# Main content - two column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    st.write("Upload a clear image of the skin lesion for analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    # Help information
    with st.expander("How to take better skin lesion photos"):
        st.write("""
        - Use good lighting (natural daylight is best)
        - Keep the camera steady and close to the lesion
        - Include a ruler for size reference if possible
        - Take photos from multiple angles
        """)

with col2:
    if uploaded_file:
        # Display uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width=250)
        
        # Analyze image
        with st.spinner("Analyzing image..."):
            result, probabilities = analyze_skin_image(uploaded_file)
        
        # Display results
        if result:
            condition = conditions[result]
            result_type = "benign" if condition['type'] == "Benign" else "malignant"
            
            st.markdown(f"""
            <div class="result-box {result_type}">
                <h3>Analysis Result: {result}</h3>
                <p><b>Classification:</b> {condition['type']}</p>
                <p><b>Description:</b> {condition['description']}</p>
                <p><b>Recommended Action:</b> {condition['action']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reminder about consulting healthcare provider
            if condition['type'] == "Malignant":
                st.warning("⚠️ This result suggests a potentially serious condition. Please consult a healthcare professional immediately.")
            
            # Display confidence levels simply
            st.subheader("Confidence Levels")
            for i, class_name in enumerate(conditions.keys()):
                confidence = probabilities[i] * 100
                st.progress(confidence/100)
                st.write(f"{class_name}: {confidence:.1f}%")
    else:
        # Educational information when no image is uploaded
        st.subheader("About Skin Lesion Analysis")
        st.write("""
        This tool examines skin lesion images and classifies them into three categories:
        
        1. **Dermatofibroma** - A benign skin growth
        2. **Melanocytic nevus** - Common mole (benign)
        3. **Melanoma** - A serious form of skin cancer
        
        Early detection of skin cancer can significantly improve treatment outcomes.
        """)
        
        # ABCDE rule in simple terms
        st.markdown("""
        <div class="info-box">
            <h4>ABCDE Rule for Skin Checks</h4>
            <ul>
                <li><b>A</b>symmetry - One half looks different from the other</li>
                <li><b>B</b>order - Irregular or poorly defined edges</li>
                <li><b>C</b>olor - Varies from one area to another</li>
                <li><b>D</b>iameter - Larger than 6mm (pencil eraser)</li>
                <li><b>E</b>volving - Changes in size, shape, or color</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Additional disclaimer at the bottom for reinforcement
st.markdown("""
<div class="disclaimer" style="margin-top: 30px;">
    ⚠️ REMEMBER: This is not a diagnostic tool. Consult a dermatologist for proper evaluation of any skin concerns.
</div>
""", unsafe_allow_html=True)