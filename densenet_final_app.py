import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Set page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding: 3rem 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #eff1f3;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .report-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .grad-cam-container {
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Class names
class_names = ['Dermatofibroma', 'Melanocyticnevus', 'Melanoma']

# Function to load the model
@st.cache_resource
def load_classification_model():
    try:
        model = load_model("my_model.keras")
        return model
    except:
        st.error("⚠️ Model file not found. Please ensure 'my_model.keras' is in the same directory as this script.")
        return None

# Function to preprocess the image
def preprocess_image(img):
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Expand dimensions to create batch
    img = np.expand_dims(img, axis=0)
    
    return img

# Function to make prediction
def predict_skin_lesion(model, img):
    # Preprocess image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    # Get the predicted class index
    predicted_class_idx = np.argmax(predictions[0])
    
    # Get class name and confidence
    class_name = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get all class probabilities
    class_probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    return class_name, confidence, class_probabilities

# Generate GradCAM visualization
def generate_gradcam(model, img, layer_name=None):
    # If layer_name is not provided, try to find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
        if layer_name is None:
            st.warning("Could not automatically determine a convolutional layer for GradCAM. Using the last layer instead.")
            layer_name = model.layers[-1].name
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Get the gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the model
        conv_outputs, predictions = grad_model(processed_img)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]
    
    # Calculate gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels of the feature map with the gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize the heatmap
    heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap))
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB for visualization
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    
    # Convert original image to RGB
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img.copy()
    
    # Resize original image if needed
    img_rgb = cv2.resize(img_rgb, (heatmap.shape[1], heatmap.shape[0]))
    
    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    
    return superimposed_img

# Informational content for each class
class_info = {
    "Dermatofibroma": {
        "description": "Dermatofibromas are common, benign skin growths that often appear as small, firm bumps. They typically have a red-brown color and are most commonly found on the legs.",
        "characteristics": ["Firm to the touch", "Usually less than 1 cm in diameter", "May itch or be tender", "Often appears after minor trauma"],
        "risk_level": "Low - Generally benign",
        "next_steps": "Usually no treatment is necessary unless the lesion is symptomatic or cosmetically concerning."
    },
    "Melanocyticnevus": {
        "description": "Melanocytic nevi, commonly known as moles, are benign accumulations of melanocytes. They can be present at birth (congenital) or appear during childhood or adolescence (acquired).",
        "characteristics": ["Usually round or oval with distinct borders", "Uniform in color", "Typically less than 6 mm in diameter", "May be flat or raised"],
        "risk_level": "Low to moderate - Monitor for changes",
        "next_steps": "Regular self-examination and dermatological check-ups. Report any changes in size, shape, color, or if bleeding occurs."
    },
    "Melanoma": {
        "description": "Melanoma is a serious form of skin cancer that develops from melanocytes. It is the most dangerous form of skin cancer and can spread to other parts of the body if not detected early.",
        "characteristics": ["Asymmetrical shape", "Irregular borders", "Variations in color", "Diameter often greater than 6 mm", "Evolving size, shape, or color"],
        "risk_level": "High - Potentially life-threatening",
        "next_steps": "Immediate consultation with a dermatologist or oncologist is strongly recommended for proper diagnosis and treatment planning."
    }
}

# Create tabs for different pages
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
    st.sidebar.title("Skin Lesion Classifier")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Analyze", "About", "How It Works", "Help"])
    
    # Load model
    model = load_classification_model()
    
    if page == "Analyze":
        st.title("🔬 Skin Lesion Classification")
        st.markdown("Upload an image of a skin lesion to classify it as Dermatofibroma, Melanocytic nevus, or Melanoma.")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Image")
            upload_option = st.radio("Choose input method:", ["Upload an image", "Use camera"])
            
            if upload_option == "Upload an image":
                uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    # Read and display the image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    # Convert BGR to RGB (OpenCV loads images in BGR)
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                    st.image(opencv_image, caption="Uploaded Image", use_column_width=True)
            else:
                # Camera input option
                camera_image = st.camera_input("Take a picture")
                if camera_image is not None:
                    # Read the image
                    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    # Convert BGR to RGB
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            
            # Only show the analyze button if an image is provided
            if (upload_option == "Upload an image" and uploaded_file is not None) or \
               (upload_option == "Use camera" and camera_image is not None):
                
                if st.button("Analyze Skin Lesion", key="analyze_button"):
                    if model is not None:
                        with st.spinner("Analyzing..."):
                            # Make prediction
                            class_name, confidence, class_probabilities = predict_skin_lesion(model, opencv_image)
                            
                            # Store results in session state for display in the second column
                            st.session_state.analysis_complete = True
                            st.session_state.class_name = class_name
                            st.session_state.confidence = confidence
                            st.session_state.class_probabilities = class_probabilities
                            st.session_state.opencv_image = opencv_image
                            
                            # Try to generate GradCAM
                            try:
                                st.session_state.gradcam_img = generate_gradcam(model, opencv_image)
                            except Exception as e:
                                st.error(f"Could not generate heatmap: {e}")
                                st.session_state.gradcam_img = None
                    else:
                        st.error("Model could not be loaded. Please check the model file.")
        
        # Results column
        with col2:
            if "analysis_complete" in st.session_state and st.session_state.analysis_complete:
                class_name = st.session_state.class_name
                confidence = st.session_state.confidence
                class_probabilities = st.session_state.class_probabilities
                
                # Display prediction with formatting based on severity
                result_color = "#28a745" if class_name != "Melanoma" else "#dc3545"
                
                st.markdown("### Diagnosis Results")
                st.markdown(f"""
                <div class="report-container">
                    <h3 style="color:{result_color};">Prediction: {class_name}</h3>
                    <h4>Confidence: {confidence*100:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display info about the predicted class
                with st.expander("Detailed Information", expanded=True):
                    st.markdown(f"**Description**: {class_info[class_name]['description']}")
                    st.markdown("**Key Characteristics**:")
                    for char in class_info[class_name]['characteristics']:
                        st.markdown(f"- {char}")
                    st.markdown(f"**Risk Level**: {class_info[class_name]['risk_level']}")
                    st.markdown(f"**Recommended Next Steps**: {class_info[class_name]['next_steps']}")
                
                # Plot probability distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                classes = list(class_probabilities.keys())
                probs = list(class_probabilities.values())
                colors = ['#28a745' if c != 'Melanoma' else '#dc3545' for c in classes]
                ax.bar(classes, probs, color=colors)
                ax.set_ylabel('Probability')
                ax.set_title('Class Probability Distribution')
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display GradCAM visualization if available
                if "gradcam_img" in st.session_state and st.session_state.gradcam_img is not None:
                    st.markdown("### Regions of Interest")
                    st.markdown("This visualization highlights the regions that influenced the model's decision:")
                    st.image(st.session_state.gradcam_img, caption="GradCAM Visualization", use_column_width=True)
            
            else:
                # If no analysis has been done, show information
                st.markdown("### Results will appear here")
                st.markdown("Upload an image and click 'Analyze' to see the classification results.")
                
                # Display example information about the classes
                st.markdown("### What We Detect")
                st.markdown("This tool can identify three types of skin lesions:")
                
                # Create three columns for the class examples
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Dermatofibroma**")
                    st.markdown("Benign skin nodule")
                    st.markdown("*Low risk*")
                
                with col2:
                    st.markdown("**Melanocytic nevus**")
                    st.markdown("Common mole")
                    st.markdown("*Low to moderate risk*")
                
                with col3:
                    st.markdown("**Melanoma**")
                    st.markdown("Serious skin cancer")
                    st.markdown("*High risk*")
    
    elif page == "About":
        st.title("About This Project")
        st.markdown("""
        This application is designed to help identify different types of skin lesions using deep learning technology. It provides a simple interface for uploading images and receiving quick assessments.
        
        ### Purpose
        
        Skin cancer is one of the most common types of cancer, and early detection is crucial for successful treatment. This tool aims to assist in the preliminary assessment of skin lesions, potentially helping to identify concerning cases that should be examined by healthcare professionals.
        
        ### The Model
        
        The classification model was developed using deep learning techniques and trained on a dataset of dermatological images. While it can provide useful insights, it should be used as a supplementary tool only and not as a replacement for professional medical advice.
        
        ### Classes
        
        The model can distinguish between three types of skin lesions:
        
        1. **Dermatofibroma** - A common benign skin nodule that usually appears on the legs
        2. **Melanocytic nevus** - Commonly known as a mole, these are typically benign growths
        3. **Melanoma** - A serious form of skin cancer that requires prompt medical attention
        
        ### Important Disclaimer
        
        This application is intended for educational and research purposes only. It is not a medical device and has not been approved for clinical use. Always consult with a qualified healthcare professional for proper diagnosis and treatment of skin conditions.
        """)
        
    elif page == "How It Works":
        st.title("How It Works")
        st.markdown("""
        ### The Process
        
        This skin lesion classifier works through a series of steps:
        
        1. **Image Capture/Upload**: You provide an image of a skin lesion either by uploading a photo or taking one with your camera.
        
        2. **Preprocessing**: The image is automatically resized to 224x224 pixels and normalized to prepare it for analysis.
        
        3. **Classification**: The deep learning model processes the image and assigns probabilities to each possible class.
        
        4. **Visualization**: The results are displayed with a probability distribution and a heatmap showing which regions of the image influenced the decision.
        
        ### The Technology
        
        The system uses a deep convolutional neural network trained on thousands of dermatological images. The model has learned to identify visual patterns associated with different types of skin lesions.
        
        ### Interpreting Results
        
        When viewing your results, consider:
        
        - **Confidence Score**: Higher percentages indicate greater confidence in the prediction
        - **Probability Distribution**: Shows how the model allocated probability across all possible classes
        - **Heatmap**: Highlights regions of interest that influenced the model's decision
        
        Remember that even high-confidence predictions should be confirmed by a healthcare professional, especially if the lesion is classified as potentially malignant.
        """)
        
    elif page == "Help":
        st.title("Help & FAQ")
        st.markdown("""
        ### Frequently Asked Questions
        
        **Q: How accurate is this tool?**  
        A: While the model has been trained on a diverse dataset, it should be used as a supportive tool and not for definitive diagnosis. Always consult with a medical professional.
        
        **Q: What types of images work best?**  
        A: Clear, well-lit images of the skin lesion taken from directly above work best. The lesion should be centered in the image with some surrounding normal skin visible.
        
        **Q: Is my data private?**  
        A: Yes, all processing is done locally in your browser. Images are not stored or sent to external servers.
        
        **Q: What should I do if the model identifies a potential melanoma?**  
        A: If the model indicates a high probability of melanoma, consult with a dermatologist immediately. Early detection of melanoma significantly improves treatment outcomes.
        
        ### Tips for Better Results
        
        1. **Good Lighting**: Take photos in bright, natural light if possible
        2. **Clear Focus**: Ensure the image is sharp and not blurry
        3. **Proper Distance**: Capture the lesion from about 10-15 cm away
        4. **Context**: Include some surrounding skin for reference
        5. **Multiple Angles**: If uncertain, take photos from different angles
        
        ### Troubleshooting
        
        If you encounter issues:
        
        - Try refreshing the page
        - Use a different browser
        - Ensure your image is in a supported format (JPG, JPEG, PNG)
        - Check that the file size is not too large
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 Skin Lesion Classifier | Final Year Project")

if __name__ == "__main__":
    main()
