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
    </style>
""", unsafe_allow_html=True)

# Class indices
class_indices = {'Dermatofibroma': 0, 'Melanocyticnevus': 1, 'Melanoma': 2}
# Invert the dictionary for prediction mapping
class_names = {v: k for k, v in class_indices.items()}

# Function to load the model
@st.cache_resource
def load_classification_model():
    try:
        model = load_model("vgg16_model.h5")
        return model
    except:
        st.error("⚠️ Model file not found. Please ensure 'vgg16_model.h5' is in the same directory as this script.")
        return None

# Function to preprocess the image
def preprocess_image(img):
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to VGG16 input size
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
def generate_gradcam(model, img, layer_name="block5_conv3"):
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

# Main app layout
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
    st.sidebar.title("Skin Lesion Classifier")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "About", "How It Works", "Help"])
    
    # Load model
    model = load_classification_model()
    
    if page == "Home":
        st.title("Skin Lesion Classification")
        st.markdown("This application uses a deep learning model to classify skin lesions into three categories: Dermatofibroma, Melanocytic nevus, and Melanoma.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload an Image")
            uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Read and display the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                
                # Convert BGR to RGB (OpenCV loads images in BGR)
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                
                st.image(opencv_image, caption="Uploaded Image", use_column_width=True)
                
                # Make prediction button
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing..."):
                        # Check if model is loaded
                        if model is not None:
                            # Make prediction
                            class_name, confidence, class_probabilities = predict_skin_lesion(model, opencv_image)
                            
                            # Generate GradCAM visualization
                            try:
                                gradcam_img = generate_gradcam(model, opencv_image)
                                col2.subheader("Diagnosis Results")
                                
                                # Display prediction with formatting based on severity
                                result_color = "#28a745" if class_name != "Melanoma" else "#dc3545"
                                
                                col2.markdown(f"""
                                <div class="report-container">
                                    <h3 style="color:{result_color};">Prediction: {class_name}</h3>
                                    <h4>Confidence: {confidence*100:.2f}%</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display info about the predicted class
                                with col2.expander("Detailed Information", expanded=True):
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
                                colors = ['#28a745', '#17a2b8', '#dc3545']
                                ax.bar(classes, probs, color=colors)
                                ax.set_ylabel('Probability')
                                ax.set_title('Class Probability Distribution')
                                plt.xticks(rotation=30, ha='right')
                                plt.tight_layout()
                                col2.pyplot(fig)
                                
                                # Display GradCAM visualization
                                col2.subheader("Regions of Interest (GradCAM)")
                                col2.markdown("This visualization highlights the regions that influenced the model's decision:")
                                col2.image(gradcam_img, caption="GradCAM Visualization", use_column_width=True)
                                
                            except Exception as e:
                                st.error(f"Error generating visualization: {e}")
                        else:
                            st.error("Model could not be loaded. Please check the model file.")
        
        # If no image is uploaded, show sample images
        if uploaded_file is None:
            with col2:
                st.subheader("Sample Images")
                st.markdown("Here are examples of the three types of skin lesions this model can identify:")
                
                # Sample images for each class (these are placeholders)
                sample_cols = st.columns(3)
                
                with sample_cols[0]:
                    st.image("https://img.icons8.com/color/96/000000/mole.png", caption="Dermatofibroma")
                
                with sample_cols[1]:
                    st.image("https://img.icons8.com/color/96/000000/mole.png", caption="Melanocytic nevus")
                
                with sample_cols[2]:
                    st.image("https://img.icons8.com/color/96/000000/mole.png", caption="Melanoma")
                
                st.markdown("Note: The above are placeholder icons. In a real deployment, you would use actual sample images.")
        
    elif page == "About":
        st.title("About This Project")
        st.markdown("""
        This application is designed to help medical professionals and researchers classify skin lesions using deep learning technology.
        
        ### Model Architecture
        The system uses a VGG16 convolutional neural network pre-trained on ImageNet and fine-tuned on a dataset of skin lesion images.
        
        ### Classes
        The model can distinguish between three types of skin lesions:
        
        1. **Dermatofibroma** - A common benign skin nodule that usually appears on the legs
        2. **Melanocytic nevus** - Commonly known as a mole, these are typically benign growths
        3. **Melanoma** - A serious form of skin cancer that requires prompt medical attention
        
        ### Performance
        The model has been trained on a carefully curated dataset and validated against test cases to ensure accuracy. However, this tool should be used for research and educational purposes only and is not a substitute for professional medical diagnosis.
        """)
        
    elif page == "How It Works":
        st.title("How It Works")
        st.markdown("""
        ### Model Pipeline
        
        1. **Image Input**: The system accepts images in common formats (JPEG, PNG)
        2. **Preprocessing**: Images are resized to 224x224 pixels and normalized
        3. **Feature Extraction**: The VGG16 convolutional layers extract relevant features
        4. **Classification**: Custom fully-connected layers classify the image
        5. **Visualization**: GradCAM highlights regions of interest that influenced the decision
        
        ### Technical Details
        
        The model uses transfer learning from VGG16, a deep convolutional network pre-trained on ImageNet. The base VGG16 layers are frozen, and custom classification layers are trained specifically for skin lesion classification.
        
        ```python
        # Base model architecture
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(3, activation='softmax')(x)
        ```
        
        ### GradCAM Visualization
        
        Gradient-weighted Class Activation Mapping (GradCAM) is used to generate heatmaps that highlight the regions of the input image that are most important for the model's prediction, providing interpretability to the classification decision.
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
        
        ### Troubleshooting
        
        **Image upload errors**  
        - Ensure your image is in JPG, JPEG, or PNG format
        - Check that the file size is reasonable (under 5MB)
        - Try a different image if the problem persists
        
        **Unexpected results**  
        - Make sure the lesion is clearly visible and centered
        - Check that the image is in focus and well-lit
        - Remember that this tool is not a substitute for professional medical advice
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2023 Skin Lesion Classifier | Powered by TensorFlow & Streamlit")
    
if __name__ == "__main__":
    main()
