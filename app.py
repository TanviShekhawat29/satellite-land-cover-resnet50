import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import io

MODEL_PATH = 'satellite_resnet_FINETUNED_90acc.h5'
IMG_SIZE = 64
NUM_CLASSES = 10

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

st.set_page_config(
    page_title="ResNet50 Satellite Classifier", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

def build_resnet50_model(input_shape, num_classes):
    # Load ResNet50 base, excluding top layers
    conv_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    conv_base.trainable = False
  
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') 
    ])
    return model

@st.cache_resource
def load_model_and_classes():
    """Builds the ResNet architecture and loads the weights manually."""
    try:
        tf.get_logger().setLevel('ERROR') 
        
        input_shape = (IMG_SIZE, IMG_SIZE, 3) 
        model = build_resnet50_model(input_shape, NUM_CLASSES)

        model.load_weights(MODEL_PATH)
            
        return model, CLASSES
    except Exception as e:
        st.error(f"Failed to load AI Model: {e}")
        st.warning(f"Check logs. Ensure '{MODEL_PATH}' was pushed correctly via Git LFS.")
        return None, None

model, CLASSES = load_model_and_classes()

def predict_image(image, model):
    
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32') / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    
    top_3_indices = np.argsort(predictions)[::-1][:3]
    results = []
    
    for i in top_3_indices:
        label = CLASSES[i]
        confidence = predictions[i] * 100
        results.append((label, confidence))
        
    return results

def main():
    st.title("🛰️ Satellite Image Classifier ($\mathbf{90\%+}$ Accuracy)")
    st.subheader("B.Tech AI/DL: ResNet50 Transfer Learning")
    st.write("---")

    if model is None:
        st.stop()
    
    st.info("This industry-grade model classifies images into 10 land-use categories.")
    
    uploaded_file = st.file_uploader(
        "Upload an image file (e.g., a satellite image patch)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Uploaded Image**")
                st.image(image, use_column_width=True)
            
            if st.button('Classify Land Use', type="primary", use_container_width=True):
                with st.spinner('Running ResNet50 Fine-Tuned Model...'):
                    results = predict_image(image, model)
                
                with col2:
                    st.markdown("**Top Predictions**")
                    
                    # Display results
                    for rank, (label, confidence) in enumerate(results):
                        st.markdown(f"**#{rank + 1}** {label}: **{confidence:.2f}%**")
                    
                    st.success("Analysis Complete!")
                    if results[0][1] > 90:
                        st.balloons()
                        
        except Exception as e:
            st.exception(e)
            st.error("An unexpected error occurred during processing.")

if __name__ == "__main__":
    main()
