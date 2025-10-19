# app.py
import streamlit as st
from PIL import Image
from pipeline import PlantPredictor

# Initialize predictor
@st.cache_resource
def load_predictor():
    return PlantPredictor()

predictor = load_predictor()

# Streamlit UI
st.title("🌿 Plant Species Classification")
st.markdown("Upload a plant image and get the predicted species below:")

uploaded_file = st.file_uploader("📸 Choose a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            predicted_class, confidence = predictor.predict(image)
        st.success(f"**Predicted Species:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
