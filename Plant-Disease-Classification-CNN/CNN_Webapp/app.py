import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)

st.markdown("""
    <style>
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .healthy { background-color: #e6f7ea; border: 1px solid #34a853; }
    .diseased { background-color: #fdecea; border: 1px solid #ea4335; }
    </style>
""", unsafe_allow_html=True)


from huggingface_hub import hf_hub_download

@st.cache_resource
def get_model():
    model_path = hf_hub_download(
        repo_id="Mann4/plant-disease-cnn",
        filename="PlantDisease.keras"
    )
    return load_model(model_path)

model = get_model()

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]


with st.sidebar:
    st.header("🌿 About")
    st.write(
        "This app uses a Convolutional Neural Network (CNN) trained on the "
        "PlantVillage dataset to identify **38 plant disease / healthy classes** "
        "across 14 crop species from a single leaf photo."
    )
    st.write("**Tech stack:** TensorFlow/Keras, Streamlit")
    st.write("**Model input:** 224×224 RGB image")
    st.caption("For educational purposes — not a substitute for professional agronomic diagnosis.")


st.title("🌿 Plant Disease Classifier")
st.write("Upload a photo of a plant leaf to detect its species and health status.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing leaf..."):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0]

    top3_idx = np.argsort(prediction)[-3:][::-1]
    top_class = class_names[top3_idx[0]]
    top_confidence = prediction[top3_idx[0]] * 100

    crop, condition = top_class.split("___") if "___" in top_class else (top_class, "")
    is_healthy = "healthy" in condition.lower()

    with col2:
        card_class = "healthy" if is_healthy else "diseased"
        status_label = "✅ Healthy" if is_healthy else "⚠️ Disease Detected"
        condition_display = condition.replace("_", " ") if condition else "Healthy"

        st.markdown(f"""
            <div class="result-card {card_class}">
                <h4>{status_label}</h4>
                <p><b>Crop:</b> {crop.replace('_', ' ')}</p>
                <p><b>Condition:</b> {condition_display}</p>
                <p><b>Confidence:</b> {top_confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Top 3 Predictions")
    for idx in top3_idx:
        label = class_names[idx].replace("___", " — ").replace("_", " ")
        conf = prediction[idx] * 100
        st.write(f"{label}")
        st.progress(min(int(conf), 100), text=f"{conf:.1f}%")

else:
    st.info("👆 Upload an image to get started, or try one of the sample images in the Screenshots folder.")