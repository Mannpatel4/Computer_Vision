import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="Animal Species Classifier",
    page_icon="🐾",
    layout="centered",
)

st.markdown("""
    <style>
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        background-color: #eef1ff;
        border: 1px solid #6366f1;
    }
    </style>
""", unsafe_allow_html=True)


from huggingface_hub import hf_hub_download

@st.cache_resource
def get_model():
    model_path = hf_hub_download(
        repo_id="Mann4/animal-species-classifier",
        filename="animal_transfer_model.keras"
    )
    return load_model(model_path)

model = get_model()


class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
    'Panda', 'Parrot', 'Tiger', 'Zebra',
]


with st.sidebar:
    st.header("🐾 About")
    st.write(
        "This app uses **transfer learning** (a pretrained CNN backbone, "
        "fine-tuned on 16 animal species) to identify the animal in a photo."
    )
    st.write("**Approach:** Frozen base model → selectively unfroze top layers → fine-tuned at a low learning rate")
    st.write("**Model input:** 224×224 RGB image")
    st.write("**Classes:** " + ", ".join(class_names))


st.title("🐾 Animal Species Classifier")
st.write("Upload a photo of an animal to identify its species.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Identifying animal..."):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0]

    top3_idx = np.argsort(prediction)[-3:][::-1]
    top_class = class_names[top3_idx[0]]
    top_confidence = prediction[top3_idx[0]] * 100

    with col2:
        st.markdown(f"""
            <div class="result-card">
                <h4>🎯 {top_class}</h4>
                <p><b>Confidence:</b> {top_confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Top 3 Predictions")
    for idx in top3_idx:
        st.write(class_names[idx])
        conf = prediction[idx] * 100
        st.progress(min(int(conf), 100), text=f"{conf:.1f}%")

else:
    st.info("👆 Upload an image to get started, or try one of the sample images in the Screenshots folder.")