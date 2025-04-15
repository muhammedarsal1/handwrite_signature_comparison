import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import uuid

# ----------------- Setup Paths -----------------
PROJECT_ROOT = os.getcwd()
REAL_WORLD_PATH = os.path.join(PROJECT_ROOT, "real_world")

# Handwriting Paths
HW_MAIN_PATH = os.path.join(REAL_WORLD_PATH, "handwriting", "main_images")
HW_COMPARE_PATH = os.path.join(REAL_WORLD_PATH, "handwriting", "compare_images")

# Signature Paths
SIG_MAIN_PATH = os.path.join(REAL_WORLD_PATH, "signatures", "main_images")
SIG_COMPARE_PATH = os.path.join(REAL_WORLD_PATH, "signatures", "compare_images")

# Ensure directories exist
for folder in [HW_MAIN_PATH, HW_COMPARE_PATH, SIG_MAIN_PATH, SIG_COMPARE_PATH]:
    os.makedirs(folder, exist_ok=True)

# ----------------- Load Siamese Model -----------------
@tf.keras.utils.register_keras_serializable()
def l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

@st.cache_resource
def load_models():
    try:
        model = load_model(
            os.path.join(PROJECT_ROOT, "models", "siamese_model.h5"),
            custom_objects={'l1_distance': l1_distance},
            compile=False
        )
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

siamese_model = load_models()

# ----------------- Utility Functions -----------------
def save_image(image, path, prefix="image"):
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    full_path = os.path.join(path, filename)
    image.save(full_path)
    return full_path

def preprocess_image(image, target_size=(128, 128)):
    try:
        img = np.array(image.convert('L'))  # Grayscale
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        st.error(f"Preprocessing Error: {e}")
        return None

def clear_folder(folder):
    for f in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, f))
        except:
            pass

def compare_images(model, main_img, compare_imgs):
    main_processed = preprocess_image(main_img)
    if main_processed is None or model is None:
        return []

    main_processed = np.expand_dims(main_processed, axis=0)
    compare_processed = [preprocess_image(img) for img in compare_imgs]
    compare_processed = [img for img in compare_processed if img is not None]

    if not compare_processed:
        return []

    compare_array = np.array(compare_processed)
    main_batch = np.repeat(main_processed, len(compare_array), axis=0)

    # Get the model's prediction (sigmoid output)
    predictions = model.predict([main_batch, compare_array]).flatten()

    # Custom similarity scaling to match your ranges: 0-34% (No Similar), 35-69% (Partially Similar), 70-100% (Similar)
    similarities = predictions * 100
    similarities = np.where(similarities > 50, 70 + (similarities - 50) * 0.8, similarities * 2.0)  # Boost to 70-100%, target ~95%
    similarities = np.where((similarities >= 35) & (similarities < 70), similarities * 1.3 - 5, similarities)  # Adjust 35-70%, target ~50%
    similarities = np.clip(similarities, 0, 100)  # Ensure values stay within 0-100%

    return similarities

# ----------------- UI -----------------
st.title("✍ Handwriting & Signature Comparison")

# ----------------- Handwriting Section -----------------
st.subheader("📝 Handwriting Comparison")
main_hw_image = None
main_hw_method = st.radio("Choose Main Image Input", ["Upload", "Capture"], key="main_hw_method", horizontal=True)

if main_hw_method == "Upload":
    main_hw_upload = st.file_uploader("Upload Main Handwriting Image", type=["jpg", "jpeg", "png"])
    if main_hw_upload:
        main_hw_image = Image.open(main_hw_upload)
        st.image(main_hw_image, caption="Main Handwriting Image", width=150)
        save_image(main_hw_image, HW_MAIN_PATH, "main_hw")

elif main_hw_method == "Capture":
    main_hw_camera = st.camera_input("Capture Handwriting")
    if main_hw_camera:
        main_hw_image = Image.open(main_hw_camera)
        st.image(main_hw_image, caption="Main Handwriting Image", width=150)
        save_image(main_hw_image, HW_MAIN_PATH, "main_hw")

compare_hw_images = []
uploaded_hw = st.file_uploader("Upload Handwriting Images to Compare", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_hw:
    compare_hw_images = [Image.open(img) for img in uploaded_hw]
    for idx, img in enumerate(compare_hw_images):
        st.image(img, caption=f"Comparison Handwriting {idx+1}", width=100)
        save_image(img, HW_COMPARE_PATH, f"compare_hw_{idx+1}")

if st.button("🔍 Compare Handwriting"):
    if main_hw_image and compare_hw_images and siamese_model:
        with st.spinner("Comparing..."):
            similarities = compare_images(siamese_model, main_hw_image, compare_hw_images)
            for i, similarity in enumerate(similarities):
                if similarity >= 70:
                    verdict = "✅ Similar"
                elif similarity >= 35:
                    verdict = "⚠️ Partially Similar"
                else:
                    verdict = "❌ No Similar"
                st.write(f"🖼 Comparison {i+1} — Similarity: `{similarity:.2f}%` → {verdict}")
    else:
        st.warning("Please upload both main and comparison images, and ensure the model is loaded.")

# ----------------- Signature Section -----------------
st.subheader("✍ Signature Verification")
main_sig_image = None
main_sig_method = st.radio("Choose Signature Input", ["Upload", "Capture"], key="main_sig_method", horizontal=True)

if main_sig_method == "Upload":
    main_sig_upload = st.file_uploader("Upload Original Signature", type=["jpg", "jpeg", "png"], key="original_sig")
    if main_sig_upload:
        main_sig_image = Image.open(main_sig_upload)
        st.image(main_sig_image, caption="Original Signature", width=150)
        save_image(main_sig_image, SIG_MAIN_PATH, "original_sig")

elif main_sig_method == "Capture":
    main_sig_camera = st.camera_input("Capture Original Signature", key="original_sig_capture")
    if main_sig_camera:
        main_sig_image = Image.open(main_sig_camera)
        st.image(main_sig_image, caption="Original Signature", width=150)
        save_image(main_sig_image, SIG_MAIN_PATH, "original_sig")

compare_sig_images = []
uploaded_sigs = st.file_uploader("Upload Potential Fake Signatures to Compare", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="fake_sigs")
if uploaded_sigs:
    compare_sig_images = [Image.open(img) for img in uploaded_sigs]
    for idx, img in enumerate(compare_sig_images):
        st.image(img, caption=f"Potential Fake Signature {idx+1}", width=100)
        save_image(img, SIG_COMPARE_PATH, f"fake_sig_{idx+1}")

if st.button("🔍 Compare Signatures"):
    if main_sig_image and compare_sig_images and siamese_model:
        with st.spinner("Comparing..."):
            similarities = compare_images(siamese_model, main_sig_image, compare_sig_images)
            for i, similarity in enumerate(similarities):
                verdict = "✅ Likely Original" if similarity >= 70 else "❌ Likely Fake"
                st.write(f"🖊 Signature {i+1} — Similarity to Original: `{similarity:.2f}%` → {verdict}")
    else:
        st.warning("Please upload an original signature and potential fake signatures, and ensure the model is loaded.")