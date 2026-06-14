"""
==================================================================================
 SkillCraft Technology - Machine Learning Internship
 TASK 04: Hand Gesture Recognition using Deep Learning (Transfer Learning)
==================================================================================

This Streamlit web app allows a user to upload an image of a hand gesture and
receive a real-time prediction of the gesture class along with a confidence
score, powered by a MobileNetV2-based transfer learning model trained on the
Kaggle "LeapGestRecog" dataset.

Author : SkillCraft ML Intern
Dataset: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
==================================================================================
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image


# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "gesture_model.h5")
EXAMPLES_DIR = os.path.join(BASE_DIR, "images")

# Target image size expected by the MobileNetV2 backbone
IMG_SIZE = (224, 224)

# Class names correspond to the 10 gesture folders inside the
# LeapGestRecog dataset (in their natural sorted order: 01_palm ... 10_down).
CLASS_NAMES = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]

DISPLAY_LABELS = {
    "01_palm": "Open Palm 🖐️",
    "02_l": "L Shape 👆",
    "03_fist": "Closed Fist ✊",
    "04_fist_moved": "Moving Fist ✊↔️",
    "05_thumb": "Thumbs Up 👍",
    "06_index": "Index Finger ☝️",
    "07_ok": "OK Sign 👌",
    "08_palm_moved": "Moving Palm 🖐️↔️",
    "09_c": "C Shape 🤏",
    "10_down": "Thumbs Down 👎",
}

GESTURE_DESCRIPTIONS = {
    "01_palm": "An open hand shown flat with all fingers extended.",
    "02_l": "The thumb and index finger form the letter 'L'.",
    "03_fist": "A clenched fist with all fingers curled inward.",
    "04_fist_moved": "A clenched fist captured while moving across the frame.",
    "05_thumb": "A fist with only the thumb extended upward.",
    "06_index": "A fist with only the index finger pointing outward.",
    "07_ok": "Thumb and index finger touching to form a circle (OK sign).",
    "08_palm_moved": "An open palm captured while moving across the frame.",
    "09_c": "Fingers curved to resemble the letter 'C'.",
    "10_down": "A fist with the thumb extended downward.",
}


# ----------------------------------------------------------------------------
# 2. MODEL LOADING (cached so it only loads once, not on every interaction)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading gesture recognition model...")
def load_gesture_model(model_path: str):
    """
    Load the trained Keras MobileNetV2 transfer-learning model from disk.

    Returns
    -------
    tf.keras.Model or None
        The loaded model, or None if the file could not be found / loaded.
    """
    if not os.path.exists(model_path):
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception:  # pragma: no cover
        return None


gesture_model = load_gesture_model(MODEL_PATH)


# ----------------------------------------------------------------------------
# 3. IMAGE PREPROCESSING
# ----------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a batch-ready numpy array suitable for
    MobileNetV2 inference.

    Steps:
        1. Convert to RGB (handles grayscale / RGBA uploads).
        2. Resize to (224, 224) as expected by MobileNetV2.
        3. Apply MobileNetV2's preprocessing function (scales pixels to [-1, 1]).
        4. Add a batch dimension.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)          # scales to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)    # shape -> (1, 224, 224, 3)

    return img_array


# ----------------------------------------------------------------------------
# 4. PREDICTION FUNCTION
# ----------------------------------------------------------------------------
def predict_gesture(image: Image.Image):
    """
    Run inference on the uploaded image and return:
        1. A dictionary {display_label: probability} for all 10 classes.
        2. A Markdown-formatted summary describing the top prediction.
    """
    processed = preprocess_image(image)
    predictions = gesture_model.predict(processed, verbose=0)[0]

    confidence_scores = {
        DISPLAY_LABELS[CLASS_NAMES[i]]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    top_index = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_index]
    top_confidence = float(predictions[top_index]) * 100

    summary = (
        f"### 🏆 Predicted Gesture: **{DISPLAY_LABELS[top_class]}**\n"
        f"**Confidence: {top_confidence:.2f}%**\n\n"
        f"📝 _{GESTURE_DESCRIPTIONS[top_class]}_"
    )

    return confidence_scores, summary


# ----------------------------------------------------------------------------
# 5. EXAMPLE IMAGES
# ----------------------------------------------------------------------------
def get_example_images(examples_dir: str):
    """
    Collect a list of example image file paths from the `images/` directory.
    Falls back to an empty list if the directory does not exist or has no
    images.
    """
    if not os.path.isdir(examples_dir):
        return []

    valid_extensions = (".png", ".jpg", ".jpeg")
    return sorted(
        os.path.join(examples_dir, f)
        for f in os.listdir(examples_dir)
        if f.lower().endswith(valid_extensions)
    )


EXAMPLE_IMAGES = get_example_images(EXAMPLES_DIR)


# ----------------------------------------------------------------------------
# 6. PAGE CONFIG + LIGHT THEME STYLING
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition AI",
    page_icon="✋",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f9fafb; }
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 6px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 7. HEADER
# ----------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#1f2937;'>"
    "✋ AI-Powered Hand Gesture Recognition</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#4b5563;'>"
    "SkillCraft Technology — Machine Learning Internship | Task 04<br>"
    "Upload a hand gesture image and let a <b>MobileNetV2 transfer-learning "
    "model</b> predict which of the <b>10 LeapGestRecog gestures</b> it "
    "represents, complete with a confidence score."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ----------------------------------------------------------------------------
# 8. SESSION STATE FOR THE CURRENTLY SELECTED IMAGE
# ----------------------------------------------------------------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None


# ----------------------------------------------------------------------------
# 9. MAIN LAYOUT
# ----------------------------------------------------------------------------
col_input, col_output = st.columns(2)

with col_input:
    uploaded_file = st.file_uploader(
        "📤 Upload Hand Gesture Image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        st.session_state.selected_image = Image.open(uploaded_file)

    if EXAMPLE_IMAGES:
        st.markdown("**🖼️ Example Gestures (click to try)**")
        example_cols = st.columns(5)
        for i, img_path in enumerate(EXAMPLE_IMAGES[:10]):
            with example_cols[i % 5]:
                st.image(img_path, use_container_width=True)
                if st.button("Use", key=f"example_{i}"):
                    st.session_state.selected_image = Image.open(img_path)

    if st.session_state.selected_image is not None:
        st.image(
            st.session_state.selected_image,
            caption="Selected Image",
            use_container_width=True,
        )

    predict_clicked = st.button("🔍 Predict Gesture", type="primary")

with col_output:
    if predict_clicked:
        if st.session_state.selected_image is None:
            st.warning("⚠️ Please upload or select an image first.")
        elif gesture_model is None:
            st.error(
                "❌ **Model not found.** Please train the model using "
                "`hand_gesture_training.ipynb` and place `gesture_model.h5` "
                "inside the `saved_model/` directory."
            )
        else:
            confidence_scores, summary = predict_gesture(
                st.session_state.selected_image
            )
            st.markdown(summary)

            st.markdown("##### 📊 Confidence Scores (Top 5)")
            sorted_scores = sorted(
                confidence_scores.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for label, score in sorted_scores:
                st.write(f"{label} — {score * 100:.2f}%")
                st.progress(min(max(score, 0.0), 1.0))
    else:
        st.markdown("### 🤖 Prediction results will appear here.")


# ----------------------------------------------------------------------------
# 10. GESTURE REFERENCE TABLE
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### ℹ️ About the Gestures")

table_md = "| Class | Gesture | Description |\n|---|---|---|\n" + "\n".join(
    f"| `{c}` | {DISPLAY_LABELS[c]} | {GESTURE_DESCRIPTIONS[c]} |"
    for c in CLASS_NAMES
)
st.markdown(table_md)

st.markdown("---")
st.markdown(
    "🔧 **Model:** MobileNetV2 (ImageNet weights, frozen base) + custom "
    "dense classification head, fine-tuned on the "
    "[LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).  \n"
    "👨‍💻 Built with **TensorFlow / Keras** + **Streamlit**."
)
