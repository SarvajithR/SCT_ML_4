"""
==================================================================================
 SkillCraft Technology - Machine Learning Internship
 TASK 04: Hand Gesture Recognition using Deep Learning (Transfer Learning)
==================================================================================

This script launches a Gradio web application that allows a user to upload an
image of a hand gesture and receive a real-time prediction of the gesture class
along with a confidence score, powered by a MobileNetV2-based transfer learning
model trained on the Kaggle "LeapGestRecog" dataset.

Author : SkillCraft ML Intern
Dataset: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
==================================================================================
"""

import os
import numpy as np
import gradio as gr
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
# A human-friendly display label is mapped for each raw class name.
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
# 2. MODEL LOADING
# ----------------------------------------------------------------------------
def load_gesture_model(model_path: str):
    """
    Load the trained Keras MobileNetV2 transfer-learning model from disk.

    Returns
    -------
    tf.keras.Model or None
        The loaded model, or None if the file could not be found / loaded.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file not found at: {model_path}")
        print("[WARNING] Please run 'hand_gesture_training.ipynb' first to "
              "train and export the model.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Model successfully loaded from: {model_path}")
        return model
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Failed to load model: {exc}")
        return None


# Load the model once at startup so repeated predictions are fast.
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
        1. A dictionary {class_label: probability} for gr.Label (top-5 shown).
        2. A Markdown-formatted summary describing the top prediction.

    This function is wired directly to the "Predict Gesture" button in the
    Gradio interface.
    """
    if image is None:
        return {}, "⚠️ Please upload an image first."

    if gesture_model is None:
        return {}, (
            "❌ **Model not found.** Please train the model using "
            "`hand_gesture_training.ipynb` and place `gesture_model.h5` "
            "inside the `saved_model/` directory."
        )

    # Preprocess and run inference
    processed = preprocess_image(image)
    predictions = gesture_model.predict(processed, verbose=0)[0]

    # Build the {label: confidence} dictionary for gr.Label
    confidence_scores = {
        DISPLAY_LABELS[CLASS_NAMES[i]]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    # Identify the top prediction
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
    Collect a list of example image file paths from the `images/` directory
    to populate the Gradio Examples gallery. Falls back to an empty list if
    the directory does not exist or has no images.
    """
    if not os.path.isdir(examples_dir):
        return []

    valid_extensions = (".png", ".jpg", ".jpeg")
    files = sorted(
        os.path.join(examples_dir, f)
        for f in os.listdir(examples_dir)
        if f.lower().endswith(valid_extensions)
    )
    return files


EXAMPLE_IMAGES = get_example_images(EXAMPLES_DIR)


# ----------------------------------------------------------------------------
# 6. CUSTOM STYLING (Clean, Light Theme)
# ----------------------------------------------------------------------------
custom_css = """
#app-title {
    text-align: center;
    font-weight: 700;
    color: #1f2937;
}
#app-subtitle {
    text-align: center;
    color: #4b5563;
    margin-bottom: 1rem;
}
.gradio-container {
    background-color: #f9fafb !important;
}
footer {visibility: hidden}
"""

light_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill="#f9fafb",
    block_background_fill="#ffffff",
    block_border_color="#e5e7eb",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
)


# ----------------------------------------------------------------------------
# 7. GRADIO INTERFACE
# ----------------------------------------------------------------------------
with gr.Blocks(theme=light_theme, css=custom_css, title="Hand Gesture Recognition AI") as demo:

    gr.Markdown(
        "# ✋ AI-Powered Hand Gesture Recognition",
        elem_id="app-title",
    )
    gr.Markdown(
        "### SkillCraft Technology — Machine Learning Internship | Task 04\n"
        "Upload a hand gesture image and let a **MobileNetV2 transfer-learning "
        "model** predict which of the **10 LeapGestRecog gestures** it represents, "
        "complete with a confidence score.",
        elem_id="app-subtitle",
    )

    with gr.Row():
        # ---------------- Left column: input ----------------
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📤 Upload Hand Gesture Image",
                height=300,
            )
            predict_btn = gr.Button("🔍 Predict Gesture", variant="primary")

            gr.Examples(
                examples=EXAMPLE_IMAGES,
                inputs=image_input,
                label="🖼️ Example Gestures (click to try)",
                examples_per_page=10,
            )

        # ---------------- Right column: output ----------------
        with gr.Column(scale=1):
            result_summary = gr.Markdown(
                value="### 🤖 Prediction results will appear here.",
                label="Result",
            )
            confidence_label = gr.Label(
                label="📊 Confidence Scores (All Classes)",
                num_top_classes=5,
            )

    gr.Markdown(
        "---\n"
        "#### ℹ️ About the Gestures\n"
        "| Class | Gesture | Description |\n"
        "|---|---|---|\n"
        + "\n".join(
            f"| `{c}` | {DISPLAY_LABELS[c]} | {GESTURE_DESCRIPTIONS[c]} |"
            for c in CLASS_NAMES
        )
    )

    gr.Markdown(
        "---\n"
        "🔧 **Model:** MobileNetV2 (ImageNet weights, frozen base) + custom "
        "dense classification head, fine-tuned on the "
        "[LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).  \n"
        "👨‍💻 Built with **TensorFlow / Keras** + **Gradio**.",
    )

    # Wire up the predict button
    predict_btn.click(
        fn=predict_gesture,
        inputs=image_input,
        outputs=[confidence_label, result_summary],
    )


# ----------------------------------------------------------------------------
# 8. LAUNCH
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
