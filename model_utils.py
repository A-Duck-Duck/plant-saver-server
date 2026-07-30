"""
Loads the Teachable Machine Keras model (exported as keras_model.h5)
and runs predictions on uploaded images.

Drop your exported files into a `model/` folder next to this file:
    model/keras_model.h5
    model/labels.txt
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

# Teachable Machine's exported models expect 224x224 RGB input,
# normalized to the range [-1, 1] (standard MobileNetV2 preprocessing).
IMG_SIZE = (224, 224)

_model = None
_labels = None


def _load():
    """Lazy-load the model and labels once, then reuse across requests."""
    global _model, _labels
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Export from Teachable Machine "
                "(Tensorflow tab -> Keras) and place keras_model.h5 + labels.txt in the model/ folder."
            )
        _model = load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, "r") as f:
            # labels.txt lines look like: "0 Healthy"
            _labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]
    return _model, _labels


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into the normalized array the model expects."""
    img = Image.open(image_bytes).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # normalize to [-1, 1]
    return np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)


def predict(image_bytes) -> dict:
    """
    Run a single image through the model.
    Returns: {"label": str, "confidence": float, "all_scores": {label: score}}
    """
    model, labels = _load()
    arr = preprocess_image(image_bytes)
    preds = model.predict(arr, verbose=0)[0]  # shape (num_classes,)

    all_scores = {labels[i]: float(preds[i]) for i in range(len(labels))}
    best_idx = int(np.argmax(preds))

    return {
        "label": labels[best_idx],
        "confidence": float(preds[best_idx]),
        "all_scores": all_scores,
    }


def predict_combined(plant_bytes, leaf_bytes=None) -> dict:
    """
    Runs prediction on the whole-plant photo, and if a leaf closeup is provided,
    averages the two predictions (leaf closeups are often more diagnostic,
    so it gets slightly more weight).
    """
    plant_result = predict(plant_bytes)

    if leaf_bytes is None:
        return plant_result

    leaf_result = predict(leaf_bytes)

    # Weighted average: leaf closeup counts a bit more (60/40)
    labels = plant_result["all_scores"].keys()
    combined_scores = {
        label: 0.4 * plant_result["all_scores"][label] + 0.6 * leaf_result["all_scores"][label]
        for label in labels
    }
    best_label = max(combined_scores, key=combined_scores.get)

    return {
        "label": best_label,
        "confidence": combined_scores[best_label],
        "all_scores": combined_scores,
        "plant_only_result": plant_result,
        "leaf_only_result": leaf_result,
    }
