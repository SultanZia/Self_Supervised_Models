"""
predict.py — Inference Script for ISIC 2018 Skin Cancer Classification

Loads a trained Keras model and classifies a single dermoscopy image,
returning the predicted class, confidence score, and per-class probabilities.

Compatible with all models trained via train.py (ResNet50, VGG16, InceptionV3,
EfficientNetB0) and fine-tuned SimCLR/BYOL models from finetune.py.

Usage:
    python predict.py --model_path models/resnet50_final.keras --image_path path/to/image.jpg
    python predict.py --model_path models/finetuned_simclr_model.keras --image_path lesion.jpg
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

CLASS_DESCRIPTIONS = {
    'MEL':   'Melanoma',
    'NV':    'Melanocytic Nevi',
    'BCC':   'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'BKL':   'Benign Keratosis',
    'DF':    'Dermatofibroma',
    'VASC':  'Vascular Lesion',
}

IMAGE_SIZE = (224, 224)


# ── Custom objects (needed to load models trained with focal loss) ────────────

@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Re-registered focal loss for model loading compatibility."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        fl = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_sum(fl, axis=-1)
    return focal_loss_fixed


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load, resize, and normalise a dermoscopy image for model inference.

    Args:
        image_path: Path to the input .jpg / .png image file.

    Returns:
        A (1, 224, 224, 3) float32 numpy array, pixel values in [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # Add batch dimension


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model_path: str, image_path: str, top_k: int = 3) -> dict:
    """
    Run inference on a single image and return structured predictions.

    Args:
        model_path: Path to the saved .keras model file.
        image_path: Path to the input dermoscopy image.
        top_k:      Number of top predictions to return (default: 3).

    Returns:
        A dict with keys:
            - predicted_class:    Short label (e.g. 'MEL')
            - predicted_label:    Full description (e.g. 'Melanoma')
            - confidence:         Confidence of top prediction (0–1)
            - top_k_predictions:  List of (class, description, probability) tuples
            - all_probabilities:  Dict mapping each class to its probability
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'focal_loss_fixed': focal_loss()}
    )

    print(f"Preprocessing image: {image_path}")
    image_array = preprocess_image(image_path)

    print("Running inference...")
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    # Top-k predictions
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_preds = [
        (CLASS_NAMES[i], CLASS_DESCRIPTIONS[CLASS_NAMES[i]], float(probabilities[i]))
        for i in top_k_indices
    ]

    return {
        'predicted_class':   predicted_class,
        'predicted_label':   CLASS_DESCRIPTIONS[predicted_class],
        'confidence':        confidence,
        'top_k_predictions': top_k_preds,
        'all_probabilities': {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
    }


def print_results(results: dict):
    """Pretty-print prediction results to stdout."""
    print("\n" + "=" * 55)
    print("  SKIN LESION CLASSIFICATION RESULT")
    print("=" * 55)
    print(f"  Predicted class : {results['predicted_class']}")
    print(f"  Description     : {results['predicted_label']}")
    print(f"  Confidence      : {results['confidence']:.1%}")
    print()
    print("  Top-3 predictions:")
    for rank, (cls, desc, prob) in enumerate(results['top_k_predictions'], 1):
        bar = '█' * int(prob * 20)
        print(f"    {rank}. {cls:<8} {desc:<45} {prob:.1%}  {bar}")
    print("=" * 55)
    print("\n  ⚠  This tool is for research purposes only.")
    print("     Clinical decisions require a qualified dermatologist.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify a skin lesion image using a trained Keras model.'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to the saved .keras model file'
    )
    parser.add_argument(
        '--image_path', type=str, required=True,
        help='Path to the input dermoscopy image (.jpg or .png)'
    )
    parser.add_argument(
        '--top_k', type=int, default=3,
        help='Number of top predictions to display (default: 3)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = predict(args.model_path, args.image_path, args.top_k)
    print_results(results)
