"""
train.py — Supervised CNN Training for ISIC 2018 Skin Cancer Classification

Trains one of four CNN architectures (ResNet50, VGG16, InceptionV3, EfficientNetB0)
on the ISIC 2018 Task 3 dataset using transfer learning, focal loss, and class-weighted
training to address severe class imbalance.

Usage:
    python train.py --model resnet50 --epochs 10 --data_dir ./data
    python train.py --model vgg16 --epochs 10 --batch_size 32
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K


# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
NUM_CLASSES = 7
IMAGE_SIZE = (224, 224)

# ISIC 2018 training set class distribution (used for class weight calculation)
CLASS_DISTRIBUTION = {
    0: 1113,   # MEL
    1: 6705,   # NV
    2: 514,    # BCC
    3: 327,    # AKIEC
    4: 1099,   # BKL
    5: 115,    # DF
    6: 142,    # VASC
}


# ── Loss Function ─────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal loss for multi-class classification on imbalanced data.
    Reduces the relative loss for well-classified examples, focusing
    training on hard, misclassified examples.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Weighting factor for the positive class.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        fl = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_sum(fl, axis=-1)

    return focal_loss_fixed


# ── Data Pipeline ─────────────────────────────────────────────────────────────

def parse_image(image_path: str, label: tf.Tensor):
    """Decode, resize, and normalise a single image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


def create_dataset(csv_path: str, image_dir: str, batch_size: int = 32,
                   shuffle: bool = True) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from an ISIC ground truth CSV and image directory.

    Args:
        csv_path:   Path to the ground truth CSV (columns: image, MEL, NV, ..., VASC).
        image_dir:  Directory containing the .jpg images.
        batch_size: Batch size for training/evaluation.
        shuffle:    Whether to shuffle the dataset (disable for val/test).

    Returns:
        A batched, prefetched tf.data.Dataset.
    """
    df = pd.read_csv(csv_path)
    image_paths = [os.path.join(image_dir, fname) for fname in df['image']]
    labels = df.drop(columns=['image']).values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ── Model Definitions ─────────────────────────────────────────────────────────

def build_model(model_name: str) -> tf.keras.Model:
    """
    Build a transfer learning model with ImageNet weights and an L2-regularised
    classification head.

    Args:
        model_name: One of 'resnet50', 'vgg16', 'inceptionv3', 'efficientnetb0'.

    Returns:
        An uncompiled Keras model.
    """
    backbone_map = {
        'resnet50':       tf.keras.applications.ResNet50,
        'vgg16':          tf.keras.applications.VGG16,
        'inceptionv3':    tf.keras.applications.InceptionV3,
        'efficientnetb0': tf.keras.applications.EfficientNetB0,
    }

    if model_name not in backbone_map:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(backbone_map.keys())}"
        )

    base_model = backbone_map[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(
        NUM_CLASSES,
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.01)
    )(x)

    return Model(inputs=base_model.input, outputs=predictions)


# ── Training ──────────────────────────────────────────────────────────────────

def compute_class_weights(class_distribution: dict) -> dict:
    """Compute inverse-frequency class weights to counter class imbalance."""
    total = sum(class_distribution.values())
    n_classes = len(class_distribution)
    return {
        cls: total / (n_classes * count)
        for cls, count in class_distribution.items()
    }


def train(args):
    """Full training pipeline."""

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = args.data_dir
    train_csv = os.path.join(data_dir, 'ISIC2018_Task3_Training_GroundTruth',
                              'ISIC2018_Task3_Training_GroundTruth.csv')
    val_csv   = os.path.join(data_dir, 'ISIC2018_Task3_Validation_GroundTruth',
                              'ISIC2018_Task3_Validation_GroundTruth.csv')
    train_img = os.path.join(data_dir, 'ISIC2018_Task3_Training_Input')
    val_img   = os.path.join(data_dir, 'ISIC2018_Task3_Validation_Input')

    print(f"Loading datasets from {data_dir}...")
    train_data = create_dataset(train_csv, train_img, args.batch_size, shuffle=True)
    val_data   = create_dataset(val_csv,   val_img,   args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Building {args.model.upper()} model...")
    model = build_model(args.model)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=f'{args.model}_best_model.keras',
        monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(),
        metrics=['accuracy']
    )

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weights = compute_class_weights(CLASS_DISTRIBUTION)
    print(f"Class weights: {class_weights}")

    # ── Fit ────────────────────────────────────────────────────────────────────
    print(f"\nTraining {args.model.upper()} for {args.epochs} epochs...")
    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=val_data,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights
    )

    # ── Save final model ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.model}_final.keras')
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a supervised CNN on ISIC 2018 skin cancer data.'
    )
    parser.add_argument(
        '--model', type=str, default='resnet50',
        choices=['resnet50', 'vgg16', 'inceptionv3', 'efficientnetb0'],
        help='CNN architecture to train (default: resnet50)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='Root directory containing ISIC 2018 data folders (default: ./data)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./models',
        help='Directory to save trained models (default: ./models)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
