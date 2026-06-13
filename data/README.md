# Skin Cancer Classification: CNN vs Self-Supervised Learning

**MSc Data Science Dissertation — Manchester Metropolitan University (2023–2024)**  
**Supervisor:** Dr. Yanlong Zhang | **Dataset:** ISIC 2018 Challenge (Task 3)

---

## Overview

This project investigates whether self-supervised learning (SSL) methods can match the classification performance of supervised convolutional neural networks for dermatological image analysis — a domain where expert-labelled data is expensive and scarce.

Four supervised CNN architectures (ResNet50, VGG16, InceptionV3, EfficientNetB0) are benchmarked against two SSL frameworks (SimCLR and BYOL), with particular focus on **label efficiency**: how well SSL-pretrained models perform when fine-tuned on only 10%, 25%, 35%, or 50% of the available labelled data.

The central finding is that SSL pre-training enables models to achieve competitive performance with a fraction of the labelled data required by fully supervised CNNs — a result with direct implications for clinical AI deployment where annotation budgets are constrained.

---

## Dataset

**ISIC 2018 Skin Lesion Analysis Challenge — Task 3**  
7-class dermoscopy image classification (highly imbalanced)

| Class | Label | Training Samples |
|-------|-------|-----------------|
| Melanoma | MEL | 1,113 |
| Melanocytic Nevi | NV | 6,705 |
| Basal Cell Carcinoma | BCC | 514 |
| Actinic Keratosis | AKIEC | 327 |
| Benign Keratosis | BKL | 1,099 |
| Dermatofibroma | DF | 115 |
| Vascular Lesion | VASC | 142 |

**Download:** [ISIC 2018 Challenge](https://challenge.isic-archive.com/data/#2018)

---

## Results Summary

### Supervised CNN Benchmarks (Full Labels)

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| ResNet50 | 75% | Best supervised performer |
| VGG16 | 73% | Stable, slower convergence |
| InceptionV3 | 61% | Sensitive to class imbalance |
| EfficientNetB0 | 54% | Underperformed at this scale |

### SSL Label Efficiency (SimCLR — 100 epochs pretraining)

| Fine-tune Data | Test Accuracy |
|---------------|--------------|
| 10% labels | ~66% |
| 25% labels | ~68% |
| 35% labels | ~69% |
| 50% labels | ~70% |

**Key finding:** SimCLR with 25% of labels approaches the performance of fully supervised VGG16 (73%) trained on 100% of labels, demonstrating strong label efficiency.

### BYOL (Bootstrap Your Own Latent)

BYOL achieved ~65% test accuracy in linear probe evaluation, with improved feature clustering quality observed via t-SNE and UMAP visualisations compared to random initialisation.

---

## Methodology

### 1. Supervised CNN Training
- Transfer learning from ImageNet weights
- Custom focal loss to handle class imbalance (γ=2, α=0.25)
- L2 regularisation (λ=0.01) on classification head
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Class-weighted training to address NV class dominance

### 2. SimCLR Pre-training
- Backbone: ResNet50 (random initialisation, no ImageNet weights)
- Projection head: 2048 → 512 → 128
- Loss: NT-Xent (normalised temperature-scaled cross-entropy, τ=0.1)
- Augmentations: random flip, brightness, contrast, saturation, hue, crop, rotation
- Training: 100 epochs, Adam + ExponentialDecay LR, mixed precision (float16)
- Downstream evaluation: logistic regression on frozen features

### 3. BYOL Pre-training
- Online + target network with EMA updates (τ=0.99)
- Backbone: ResNet50 (ImageNet weights)
- Loss: negative cosine similarity between online predictions and target projections
- Training: 20 epochs, Adam (lr=1e-4), LR step decay

### 4. Label Efficiency Experiments
- Stratified sampling at 10%, 25%, 35%, 50% of training labels
- Fine-tuning: frozen backbone + new classification head (Dense 256 → Dropout 0.5 → Dense 7)
- Adam (lr=1e-5), categorical cross-entropy, EarlyStopping patience=10

### 5. Evaluation & Visualisation
- Classification report (per-class precision, recall, F1)
- Normalised confusion matrices
- Per-class ROC curves and Precision-Recall curves
- t-SNE and UMAP embedding visualisations
- K-means clustering (ARI, NMI scores)

---

## Repository Structure

```
skin-cancer-ssl-vs-cnn/
│
├── train.py                    # Full supervised CNN training pipeline
├── train_simclr.py             # SimCLR self-supervised pretraining
├── train_byol.py               # BYOL self-supervised pretraining
├── finetune.py                 # SSL fine-tuning with label efficiency experiments
├── predict.py                  # Inference script for a single image
├── evaluate.py                 # Evaluation: metrics, confusion matrices, curves
│
├── data/
│   └── README.md               # ISIC 2018 download instructions
│
├── models/                     # Saved .keras model files (not tracked in git)
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.x, Keras |
| Computer Vision | OpenCV, Pillow |
| Classical ML | scikit-learn (LogReg, SVM, KMeans) |
| Visualisation | Matplotlib, Seaborn, Plotly, t-SNE, UMAP |
| Data Handling | NumPy, Pandas |
| Environment | Google Colab (T4/A100 GPU), Python 3.10+ |

---

## Setup & Reproducibility

### 1. Clone the repository
```bash
git clone https://github.com/SultanZia/skin-cancer-ssl-vs-cnn.git
cd skin-cancer-ssl-vs-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download ISIC 2018 data
Follow instructions in `data/README.md`. The expected structure after extraction:
```
data/
├── ISIC2018_Task3_Training_Input/
├── ISIC2018_Task3_Training_GroundTruth/
├── ISIC2018_Task3_Validation_Input/
├── ISIC2018_Task3_Validation_GroundTruth/
├── ISIC2018_Task3_Test_Input/
└── ISIC2018_Task3_Test_GroundTruth/
```

### 4. Train supervised CNNs
```bash
python train.py --model resnet50 --epochs 10 --data_dir ./data
```

### 5. Pretrain SimCLR
```bash
python train_simclr.py --epochs 100 --batch_size 64 --data_dir ./data
```

### 6. Fine-tune with limited labels
```bash
python finetune.py --model_path models/simclr_model100.keras --label_fraction 0.1
```

### 7. Run inference
```bash
python predict.py --model_path models/resnet50_final.keras --image_path path/to/image.jpg
```

---

## Academic Context

This dissertation contributes to the growing literature on data-efficient medical AI, demonstrating that SSL pre-training can substantially reduce the labelling burden for clinical image classification tasks. The label efficiency results align with findings from the original SimCLR paper (Chen et al., 2020) and have direct applicability in pathology and dermatology settings where expert annotation is a bottleneck.

---

## Author

**Mohammed Zia Sultan**  
MSc Data Science, Manchester Metropolitan University (2023–2024)  
[github.com/SultanZia](https://github.com/SultanZia)
