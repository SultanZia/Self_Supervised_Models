# Self_Supervised_Models
Skin Cancer Classification: CNNs vs. Self-Supervised Learning


**Overview**

This project, part of my MSc Data Science dissertation at Manchester Metropolitan University (September 2024), compares the performance of Convolutional Neural Networks (CNNs) and Self-Supervised Learning (SSL) models for classifying skin cancer from dermoscopic images. Using the ISIC 2018 dataset (10,015 images across 7 imbalanced classes), I implemented and evaluated four CNN architectures—ResNet50, VGG16, InceptionV3, and EfficientNetB0—and two SSL models—SimCLR and BYOL. The study achieved up to 82% accuracy with BYOL (50% labeled data) and demonstrated robust handling of class imbalance, offering insights into scalable medical image analysis. This work showcases my expertise in deep learning, data preprocessing, and performance evaluation using Python, TensorFlow, and Keras.


**Objectives**

Assess the efficacy of CNN models (ResNet50, VGG16, InceptionV3, EfficientNetB0) for skin cancer classification.

Investigate SSL models (SimCLR, BYOL) with pre-training on unlabeled data and fine-tuning on limited labeled subsets.

Compare CNN and SSL approaches on accuracy, precision, recall, F1-score, and AUC, addressing class imbalance.

Visualize and interpret results to identify optimal models for medical diagnostics.


**Dataset**

Source: ISIC 2018 dataset (International Skin Imaging Collaboration).

Size: 10,015 training, 193 validation, and 1,512 test images.

Classes: 7 skin lesion types (e.g., Melanoma: 1,113, Nevus: 6,705, Dermatofibroma: 115), highly imbalanced.

Access: Due to its large size (~2.5 GBs), the dataset is hosted on Google Drive. Download it here:

ISIC 2018 Dataset (https://drive.google.com/file/d/1pIKslaOiSq1BQKqPcdFiWziUyNVnWI4R/view?usp=drive_link).

Instructions: Download the folder, extract it, and place it in the same directory as Project_Final3.ipynb.

Challenges: Class imbalance, inter-class similarities (e.g., Melanoma vs. Nevus), and variable image quality.


**Methodology**

**1. Data Preprocessing**
   
Resized images to 224x224 pixels and normalized pixel values to [0, 1].

Applied augmentation (rotation, scaling) to enhance generalization.

Excluded corrupted images and handled class imbalance with focal loss and class weights.

**2. CNN Models**

Architectures: ResNet50, VGG16, InceptionV3, EfficientNetB0 (pre-trained on ImageNet).

Modifications: Replaced top layers with Global Average Pooling (GAP) and dense layers with L2 regularization.

Training: Used Adam optimizer (learning rate 1e-4), batch size 32, and callbacks (EarlyStopping, ReduceLROnPlateau).

**3. SSL Models**

SimCLR: Pre-trained on unlabeled data using contrastive learning, fine-tuned with 10%, 25%, and 50% labeled data.

BYOL: Pre-trained with online/target networks, fine-tuned with 25%, 35%, and 50% labeled data.

Training: Focal loss, class weights, and L2 regularization applied during fine-tuning.

**4. Evaluation**

Metrics: Accuracy, precision, recall, F1-score, AUC.

Visualizations: ROC curves, Precision-Recall curves, accuracy/loss plots, confusion matrices.

**Sample Results:**

ResNet50: 75% accuracy, balanced performance across classes.

BYOL (50% labeled): 82% accuracy, F1-score 0.83.

SimCLR (50% labeled): 66% accuracy, struggles with minority classes.

**Key Insights**

CNNs: ResNet50 excelled with 75% accuracy, effectively balancing majority (Nevus) and minority (Dermatofibroma) classes.

SSL: BYOL outperformed with 82% accuracy (50% labeled data), showing superior generalization on minority classes like AKIEC and DF after fine-tuning.

Class Imbalance: Focal loss and class weights mitigated bias, though inter-class similarities (e.g., MEL vs. BKL) remained challenging.

Scalability: SSL reduced reliance on labeled data, with BYOL converging faster than SimCLR.

**Setup******

Clone the repository:

bash

git clone https://github.com/[your-username]/skin-cancer-classification.git
Install dependencies:
bash

pip install tensorflow keras numpy matplotlib seaborn scikit-learn

**Download the dataset:**

Get the ISIC 2018 dataset from Google Drive.

Extract and place it in the repo directory.


**Usage**

Run the notebook to preprocess data, train models, and evaluate results:

bash

jupyter notebook Project_Final3.ipynb



**Results**

ResNet50: 75% accuracy, AUC up to 0.99 (VASC), robust across imbalanced classes.

VGG16: 82% accuracy, strong on dominant classes (NV, AUC 0.93).

BYOL (50%): 82% accuracy, F1-score 0.83, best for minority classes (DF, VASC).

SimCLR (50%): 66% accuracy, improved with labeled data but lagged on minority classes.

Visualizations: ROC/PR curves and loss plots in the notebook highlight model strengths.

**Skills Demonstrated**

Data Science: Preprocessed imbalanced medical image data, applied statistical evaluation metrics.

Deep Learning: Designed and fine-tuned CNNs (ResNet50, VGG16) and SSL models (SimCLR, BYOL).

Medical Imaging: Tackled real-world healthcare challenges with class imbalance handling.

Tools: Python, TensorFlow, Keras, Matplotlib, Seaborn, Scikit-learn.

**Author**

Mohammed Zia Sultan

