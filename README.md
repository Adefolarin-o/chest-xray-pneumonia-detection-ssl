# Chest X‑Ray Pneumonia Detection using Self‑Supervised Learning (SSL) 

## Project Overview
This project implements a deep learning pipeline for detecting pneumonia from chest X‑ray images using self‑supervised learning (SSL). The approach is inspired by the SimCLR framework and leverages contrastive learning to pretrain a ResNet50 backbone with a projection head. The pretrained model is then fine‑tuned with a custom classifier to differentiate between NORMAL and PNEUMONIA cases. An interactive test explorer using IPyWidgets is provided to visualize predictions, display confidence scores, and analyze model behavior.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Model Training and Fine‑Tuning](#model-training-and-fine-tuning)
- [Evaluation & Model Saving](#evaluation--model-saving)
- [Interactive Test Explorer](#interactive-test-explorer)
- [Usage Instructions](#usage-instructions)
- [Credits](#credits)
- [License](#license)

## Installation
This project requires Python 3.x and GPU support (e.g., Google Colab with GPU enabled) for efficient training. Install the following dependencies:

```bash
pip install torch torchvision timm kagglehub scikit-learn matplotlib ipywidgets
```

Ensure that your environment supports IPyWidgets (for interactive visualizations) and that you have the necessary permissions to mount Google Drive if you plan to save/load models.

## Dataset Preparation
### Downloading the Dataset
The dataset is automatically downloaded from Kaggle using `kagglehub`. We use the dataset `paultimothymooney/chest-xray-pneumonia`, which contains chest X‑ray images labeled as NORMAL and PNEUMONIA.

### Data Splitting and Reorganization
- **SSL Pretraining**: The majority of the dataset consists of NORMAL images. To help with representation diversity (especially in small datasets), approximately 20% of PNEUMONIA images are also included.
- **Validation**: A subset of the training images is held out for SSL validation.
- **Testing**: The original test set (with both NORMAL and PNEUMONIA cases) is retained for final evaluation.

The images are shuffled and copied into separate directories:

```
chest_xray_ssl/train_ssl – for SSL pretraining
chest_xray_ssl/val_ssl – for SSL validation
chest_xray_ssl/test – for final evaluation
```

## Project Structure
The notebook is organized into several sections:

1. **Setup & Dependencies**: Installs required packages and mounts Google Drive.
2. **Dataset Download & Preparation**: Downloads the dataset, splits it, and organizes images.
3. **Data Augmentation & Custom Dataset**:
   - Implements data augmentation strategies tailored for chest X‑rays.
   - Defines a custom dataset class that returns two augmented views for contrastive learning.
4. **Model Architecture**:
   - Sets up the SSL model with a pretrained ResNet50 backbone (classification head removed) and a projection head.
   - Defines a separate classifier for pneumonia detection.
5. **Training (SSL Pretraining with Early Stopping)**: Trains the SSL model using NT‑Xent loss with early stopping callbacks.
6. **Fine‑Tuning**: Attaches a classifier to the pretrained backbone and trains on labeled data.
7. **Evaluation & Model Saving**: Evaluates on the test set and saves models to Google Drive.
8. **Interactive Test Explorer**: Provides an IPyWidgets interface for test sample selection and prediction visualization.

## Model Training and Fine‑Tuning

### Phase 1 – SSL Pretraining
- **Backbone**: ResNet50 pretrained on ImageNet (classification layer removed).
- **Projection Head**: MLP with BatchNorm, ReLU, and Dropout projects 2048‑dimensional features to a 128‑dimensional space.
- **Loss Function**: NT‑Xent loss (Normalized Temperature‑scaled Cross Entropy).
- **Early Stopping**: Training stops when validation loss does not improve.

### Phase 2 – Fine‑Tuning
- **Classifier**: Custom classification head with dense layers, BatchNorm, ReLU, and Dropout.
- **Training**: Fine-tuned on labeled data to classify NORMAL vs. PNEUMONIA.
- **Optimization**: Learning rate adjustments and callbacks ensure stable training.

## Evaluation & Model Saving
- **Evaluation**: After fine‑tuning, accuracy, loss, and AUC metrics are reported.
- **Saving Models**:
  ```bash
  best_ssl_model.pth   # Best pretrained model
  best_pneumonia_classifier.pth   # Best fine-tuned classifier
  ```
- **Loading for Inference**:
  ```python
  ssl_model = init_ssl_model()  # Initialize architecture
  ssl_model.load_state_dict(torch.load('best_ssl_model.pth'))
  ssl_model.eval()
  ```

## Interactive Test Explorer
An IPyWidgets interface allows:
- **Selecting a Test Sample**: Dropdown menu for test image selection.
- **Adjusting Confidence Threshold**: Slider to modify confidence threshold.
- **Visualizing Predictions**: Displays the image, true label, predicted label, and confidence scores.

## Usage Instructions
1. **Installation**: Follow the installation steps.
2. **Running the Notebook**:
   - Open in Google Colab or Jupyter.
   - Execute cells sequentially.
3. **Skipping Training**:
   - If pretrained models are available, skip cells 11-14(training and evaluation) and load saved models.
4. **Exploring Predictions**:
   - Use the Interactive Test Explorer to analyze model behavior.

## Credits
- Dataset: [Paul Timothy Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- SSL Framework: Inspired by SimCLR.
- Model Architecture: Built using PyTorch and timm.

## License
This project is released under the MIT License.
