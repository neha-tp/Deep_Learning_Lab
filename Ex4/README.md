# Experiment 4: Transfer Learning and Comparison of Deep CNN Architectures on CIFAR-10

This folder contains a comprehensive study and implementation of various classic and state-of-the-art Convolutional Neural Network (CNN) architectures on the CIFAR-10 dataset, covering scratch training, transfer learning, fine-tuning, and a large-scale hyperparameter sweep.

## Overview of Architectures Evaluated
The experiment compares five major architectures:
1. **LeNet-5 (Trained from scratch)**: The classic shallow network consisting of Conv2D and AveragePooling2D layers.
2. **AlexNet (Trained from scratch)**: A deeper CNN architecture featuring multiple Conv2D layers, MaxPooling2D, and Dropout layers for regularization.
3. **GoogleNet (Trained from scratch)**: An implementation utilizing custom **Inception modules** to concatenate parallel 1 × 1, 3 × 3, and 5 × 5 convolutions and 3 × 3 pooling.
4. **VGG16 (Transfer Learning)**: Leveraging pre-trained weights on ImageNet. Evaluated under two modes:
   - *Frozen Base*: The feature extraction layers are frozen, and only the dense classification head is trained.
   - *Fine-Tuned*: Unfreezing the last 4 layers of the VGG16 base block to allow custom feature adaptation with a small learning rate.
5. **ResNet50 (Transfer Learning)**: Leveraging pre-trained ImageNet weights. Similarly evaluated under:
   - *Frozen Base*: All bottleneck residual blocks are frozen.
   - *Fine-Tuned*: Unfreezing the last 10 layers of the ResNet50 base block.

## Dataset Information
The models utilize the CIFAR-10 dataset:
* **Training Images**: 50,000 samples
* **Testing Images**: 10,000 samples
* **Resolution**: 32 × 32 pixels, 3 channels (RGB)
* **Features**: Normalized pixel values scaled between [0, 1] after dividing by 255.0.
* **Target Classes (10 categories)**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

## Model Architectures & Builders
* **LeNet-5**: Input -> Conv2D (6 filters, 5 × 5) -> AvgPool -> Conv2D (16 filters, 5 × 5) -> AvgPool -> Flatten -> Dense (120) -> Dense (84) -> Dense (10, Softmax).
* **AlexNet**: Input -> Conv2D (96 filters, 3 × 3) -> MaxPool -> Conv2D (256 filters, 3 × 3) -> MaxPool -> Conv2D (384, 3 × 3) -> Conv2D (384, 3 × 3) -> Conv2D (256, 3 × 3) -> MaxPool -> Flatten -> Dense (1024, Dropout 0.5) -> Dense (1024, Dropout 0.5) -> Dense (10, Softmax).
* **GoogleNet (Inception)**: Input -> Conv2D (64) -> MaxPool -> Inception Block (3a) -> Inception Block (3b) -> MaxPool -> GlobalAvgPool -> Dropout (0.4) -> Dense (10, Softmax).
* **VGG16 Classifier**: Pre-trained base -> GlobalAvgPool -> Dense (256, ReLU) -> Dense (10, Softmax).
* **ResNet50 Classifier**: Pre-trained base -> GlobalAvgPool -> Dense (256, ReLU) -> Dense (10, Softmax).

## Hyperparameter Search Space
A grid search evaluates combinations of VGG16 transfer configurations to analyze network performance:
* **Learning Rates**: 0.001, 0.0001
* **Batch Sizes**: 16, 32, 64
* **Optimizers**: Adam, SGD
* **Dense Units**: 128, 256
* **Frozen Layer Configurations**: "All" (Frozen Base) vs. "Partial" (Fine-Tuning the last 4 layers)

## Dependency List
To run the notebook, ensure Python 3.8+ is installed along with the following packages:
* **TensorFlow** (`tensorflow`) - Deep learning framework for building, training, and transfer learning.
* **NumPy** (`numpy`) - Matrix operations and array conversions.
* **Pandas** (`pandas`) - Data structure for storing and comparing model metrics and hyperparameter sweep records.
* **Scikit-Learn** (`scikit-learn`) - Evaluating models using Accuracy, Precision, Recall, F1-Score, and Classification Reports.
* **Matplotlib** (`matplotlib`) - Plotting sample grids, training curves, and confusion matrix heatmaps.
* **Seaborn** (`seaborn`) - Visualizing heatmaps.
* **Jupyter Notebook / Google Colab**

## Execution Instructions

### Option 1: Running Locally
1. **Navigate to Folder**:
   ```bash
   cd Deep_Learning_Lab/Ex4
   ```
2. **Establish Virtual Environment**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn jupyter
   ```
4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
5. Open `ex4.ipynb` and run all code cells.

### Option 2: Running in Google Colab
1. Navigate to Google Colab.
2. Select **Upload** and choose `ex4.ipynb` from this folder.
3. Colab will execute the first cell to install libraries:
   ```python
   !pip install -q tensorflow numpy pandas matplotlib seaborn scikit-learn
   ```
4. Run all code cells (`Runtime -> Run all`) to perform dataset loading, model training/resuming from checkpoints, transfer learning, fine-tuning, and the hyperparameter study.

## Key Visualizations & Outputs Generated
The notebook outputs comprehensive diagnostic plots, comparisons, and CSV files:
* **Sample Images Grid**: Saves a grid of 10 labeled CIFAR-10 training samples as `sample_images.png`.
* **Training Dynamics (Accuracy/Loss Curves)**: Saves accuracy and loss plots for all models:
  - LeNet-5: `lenet5_accuracy.png` and `lenet5_loss.png`
  - AlexNet: `alexnet_accuracy.png` and `alexnet_loss.png`
  - VGG16 (Frozen): `vgg16_frozen_accuracy.png` and `vgg16_frozen_loss.png`
  - VGG16 (Fine-Tuned): `vgg16_finetuned_accuracy.png` and `vgg16_finetuned_loss.png`
  - ResNet50 (Frozen): `resnet50_frozen_accuracy.png` and `resnet50_frozen_loss.png`
  - ResNet50 (Fine-Tuned): `resnet50_finetuned_accuracy.png` and `resnet50_finetuned_loss.png`
  - GoogleNet: `googlenet_accuracy.png` and `googlenet_loss.png`
* **Confusion Matrix**: Heatmap rendering VGG16 fine-tuned predictions (`confusion_matrix_vgg16.png`).
* **Comparison Tables**:
  - `comparison_table_18_2.csv`: Summary of Parameters, Test Accuracy (%), and Training Time (s) across all 5 architectures.
  - `precision_recall_f1_table.csv`: Detailed Precision, Recall, and F1-Scores (macro averages) for each of the 5 architectures.
  - `hyperparameter_study_results.csv`: Row-by-row logs of the hyperparameter search iterations with Validation Accuracy.
