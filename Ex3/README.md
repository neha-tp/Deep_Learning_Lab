# Experiment 3: Convolutional Neural Network (CNN) for Image Classification on CIFAR-10

This folder contains the implementation of a Convolutional Neural Network (CNN) using TensorFlow/Keras to perform image classification on the CIFAR-10 dataset.

## Dataset Information
The model utilizes the CIFAR-10 dataset:
* **Training Images**: 50,000 samples
* **Testing Images**: 10,000 samples
* **Resolution**: 32 × 32 pixels, 3 channels (RGB)
* **Features**: Normalized pixel values scaled between [0, 1] after dividing by 255.0.
* **Target Classes (10 categories)**:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

## Model Architecture
The network is built using the Keras Sequential API with the following structure:
* **Input Layer**: shape `(32, 32, 3)`
* **Convolutional Layer 1**: 32 filters, 3 × 3 kernel, ReLU activation, same padding
* **Max Pooling Layer 1**: 2 × 2 pool size
* **Convolutional Layer 2**: 64 filters, 3 × 3 kernel, ReLU activation, same padding
* **Max Pooling Layer 2**: 2 × 2 pool size
* **Convolutional Layer 3**: 128 filters, 3 × 3 kernel, ReLU activation, same padding
* **Max Pooling Layer 3**: 2 × 2 pool size
* **Flatten Layer**: Converts the 3D feature maps into a 1D vector
* **Dense (Hidden) Layer**: 128 neurons, ReLU activation
* **Dropout Layer**: Dropout rate of 0.5 to prevent overfitting
* **Output Layer**: 10 neurons, Softmax activation

* **Optimizer**: Adam
* **Loss Function**: Categorical Cross-Entropy
* **Training Parameters**: Trained for 10 epochs with a batch size of 64 and a validation split of 20%.

## Dependency List
To run the notebook, ensure Python 3.8+ is installed along with the following packages:
* **TensorFlow** (`tensorflow`) - For building and training the CNN.
* **NumPy** (`numpy`) - For matrix operations and data manipulation.
* **Scikit-Learn** (`scikit-learn`) - For evaluation metrics (Accuracy, Precision, Recall, F1-Score) and classification reports.
* **Matplotlib** (`matplotlib`) - For plotting sample images, training curves, and feature maps.
* **Seaborn** (`seaborn`) - For generating countplots and confusion matrix heatmaps.
* **Jupyter Notebook / Google Colab**

## Execution Instructions

### Option 1: Running Locally
1. **Navigate to Folder**:
   ```bash
   cd Deep_Learning_Lab/Ex3
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
   pip install numpy matplotlib seaborn tensorflow scikit-learn jupyter
   ```
4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
5. Open `ex3.ipynb` and run all code cells.

### Option 2: Running in Google Colab
1. Navigate to Google Colab.
2. Select **Upload** and choose `ex3.ipynb` from this folder.
3. Run all code cells (`Runtime -> Run all`) to automatically load the CIFAR-10 dataset, train the model, and visualize predictions.

## Key Visualizations & Outputs Generated
The notebook outputs comprehensive diagnostic plots:
* **Sample Images grid**: Renders a 2x5 grid showing 10 sample images from the training set with their corresponding class labels.
* **Class Distribution plot**: A countplot verifying the balanced class distribution of the training set.
* **Training Dynamics**: Curves plotting Training & Validation Accuracy and Loss over 10 epochs.
* **Evaluation Metrics**: Prints test Accuracy, Loss, weighted Precision, weighted Recall, and weighted F1-Score.
* **Classification Report**: Outputs a per-class breakdown of Precision, Recall, and F1-Score.
* **Confusion Matrix**: A blue-themed heatmap detailing predicted versus actual classes.
* **Feature Map Visualizations**: Viridis-colored heatmaps rendering the activation maps from all three Convolution layers to visualize how the network extracts features (edges, textures, shapes) from a sample test image (e.g., an airplane).
