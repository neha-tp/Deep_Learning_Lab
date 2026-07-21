# Experiment 2: Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

This folder contains the implementation of a **Multi-Layer Perceptron (MLP)** using **TensorFlow/Keras** for image classification. The model is trained, evaluated, and optimized on the **Fashion-MNIST dataset** to classify fashion articles into 10 categories. Automated hyperparameter tuning is executed using `RandomizedSearchCV` via the `SciKeras` wrapper.

---

## Dataset Information

The model utilizes the **Fashion-MNIST** dataset:

* **Training Images:** 60,000 samples
* **Testing Images:** 10,000 samples
* **Resolution:** $28 \times 28$ pixels, grayscale
* **Features:** Normalized pixel values scaled between `[0, 1]` after flattening the 2D matrix into a 784-dimensional vector.
* **Target Classes (10 categories):**
  1. `T-shirt`
  2. `Trouser`
  3. `Pullover`
  4. `Dress`
  5. `Coat`
  6. `Sandal`
  7. `Shirt`
  8. `Sneaker`
  9. `Bag`
  10. `Ankle Boot`

---

## Model Architectures

### 1. Baseline Model
* **Input Layer:** 784 dimensions (flattened $28 \times 28$ image)
* **Hidden Layer 1:** Dense layer (128 units, ReLU activation)
* **Hidden Layer 2:** Dense layer (64 units, ReLU activation)
* **Output Layer:** Dense layer (10 units, Softmax activation)
* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-Entropy

### 2. Hyperparameter Search Space
Automated tuning via cross-validated random search evaluates candidate configurations across:
* **Hidden Layers:** 1, 2, or 3 layers
* **Hidden Neurons:** 32, 64, 128, or 256
* **Learning Rates:** 0.1, 0.01, 0.001
* **Optimizers:** Adam, SGD, RMSProp
* **Activation Functions:** ReLU, Tanh, Sigmoid
* **Dropout Rates:** 0.0, 0.2, 0.5
* **Batch Sizes:** 16, 32, 64, 128
* **Epochs:** 10, 20, 30

---

## Dependency List

To run the notebook or code, ensure Python 3.8+ is installed along with the following packages:

* **TensorFlow** (`tensorflow`) - For MLP building, loading dataset, and training.
* **Scikit-Learn** (`scikit-learn==1.5.2`) - Required for evaluation metrics, classification report, and parameter search.
* **SciKeras** (`scikeras`) - Scikit-Learn wrapper interface for Keras models.
* **NumPy** (`numpy`) - Matrix operations and array conversions.
* **Pandas** (`pandas`) - Structuring CV results and metric comparisons.
* **Matplotlib** (`matplotlib`) - Generating classification curves and bar charts.
* **Seaborn** (`seaborn`) - Visualizing class distribution counts and confusion matrix heatmaps.
* **Jupyter Notebook / Google Colab**

> [!IMPORTANT]
> To avoid compatibility issues between Scikit-Learn's `RandomizedSearchCV` and `SciKeras`, it is highly recommended to install the matching versions:
> ```bash
> pip uninstall -y scikit-learn
> pip install scikit-learn==1.5.2 scikeras
> ```

---

## Execution Instructions

### Option 1: Running Locally

1. **Navigate to Folder:**
   ```bash
   cd Deep_Learning_Lab/Ex2
   ```

2. **Establish Virtual Environment:**
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow
   pip install scikit-learn==1.5.2 scikeras jupyter
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Open `ex2.ipynb` and run all code cells.

---

### Option 2: Running in Google Colab

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and choose `ex2.ipynb` from this folder.
3. Colab will execute the first code cell to install the compatible library versions:
   ```python
   !pip uninstall -y scikit-learn
   !pip install scikit-learn==1.5.2
   !pip install scikeras
   ```
4. Run all code cells (`Runtime -> Run all`) to download the Fashion-MNIST dataset automatically and perform training/tuning.

---

## Key Visualizations & Outputs Generated

The notebook outputs comprehensive diagnostic plots:
* **Sample Images grid:** Renders 10 grayscale apparel samples.
* **Class Distribution plot:** Countplot verifying the dataset's class balance.
* **Baseline training dynamics:** Curves plotting Training & Validation Accuracy and Loss over 20 epochs.
* **Baseline metrics & reports:** Confusion matrix heatmap and classification report (Precision, Recall, F1).
* **Hyperparameter Search Performance:** Line plot tracking CV accuracy across search iterations.
* **Optimized Model evaluations:** Confusion matrix (Greens palette) and classification report for the tuned network.
* **Comparison Bar Chart:** Visual comparison of final test accuracy between baseline and optimized models.
