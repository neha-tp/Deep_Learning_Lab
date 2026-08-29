# Experiment 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning, and Cross-Validation

This directory contains the implementation and experimental study of Convolutional Neural Networks (CNNs) using the **MobileNetV2** architecture on the **Oxford-IIIT Pet Dataset**. The experiment systematically evaluates the impacts of weight initialization, regularization, optimization algorithms, hyperparameters, transfer learning, and cross-validation on image classification performance.

---

## 1. Objective & Learning Outcomes
* **Objective**: Systematically study the effect of design choices—specifically weight initialization, regularization, optimization, CNN hyperparameters, transfer learning, fine-tuning, and cross-validation—on image classification.
* **Learning Outcomes**:
  * Analyze and compare weight initialization strategies (Zero, Random, Xavier, and He).
  * Identify and mitigate overfitting using training/validation curves.
  * Compare optimizer convergence (SGD, Momentum, RMSProp, and Adam).
  * Understand Batch Normalization, Dropout, and basic CNN hyperparameters.
  * Implement transfer learning (Feature Extraction vs. Fine-Tuning) using MobileNetV2.
  * Apply 5-fold cross-validation for model selection and evaluate performance on an independent test set.

---

## 2. Dataset & Experimental Setup
* **Dataset**: Oxford-IIIT Pet Dataset (cats and dogs belonging to 37 breeds).
* **Pre-processing**: All images are resized to $224 \times 224 \times 3$ and normalized matching the input requirements of the pre-trained MobileNetV2.
* **Base Architecture**: MobileNetV2, a lightweight CNN featuring depthwise separable convolutions, inverted residual blocks, linear bottlenecks, batch normalization, and ReLU6 activations.

---

## 3. Key Experimental Sections

### A. Weight Initialization
Comparison of initialization strategies on training speed and convergence stability:
* **Zero Initialization**
* **Random Initialization**
* **Xavier/Glorot Initialization**
* **He Initialization**

### B. Regularization & Overfitting
Evaluation of strategies to reduce the generalization gap:
* **No Regularization** (Baseline)
* **L2 Regularization**
* **Dropout**
* **Batch Normalization (BN)** (with a numerical walkthrough of mean, variance, scale, and shift operations)

### C. Optimization Algorithms
Analysis of convergence speed, loss trends, and validation performance using:
* **SGD**
* **Momentum**
* **RMSProp**
* **Adam**

### D. Hyperparameter Tuning
Controlled studies of individual hyperparameters:
* **Learning Rate**: $0.001$ vs. $0.0001$
* **Batch Size**: $16, 32, 64$
* **Dropout Rate**: $0, 0.25, 0.5$
* **Optimizer Choice**: SGD vs. Adam
* **Fine-Tuning Learning Rate**: $10^{-4}$ vs. $10^{-5}$
* **Layer Freezing**: Fully frozen base vs. partial unfreezing

### E. Transfer Learning Modes
* **Feature Extraction (Case A)**: Pre-trained MobileNetV2 base is frozen; only a custom dense classification head is trained.
* **Fine-Tuning (Case B)**: Selected upper layers of the base network are unfrozen and trained with a smaller learning rate to prevent representation destruction.

### F. 5-Fold Cross-Validation
* Selection of $3\text{--}4$ promising configurations based on individual tuning.
* Validation on the training split using 5-fold cross-validation to select the final candidate model based on accuracy, standard deviation ($\bar{A} \pm SD$), and computational efficiency.

### G. Final Model Evaluation
* Retraining the best configuration on the complete training set.
* Final evaluation on the untouched independent test set, generating precision, recall, F1-score, confusion matrix, and misclassified image analysis.

---

## 4. Dependencies
To run the notebook, ensure Python 3.8+ is installed along with the following packages:
* **TensorFlow** (`tensorflow`) - Core deep learning framework.
* **NumPy** (`numpy`) - Matrix manipulations.
* **Pandas** (`pandas`) - Structuring and comparing tabular results.
* **Scikit-Learn** (`scikit-learn`) - Cross-validation splits and evaluation metrics (accuracy, precision, recall, F1).
* **Matplotlib / Seaborn** (`matplotlib`, `seaborn`) - Visualizing loss/accuracy curves and confusion matrices.
* **Jupyter Notebook / Google Colab**

---

## 5. Execution Instructions

### Option 1: Running Locally
1. **Navigate to the Folder**:
   ```bash
   cd Deep_Learning_Lab/Ex5
   ```
2. **Establish a Virtual Environment**:
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
5. Open `ex5.ipynb` and execute all cells.

### Option 2: Running in Google Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Select the **Upload** tab and select the `ex5.ipynb` file from this directory.
3. Colab will execute the cells. If needed, install dependencies via:
   ```python
   !pip install -q tensorflow numpy pandas matplotlib seaborn scikit-learn
   ```
4. Run all code cells (`Runtime -> Run all`) to perform dataset download/caching, model construction, transfer learning sweeps, cross-validation, and final evaluation.

---

## 6. Key Visualizations & Outputs Generated
The notebook generates and visualizes the following 15 plots (as described in the experiment details):
1. **Plot 1**: Training Loss vs. Epoch for different weight initializations.
2. **Plot 2**: Validation Accuracy vs. Epoch for different weight initializations.
3. **Plot 3**: Training and Validation Accuracy vs. Epoch (Generalization Gap).
4. **Plot 4**: Training and Validation Loss vs. Epoch (Overfitting Identification).
5. **Plot 5**: Validation Accuracy with vs. without Batch Normalization.
6. **Plot 6**: Training Loss vs. Epoch for SGD, Momentum, RMSProp, and Adam.
7. **Plot 7**: Validation Accuracy vs. Epoch for different optimizers.
8. **Plot 8**: Learning Rate vs. Validation Accuracy.
9. **Plot 9**: Batch Size vs. Validation Accuracy.
10. **Plot 10**: Dropout Rate vs. Validation Accuracy.
11. **Plot 11**: Feature Extraction vs. Fine-Tuning Accuracy Curves.
12. **Plot 12**: Training and Validation Loss before and after Fine-Tuning.
13. **Plot 13**: 5-Fold Cross-Validation Accuracy for different candidate configurations (with standard deviation error bars).
14. **Plot 14**: Confusion Matrix on the test set.
15. **Plot 15**: Representative Misclassified Images with error analysis.
