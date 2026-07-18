# Experiment 1: Single Layer Perceptron for Binary Classification

This folder contains the implementation of a **Single Layer Perceptron** from scratch using the classical perceptron learning rule and a step activation function. The model is trained and evaluated on the **Banknote Authentication Dataset** to classify banknotes as genuine or forged.

---

## Dataset Information

The **Banknote Authentication Dataset** is obtained from the **UCI Machine Learning Repository**. 

* **Source URL:** [UCI Machine Learning Repository - Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
* **Dataset File:** `data_banknote_authentication.txt` (included in this folder)
* **Number of Instances:** 1,372 (762 Genuine, 610 Forged)
* **Number of Attributes:** 4 continuous features + 1 class label

### Feature Details
1. **Variance** (continuous): Variance of the Wavelet Transformed image.
2. **Skewness** (continuous): Skewness of the Wavelet Transformed image.
3. **Curtosis** (continuous): Curtosis of the Wavelet Transformed image.
4. **Entropy** (continuous): Entropy of the image.
5. **Class** (binary target): 
   * `0`: Genuine Banknote
   * `1`: Forged Banknote

---

## Dependency List

To run the notebook or python script, the following dependencies are required:

* **Python 3.x**
* **NumPy** (`numpy`) - For mathematical computations and matrix operations.
* **Pandas** (`pandas`) - For data loading, manipulation, and summary statistics.
* **Matplotlib** (`matplotlib`) - For plotting and visualization.
* **Seaborn** (`seaborn`) - For statistical data visualization (heatmaps, countplots, boxplots).
* **Scikit-Learn** (`scikit-learn`) - For dataset splitting, standardization (`StandardScaler`), evaluation metrics, and comparison with scikit-learn's built-in `Perceptron`.
* **Jupyter Notebook / Google Colab** - For interactive execution.

---

## Execution Instructions

You can run the notebook either **locally** or in **Google Colab**.

### Option 1: Running Locally 

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/neha-tp/Deep_Learning_Lab.git
   cd Deep_Learning_Lab/Ex1
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. **Adjust the Dataset Path:**
   Open `Ex1.ipynb` in your editor or Jupyter. In the third code cell, update the dataset path to read the local copy in the same directory:
   ```python
   # Original:
   # path = "/content/drive/MyDrive/DL Lab/ex1/data_banknote_authentication.txt"
   
   # Updated for local run:
   path = "data_banknote_authentication.txt"
   ```

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `Ex1.ipynb` and run all the cells.

---

### Option 2: Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `Ex1.ipynb` notebook from this folder.
3. Upload `data_banknote_authentication.txt` to your Google Drive or upload it directly to the Colab runtime files.
4. If you upload to Colab runtime files:
   - Comment out the `google.colab` drive mount code cell.
   - Change the `path` variable to `"data_banknote_authentication.txt"`.
5. Execute the cells sequentially.

---

## Key Visualizations Generated
The execution generates several diagnostic and exploratory plots:
* **Feature Histograms:** Showing the distribution of individual banknote features.
* **Correlation Heatmap:** Visualizing the Pearson correlation coefficients between features and targets.
* **Scatter Plot:** Plotting Variance vs Skewness colored by banknote class.
* **Feature Boxplots:** Visualizing summary statistics and outliers.
* **Training Error vs Epoch:** Shows the convergence behavior of the custom Perceptron (error goes to 0).
* **Weight & Bias Evolution:** Illustrating how parameters change during training.
* **Decision Boundary:** Visualizing the classification boundary using the first two features.
* **Confusion Matrix:** Showing the breakdown of true vs predicted values.
