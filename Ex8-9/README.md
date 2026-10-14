# Experiments 8 & 9: Effect of Attention Type and Attention Position in a Pretrained ResNet50

This folder contains the complete implementation, experimental evaluation, and qualitative analysis for **Experiments 8 and 9** of the Deep Learning Laboratory. 

The investigation explores how different **attention mechanisms** modify feature representations in a pretrained convolutional neural network (**ResNet50** pretrained on ImageNet) and evaluates how the **stage-wise position of attention insertion** influences downstream image classification performance on fine-grained flower categories.

---

## Overview & Objectives

Attention mechanisms enable neural networks to dynamically prioritize task-relevant features while suppressing background clutter. In this study, we systematically analyze:
1. **Attention Types**: Channel Attention (SE), Spatial Attention, Sequential Channel-Spatial Attention (CBAM), Lightweight Self-Attention, Efficient Channel Attention (ECA), and Coordinate Attention.
2. **Attention Positions**: Inserting attention modules across ResNet50 feature extraction stages:
   * **P1 (`conv2_block3_out`)**: Early-stage features ($56 \times 56 \times 256$) — edges, colors, local textures.
   * **P2 (`conv3_block4_out`)**: Mid-level features ($28 \times 28 \times 512$) — corner structures, motifs, contours.
   * **P3 (`conv4_block6_out`)**: High-level semantic features ($14 \times 14 \times 1024$) — object parts and discriminative regions (*default candidate position*).
   * **P4 (`conv5_block3_out`)**: Deepest abstract representations ($7 \times 7 \times 2048$) — compact global class concepts.
3. **Multi-Stage & Ordering Configurations**: Multi-stage insertion (e.g., P2 + P3, all stages) and ordering variations (Channel $\rightarrow$ Spatial vs Spatial $\rightarrow$ Channel).
4. **Interpretability & Explainability**: Comparing model explanations via Grad-CAM heatmaps and analyzing internal attention maps across mechanisms.

---

## Dataset Information

The experiments are conducted on the **TensorFlow Flowers (`tf_flowers`)** dataset.

* **Source:** [TensorFlow Datasets - tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers)
* **Total Images:** 3,670 natural RGB images
* **Classes (5):** `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`
* **Dataset Splits:**
  * **Training Set:** 70% (~2,569 images)
  * **Validation Set:** 15% (~550 images)
  * **Test Set:** 15% (~551 images)
* **Preprocessing & Augmentation:**
  * Input dimensions resized to $224 \times 224 \times 3$.
  * Normalized with ResNet50 ImageNet preprocessing (`tf.keras.applications.resnet50.preprocess_input`).
  * Augmentation pipeline: Random horizontal flips, small rotations ($\pm 15^\circ$), random zoom, and mild contrast adjustments.

---

## Attention Mechanisms Implemented

1. **Channel Attention (Squeeze-and-Excitation / SE)**:
   * Aggregates spatial dimensions using Global Average Pooling and Global Max Pooling.
   * Shared multi-layer perceptron (reduction ratio $r = 16$) followed by a Sigmoid activation.
   * Produces a 1D channel descriptor $M_c \in \mathbb{R}^{1 \times 1 \times C}$ to recalibrate channel importances.

2. **Spatial Attention**:
   * Computes channel-wise average and max projections across feature slices.
   * Concatenates pooled feature maps ($2 \times H \times W$) and applies a convolutional layer (default $7 \times 7$ kernel) with Sigmoid activation.
   * Produces a 2D spatial mask $M_s \in \mathbb{R}^{H \times W \times 1}$ emphasizing discriminative object regions.

3. **Sequential Channel-Spatial Attention (CBAM)**:
   * Serially applies Channel Attention followed by Spatial Attention:
     $$F' = M_c(F) \otimes F, \quad F'' = M_s(F') \otimes F'$$

4. **Lightweight Self-Attention (Non-Local / Dot-Product Attention)**:
   * Maps features into Query ($Q$), Key ($K$), and Value ($V$) representations using $1 \times 1$ convolutions.
   * Computes spatial affinity matrix $\text{Softmax}(Q K^T / \sqrt{d})$ capturing long-range contextual dependencies.
   * Modulates $V$ and adds a learnable residual shortcut.

5. **Additional Attention Mechanisms**:
   * **ECA (Efficient Channel Attention)**: Captures local cross-channel interaction without dimensionality reduction via adaptive 1D convolution.
   * **Coordinate Attention**: Embeds positional information along horizontal and vertical spatial directions independently to preserve coordinate sensitivity.

---

## Experimental Ablation Studies

* **Ablation 1 (Attention Type)**: Compares Baseline ResNet50 against Channel, Spatial, Sequential CBAM, and Self-Attention positioned at stage P3.
* **Ablation 2 (Attention Position)**: Examines spatial and channel attention across stages P1, P2, P3, and P4 to evaluate where attention provides maximal utility.
* **Ablation 3 (Multi-Stage Attention)**: Evaluates single-stage vs combined multi-stage insertion.
* **Ablation 4 (Attention Ordering)**: Compares Channel $\rightarrow$ Spatial (standard CBAM) vs Spatial $\rightarrow$ Channel.
* **Ablation 5 (Kernel Size)**: Compares $3 \times 3$ vs $7 \times 7$ spatial attention receptive fields.
* **Ablation 6 (Advanced Mechanisms)**: Compares CBAM against ECA and Coordinate Attention in terms of parameter efficiency, inference latency, and accuracy.

---

## Key Visualizations & Plots (`plots/`)

All generated evaluation curves, diagnostic plots, and interpretability figures are saved in the `plots/` directory:

| Filename | Description |
| :--- | :--- |
| `Sample_Images_from_All_Classes.png` | Exemplar samples illustrating within-class diversity and background clutter across all 5 flower classes. |
| `Training_and_Validation_Accuracy.png` | Epoch-wise training and validation accuracy trajectories across configurations. |
| `Training_and_Validation_Loss.png` | Categorical cross-entropy loss convergence across training epochs. |
| `Attention_Mechanism_vs_Classification_Accuracy.png` | Bar chart comparison of test accuracy achieved by each attention variant. |
| `Attention_Mechanism_vs_Macro_F1-score.png` | Macro F1-score performance across all attention architectures. |
| `Confusion_Matrices.png` | Multi-panel confusion matrices breaking down true vs predicted class distributions. |
| `Channel_Attention_Weights_(Top_20).png` | Activation magnitudes of the top 20 most influential feature channels. |
| `Spatial_Attention_at_P3.png` | Internal 2D spatial attention activation heatmaps extracted at stage P3. |
| `Spatial_Attention_at_Different_Positions.png` | Comparison of spatial attention focus when placed at P1, P2, P3, and P4. |
| `Self-Attention_Response_at_P3.png` | Spatial query-key affinity maps generated by the self-attention layer. |
| `Feature_Maps_Before_and_After_Attention.png` | Qualitative demonstration of feature map refinement after attention modulation. |
| `Grad-CAM_Baseline_vs_Attention.png` | Post-hoc Grad-CAM saliency maps comparing Baseline ResNet50 vs Attention-augmented models. |
| `Grad-CAM_Comparison_of_Additional_Attention_Mechanisms.png` | Grad-CAM attribution comparison across CBAM, ECA, and Coordinate Attention. |
| `Qualitative_Comparison_Across_Attention_Mechanisms.png` | Comprehensive multi-model visual comparison across challenging test instances. |
| `Test_Accuracy_vs_Total_Model_Parameters.png` | Parameter efficiency scatter plot showing accuracy gained per added parameter. |
| `Macro_F1-score_vs_Inference_Time_per_Image.png` | Accuracy vs inference latency trade-off curve across all model variants. |

---

## Dependency List

To execute the notebook and reproduce all experiments, install the following dependencies:

* **Python 3.9+**
* **TensorFlow** (`tensorflow >= 2.10`)
* **Keras** (`keras`)
* **TensorFlow Datasets** (`tensorflow-datasets`)
* **NumPy** (`numpy`)
* **Pandas** (`pandas`)
* **Matplotlib** (`matplotlib`)
* **Seaborn** (`seaborn`)
* **Scikit-Learn** (`scikit-learn`)
* **OpenCV** (`opencv-python`)
* **Jupyter Notebook / JupyterLab / Google Colab**

---

## Execution Instructions

You can run the experiment either **locally** or on **Google Colab**.

### Option 1: Running Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/neha-tp/Deep_Learning_Lab.git
   cd Deep_Learning_Lab/Ex8-9
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
   pip install tensorflow tensorflow-datasets keras numpy pandas matplotlib seaborn scikit-learn opencv-python jupyter
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `Ex8-9.ipynb` and execute the cells in sequence. The dataset will download automatically via `tensorflow_datasets` on first run.

---

### Option 2: Running in Google Colab (Recommended for GPU Acceleration)

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `Ex8-9.ipynb` from this folder.
3. Change the runtime type to **GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Execute the cells sequentially. Outputs and diagnostic plots will be generated and saved to the runtime environment.
