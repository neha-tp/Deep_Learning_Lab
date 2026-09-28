# Experiment 7: Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders

This folder contains the complete end-to-end implementation and experimental evaluation of various Autoencoder architectures on the **MNIST Handwritten Digit Dataset**:
1. **Fully Connected Autoencoder (FC-AE)**: A multi-layer dense autoencoder to learn compressed latent representations and evaluate reconstruction quality.
2. **Convolutional Autoencoder (CAE)**: A convolutional encoder-decoder network utilizing 2D convolutions and upsampling/transposed convolutions to preserve spatial locality.
3. **Denoising Autoencoder (DAE)**: A convolutional autoencoder trained to reconstruct clean digits from corrupted inputs perturbed with additive Gaussian noise and salt-and-pepper noise.
4. **Variational Autoencoder (VAE)**: A probabilistic generative model utilizing the reparameterization trick, optimizing the Evidence Lower Bound (ELBO) with Reconstruction Loss and KL Divergence, exploring a 2D continuous latent manifold, and synthesizing novel digits from Gaussian priors.

---

## Dataset Information

The experiment utilizes the standard **MNIST Handwritten Digit Database**.

* **Source:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges ([MNIST Database](https://yann.lecun.com/exdb/mnist/) / `tf.keras.datasets.mnist`)
* **Number of Instances:** 60,000 training images, 10,000 test images (10,000 train / 2,000 test used for laboratory execution protocol)
* **Image Dimensions:** $28 \times 28$ grayscale pixels ($28 \times 28 \times 1$)
* **Pixel Value Range:** Normalized from $[0, 255]$ to $[0.0, 1.0]$
* **Classes (10 digits):** `0` through `9` (labels are held out during training and used strictly for latent space visualization and class-conditional analysis)

---

## Model Architectures & Implementations

### 1. Fully Connected Autoencoder (FC-AE)
* **Input:** Flattened image vector ($x \in \mathbb{R}^{784}$).
* **Encoder:** Dense(128, ReLU) $\to$ Dense(32, ReLU) $\to$ Dense(16, ReLU) (Latent bottleneck $d_z = 16$).
* **Decoder:** Dense(32, ReLU) $\to$ Dense(128, ReLU) $\to$ Dense(784, Sigmoid).
* **Loss Function:** Binary Cross-Entropy / Mean Squared Error.

### 2. Convolutional Autoencoder (CAE)
* **Input:** Grayscale image tensor ($28 \times 28 \times 1$).
* **Encoder:** Conv2D(32, $3 \times 3$, ReLU) $\to$ MaxPooling2D($2 \times 2$) $\to$ Conv2D(64, $3 \times 3$, ReLU) $\to$ MaxPooling2D($2 \times 2$) $\to$ Conv2D(64, $3 \times 3$, ReLU) $\to$ compressed feature map ($7 \times 7 \times 64$).
* **Decoder:** UpSampling2D($2 \times 2$) $\to$ Conv2D(32, $3 \times 3$, ReLU) $\to$ UpSampling2D($2 \times 2$) $\to$ Conv2D(1, $3 \times 3$, Sigmoid).

### 3. Denoising Convolutional Autoencoder (DAE)
* **Corruption Process:** Additive Gaussian noise $\tilde{x} = \text{clip}(x + n, 0, 1)$ where $n \sim \mathcal{N}(0, \sigma^2)$ with $\sigma \in \{0.1, 0.2, 0.3\}$, and salt-and-pepper noise.
* **Target:** Clean ground-truth image $x$, compelling the network to project off-manifold noisy states back onto the manifold of valid digits.

### 4. Variational Autoencoder (VAE)
* **Latent Space:** 2D probabilistic latent space ($z \in \mathbb{R}^2$) for direct visualization and manifold navigation.
* **Encoder:** Convolutional feature extraction flattening to two parallel dense heads outputting $\mu$ (mean vector) and $\log \sigma^2$ (log-variance).
* **Reparameterization Trick:** Differentiable stochastic sampling:
  $$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
* **Decoder:** Dense projection $\to$ Reshape $\to$ Conv2DTranspose / UpSampling layers reconstructing $28 \times 28 \times 1$ image probabilities.
* **Objective:** Maximizing the Evidence Lower Bound (ELBO):
  $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{KL}}$$
  $$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## Dependency List

To run the notebook, the following dependencies are required:

* **Python 3.x**
* **TensorFlow / Keras** (`tensorflow`) - For building, training, and evaluating deep learning models.
* **NumPy** (`numpy`) - For high-performance numerical routines and multidimensional array operations.
* **Matplotlib** (`matplotlib`) - For image grid rendering, latent space plotting, and metric visualizations.
* **Scikit-Image** (`scikit-image`) - For calculating structural image similarity metrics (`structural_similarity` / SSIM).
* **Scikit-Learn** (`scikit-learn`) - For dataset utilities and metric calculations.
* **SciPy** (`scipy`) - For statistical distributions and sampling.
* **Jupyter Notebook / Google Colab** - For interactive cell-by-cell execution.

---

## Execution Instructions

You can run the notebook either **locally** or in **Google Colab**.

### Option 1: Running Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/neha-tp/Deep_Learning_Lab.git
   cd Deep_Learning_Lab/Ex7
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
   pip install tensorflow numpy matplotlib scikit-image scikit-learn scipy jupyter
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `ex7.ipynb` and run all cells sequentially.

---

### Option 2: Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `ex7.ipynb` notebook from this folder.
3. Enable GPU acceleration:
   - Navigate to **Runtime** > **Change runtime type**.
   - Select **T4 GPU** (or standard GPU) as the Hardware accelerator.
4. Execute the cells sequentially. The dataset (`mnist`) is automatically downloaded via `keras.datasets`.

---

## Key Visualizations Generated

The notebook execution produces several diagnostic, quantitative, and qualitative visualizations:
* **Original vs. Reconstructed Images (FC-AE):** Visual grid comparing ground truth test digits against their 16-dimensional latent reconstructions.
* **Training and Validation Loss Curves:** Showing convergence behavior of the FC-AE without overfitting.
* **FC-AE vs. Convolutional AE Reconstructions:** Side-by-side visual comparison demonstrating sharper edges and superior preservation of stroke topology with convolutional layers.
* **Clean vs. Noisy vs. Denoised Visualizations:** Evaluating DAE performance across increasing noise standard deviations ($\sigma \in \{0.1, 0.2, 0.3\}$).
* **Noise Level vs. Reconstruction Metrics:** Plots of MSE, MAE, and SSIM showing metric degradation curves under increasing noise levels.
* **VAE 2D Latent Space Manifold:** Scatter plot of test digits projected onto $(z_1, z_2)$ colored by digit class, demonstrating learned continuous clustering.
* **VAE Generative Sampling:** Visual grids of newly synthesized digits generated by decoding random vectors drawn from the prior $z \sim \mathcal{N}(0, I)$.
* **Latent Space Interpolation:** Decoded image sequences along linear interpolation paths between distinct digit classes ($1 \to 7$ and $3 \to 8$), showing continuous morphological transitions.
* **VAE Loss Convergence Profiles:** Tracking reconstruction loss and KL-divergence regularization across epochs.
* **Reconstruction Error Distribution & Outlier Analysis:** Histogram of per-sample reconstruction errors on the test set, alongside inspection of the top-5 highest-error samples.
* **Latent Dimension Ablation Study:** Plot of reconstruction MSE and SSIM across bottleneck sizes ($d_z \in \{2, 8, 16, 32\}$).
