# Experiment 8: Autoencoders and Variational Autoencoders (VAE)

## 📌 Objective
The objective of this experiment is to implement Autoencoders (AE) and Variational Autoencoders (VAE) using PyTorch to learn latent representations of data and compare their reconstruction and generative capabilities.

---

## 📊 Dataset
- Dataset: Fashion-MNIST
- Image size: 28×28 grayscale
- Classes: Clothing categories (shirts, shoes, bags, etc.)

### Data Split
- 80% Training
- 10% Validation
- 10% Testing

---

## ⚙️ Preprocessing
- Normalized pixel values to [0,1]
- Flattened images for fully connected models

---

## 🧠 Models Implemented

### 1. Autoencoder (AE)
- Encoder: Fully connected layers → latent vector
- Decoder: Fully connected layers → reconstructed image
- Deterministic model

### 2. Variational Autoencoder (VAE)
- Learns mean (μ) and variance (σ)
- Uses reparameterization trick:
  
  z = μ + σ · ε

- Probabilistic generative model

---

## 📉 Loss Functions
- Binary Cross Entropy (BCE)
- Mean Squared Error (MSE)
- VAE includes:
  - Reconstruction Loss
  - KL Divergence

---

## ⚙️ Optimizers Used
- SGD
- RMSprop
- Adam

---

## 🧪 Experiments Conducted
- Latent dimensions: 2, 8, 16, 32
- Loss comparison: BCE vs MSE
- Optimizer comparison
- Autoencoder vs VAE comparison

---

## 🔍 Latent Space Analysis
- Interpolation between latent vectors:
  
  z = (1 − α)z₁ + αz₂

- Observed smooth transitions in VAE
- 2D latent space visualization (for interpretability)

---

## 📈 Results

### Autoencoder
- Better reconstruction quality
- Sharp output images
- No generative capability

### Variational Autoencoder
- Slightly blurry reconstructions
- Smooth latent space
- Capable of generating new samples

---

## ⚖️ Comparison

| Feature | Autoencoder | VAE |
|--------|------------|-----|
| Type | Deterministic | Probabilistic |
| Reconstruction | Sharp | Slightly Blurry |
| Generation | ❌ | ✅ |
| Latent Space | Unstructured | Smooth & Continuous |

---

## 🧠 Observations
- Increasing latent dimension improves reconstruction
- BCE produces sharper outputs than MSE
- Adam optimizer converges faster
- KL divergence regularizes latent space

---

## 📊 Experiment Tracking
- Tool: Weights & Biases (W&B)
- Metrics tracked:
  - Training loss
  - Validation loss
  - Reconstruction error

---

## 🚀 How to Run
```bash
pip install torch torchvision wandb
python train.py
