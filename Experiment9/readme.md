
---

# 📕 README — **Experiment 9: GAN & DCGAN**

```markdown
# Experiment 9: Generative Adversarial Networks (GANs)

## 📌 Objective
The objective of this experiment is to implement and compare Vanilla GAN and Deep Convolutional GAN (DCGAN) architectures for image generation and analyze their performance under different configurations.

---

## 📊 Dataset
- Dataset: Fashion-MNIST
- Image size: 28×28 grayscale

---

## ⚙️ Preprocessing
- Normalized images to [-1, 1]
- Optional augmentation: Horizontal flip

---

## 🧠 Models Implemented

### 1. Vanilla GAN
- Generator: Fully connected layers
- Discriminator: Fully connected layers

### 2. DCGAN
- Generator: Transposed convolution layers
- Discriminator: Convolutional layers
- Uses Batch Normalization and LeakyReLU

---

## ⚔️ GAN Architecture

### Generator
- Input: Random noise vector
- Output: Fake image

### Discriminator
- Input: Image (real/fake)
- Output: Probability

---

## 📉 Loss Functions
- Binary Cross Entropy (BCE)
- Least Squares GAN (LSGAN)
- Wasserstein Loss (WGAN)

---

## ⚙️ Optimizers Used
- SGD
- RMSprop
- Adam

---

## 🔁 Training Process
- Train discriminator on real and fake images
- Train generator to fool discriminator
- Alternate updates

---

## 📈 Metrics Tracked
- Generator Loss
- Discriminator Loss
- Generated images over epochs

---

## 🧪 Experiments Conducted
- Vanilla GAN vs DCGAN
- Loss comparison (BCE, LSGAN, WGAN)
- Optimizer comparison

---

## 📊 Results

### Vanilla GAN
- Unstable training
- Blurry images

### DCGAN
- Improved image quality
- Better feature learning

---

## ⚖️ Comparison

| Feature | GAN | DCGAN |
|--------|-----|-------|
| Architecture | Fully Connected | Convolutional |
| Image Quality | Low | High |
| Stability | Low | Better |
| Training | Difficult | Easier |

---

## 🧠 Observations
- DCGAN produces clearer and more realistic images
- Adam optimizer gives best performance
- WGAN improves stability and reduces mode collapse
- Training is sensitive to hyperparameters

---

## ⚠️ Challenges
- Mode Collapse
- Vanishing Gradients
- Oscillatory Training

---

## 📊 Experiment Tracking
- Tool: Weights & Biases (W&B)
- Logged:
  - Loss curves
  - Generated images

---

## 🚀 How to Run
```bash
pip install torch torchvision wandb
python train_gan.py
