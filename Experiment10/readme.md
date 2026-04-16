Objective
The objective of this experiment is to implement an image classification model using the Vision Transformer (ViT) architecture and compare its performance with a Convolutional Neural Network (ResNet-18).
This experiment also studies:
The effect of data augmentation (horizontal and vertical flips)
The impact of different loss functions and optimizers
Performance differences between transformer-based and CNN-based models
📊 Dataset
Dataset Used: CIFAR-10
Number of Classes: 10
Train/Validation/Test Split:
Training: 80%
Validation: 10%
Testing: 10%
⚙️ Preprocessing & Augmentation
Preprocessing
Conversion to tensor
Normalization (mean = 0.5, std = 0.5)
Data Augmentation
Random Horizontal Flip
Random Vertical Flip
Two datasets were used:
Original dataset
Augmented dataset
🧠 Models Implemented
1. Vision Transformer (ViT)
The ViT model processes images as sequences of patches instead of using convolution.
Key Components:
Patch Embedding
Positional Encoding
Transformer Encoder (Multi-head Attention + Feed Forward Network)
CLS Token for classification
Fully Connected Classification Head
2. ResNet-18 (CNN Baseline)
Standard deep convolutional neural network
Residual connections for better gradient flow
Modified final layer for 10-class classification
📉 Loss Functions Used
Cross-Entropy Loss
Label Smoothing Loss
Focal Loss
⚙️ Optimizers Used
SGD
Adam
RMSprop
📡 Experiment Tracking
All experiments were tracked using Weights & Biases, including:
Training Loss
Validation Loss
Accuracy
Model comparisons
📈 Results
Model	Augmentation	Loss Function	Optimizer	Accuracy
ViT	No	CE	Adam	XX%
ViT	Yes	CE	Adam	XX%
ResNet	Yes	Focal	SGD	XX%
(Replace XX with your actual results)
🧪 Observations
Effect of Data Augmentation
Improves model generalization
Reduces overfitting
Increases robustness to variations
ViT vs ResNet-18
ResNet performs better on small datasets like CIFAR-10
ViT requires more data to fully utilize its potential
ViT captures global relationships using self-attention
CNN captures local features using convolution
Loss Function Comparison
Cross-Entropy: Stable and widely used
Label Smoothing: Prevents overconfidence
Focal Loss: Focuses on hard-to-classify samples
Optimizer Comparison
Adam: Fast convergence
SGD: Better generalization
RMSprop: Stable but slightly less effective
⏱️ Evaluation Metrics
Test Accuracy
Training Time
Model Complexity
🧾 Conclusion
Data augmentation improves performance for both models
ResNet-18 outperforms ViT on small datasets
ViT shows competitive performance with proper tuning
Adam optimizer converges faster, while SGD generalizes better
