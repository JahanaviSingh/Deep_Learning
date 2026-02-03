Experiment4/
│
├── data/                  # Dataset storage
│   ├── cats_dogs/         # Extracted Kaggle dataset
│   └── cifar-10-batches-py/
│
├── weights/               # Saved model weights (.pth files)
│   ├── CIFAR10/
│   └── CatsDogs/
│
├── cnn_implementation.ipynb  # Main code for training and evaluation
├── requirements.txt       # Dependencies
└── README.md              # This documentation

Experiments Conducted
We explored various configurations by permuting the following hyperparameters:

1. Activation Functions

ReLU (Rectified Linear Unit)

Tanh (Hyperbolic Tangent)

Leaky ReLU

2. Weight Initialization Techniques

Xavier Initialization (Glorot)

Kaiming Initialization (He)

Random Normal Initialization

3. Optimizers

SGD (Stochastic Gradient Descent)

Adam (Adaptive Moment Estimation)

RMSprop

⚙️ Model Architecture
The custom CNN consists of:

3 Convolutional Blocks: Each containing Conv2d → BatchNorm → Activation → MaxPool.

Classification Head: Flatten → Dropout (0.5) → Linear (FC) → Output.

🚀 How to Run
1. Prerequisites

Install the required libraries:

Bash
pip install torch torchvision numpy matplotlib scikit-learn
2. Dataset Setup

CIFAR-10: The code will automatically download this.

Cats vs Dogs:

Download from Kaggle.

Extract the train.zip file.

Organize the images into subfolders so the path looks like:

Experiment4/data/cats_dogs/cat/

Experiment4/data/cats_dogs/dog/

3. Execution

Open the Jupyter Notebook and run all cells:

Bash
jupyter notebook cnn_implementation.ipynb




