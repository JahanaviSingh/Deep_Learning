 # Experiment 5: Text Generation Using RNN (Sequential Model)

## 1. Objective
The goal of this experiment is to implement and compare two different approaches for text generation using Recurrent Neural Networks (RNNs) on a dataset of 100 poems. We explore manual implementation using NumPy and deep learning implementation using PyTorch with One-Hot Encoding and Trainable Word Embeddings.

---

## 2. Implementation Overview

### Part 1: RNN From Scratch (NumPy)
Before using high-level libraries, we implemented the core logic of an RNN cell. 
- **Hidden State Calculation:** $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$
- **Purpose:** To understand the vanishing gradient problem and how the hidden state acts as memory.

### Part 2: One-Hot Encoding Approach
In this approach, words are represented as sparse vectors where only one index is '1'.
- **Preprocessing:** Tokenized text into words and mapped them to unique indices.
- **Architecture:** PyTorch `nn.RNN` with input size equal to the vocabulary size.
- **Challenge:** High memory consumption for large vocabularies.


### Part 3: Trainable Word Embeddings Approach
Instead of sparse vectors, we used an `nn.Embedding` layer to learn dense, continuous vector representations.
- **Preprocessing:** Words converted to integer indices.
- **Architecture:** Embedding layer followed by an RNN and a Linear layer.
- **Advantage:** Captures semantic relationships between words and reduces dimensionality.


---

## 3. Code Implementation

### Data Loading and Preprocessing
```python
# Used pandas to load '/Users/jahanavisingh/Downloads/poems-100.csv'
# Targeted the 'text' column for training.
# Vocabulary size was determined after tokenization.

Text Generation Quality

The models were tested with a seed word from the dataset to generate a 10-word sequence.

One-Hot Output: Often repeats frequent words but lacks semantic flow.

Embedding Output: Generally produces more coherent sequences as the model learns word contexts.

Advantages of Embeddings

Efficiency: Drastically reduces the number of parameters in the input layer.

Generalization: The model can predict better sequences by understanding that "sun" and "moon" might appear in similar contexts.

5. Conclusion
While One-Hot encoding is useful for understanding the basic mechanics of input layers, Trainable Word Embeddings are significantly more efficient and effective for text generation tasks. The RNN model successfully learned to predict the next word in a sequence based on the poetic structures provided in the dataset.
