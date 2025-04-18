# **DNA Sequence Prediction using N-Gram, LSTM, and Transformer Models**

## **Table of Contents**
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [N-Gram Model](#n-gram-model)
  - [LSTM Model](#lstm-model)
  - [Transformer Model](#transformer-model)
- [Installation](#installation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## **Introduction**
DNA sequence prediction is a crucial task in bioinformatics, enabling researchers to analyze genetic patterns, predict mutations, and model gene structures. This project implements three machine learning approaches to predict nucleotide sequences: **N-Gram, LSTM, and Transformer models**.

## **Project Overview**
The goal of this project is to develop machine learning models that can:
1. Learn patterns in DNA sequences.
2. Predict missing or next nucleotides in a given sequence.
3. Evaluate the model's performance using **perplexity** as a key metric.

We explore the following methods:
- **N-Gram Model**: Uses statistical language modeling.
- **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in sequences.
- **Transformer Model**: Uses self-attention for sequence prediction.

## **Dataset**
We use nucleotide sequences of human genes from the **NCBI Gene Database**. The dataset consists of:
- Gene symbols, descriptions, and types.
- Nucleotide sequences represented as `A`, `T`, `C`, `G`.
- Train-validation split: **80% training, 20% testing**.

## **Model Architectures**

### **N-Gram Model**
- Uses **Maximum Likelihood Estimation (MLE)** for probability distribution.
- Converts DNA sequences into **N-grams (bigrams, trigrams, etc.)**.
- Evaluates prediction capability using **perplexity**.

### **LSTM Model**
- Deep learning model designed for sequential data.
- Captures long-term dependencies in DNA sequences.
- Uses **embedding layers, LSTM layers, and softmax activation**.

### **Transformer Model**
- Uses **self-attention mechanisms** to process sequences.
- More efficient than LSTM for long sequences.
- Implemented using **Positional Encoding, Multi-Head Attention, and Feed-Forward layers**.

## **Installation**
To set up the project, follow these steps:

### **1. Clone the Repository**
```bash
git clone https://github.com/Harshvardhan2164/Power-of-Generative-AI-in-Genomics.git
cd Power-of-Generative-AI-in-Genomics
```

### **2. Install Dependencies**
Ensure you have Python **3.8+** installed, then install required libraries:
```bash
pip install -r requirements.txt
```

## **Evaluation Metrics**
We use **perplexity** to evaluate model performance:
- Lower perplexity = **better model predictions**.
- N-gram models typically have higher perplexity than LSTMs and Transformers.

## **Results**
| Model         | Perplexity |
|--------------|-----------|
| N-Gram (n=3) | 3.8       |
| LSTM         | 2.9       |
| Transformer  | 2.5       |

Transformers perform best due to their ability to capture long-range dependencies.

## **Future Work**
- Implement **Bidirectional LSTMs** to improve accuracy.
- Use **pre-trained DNA embeddings**.
- Expand dataset to include more genetic variations.

## **Contributors**
- **Harshvardhan Sharma** ([GitHub](https://github.com/Harshvardhan2164))
- **Shantanu Gupta** ([GitHub](https://github.com/shantanugupta2004))
- **Avani Gajallewar** ([GitHub](https://github.com/avanig1834))

---

### **License**
This project is licensed under the MIT License.