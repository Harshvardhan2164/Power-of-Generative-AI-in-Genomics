# **Transformer-based Model for DNA Sequence Prediction**

## **Method Overview**
- **Model Type**: Transformer based deep learning model
- **Data**: Randomly generated nucleotide sequences & additional features
- **Goal**: Classify sequences into one of five classes
- **Evaluation**: Accuracy and loss metrics

## **Code Explanation**
### **Hyperparameters**
Defined at the beginning of the script:
- `VOCAB_SIZE = 5000`: Number of unique tokens in the dataset
- `SEQ_LEN = 1000`: Maximum sequence length
- `EMBED_DIM = 128`: Embedding dimension
- `LSTM_DIM = 64`: Transformer layer dimension
- `NUM_LAYERS = 2`: Number of LSTM layers
- `LR = 1e-3`: Learning rate for the optimizer

### **Dataset**
- **`X_train_sequences`**: Simulated DNA sequences as integer tokens
- **`X_train_other_features`**: Additional features (e.g., metadata)
- **`y_train_sequences`**: Labels (0-4) for classification

### **Evaluation**
The model is tested using a **test dataset**, where **perplexity** is calculated to measure prediction confidence.

Expected outputs:
1. Model summary
2. Training loss and accuracy per epoch
3. Final test accuracy

## **Results**
| Model         | Perplexity Score |
|--------------|--------------|
| LSTM (2 layers) | 3.70       |
| LSTM (4 layers) | 3.71       |
| LSTM (4 layers) | 3.68       |
| LSTM (5 layers) | 3.69       |

Lower perplexity scores indicate better model performance.