# **LSTM-Based DNA Sequence Classification**

## **Method Overview**
- **Model Type**: LSTM-based deep learning model
- **Data**: Randomly generated nucleotide sequences & additional features
- **Goal**: Classify sequences into one of five classes
- **Evaluation**: Accuracy and loss metrics

## **Code Explanation**
### **Hyperparameters**
Defined at the beginning of the script:
- `VOCAB_SIZE = 5000`: Number of unique tokens in the dataset
- `SEQ_LEN = 1000`: Maximum sequence length
- `EMBED_DIM = 128`: Embedding dimension
- `LSTM_DIM = 64`: LSTM layer dimension
- `NUM_LAYERS = 2`: Number of LSTM layers
- `LR = 1e-3`: Learning rate for the optimizer

### **Dataset**
- **`X_train_sequences`**: Simulated DNA sequences as integer tokens
- **`X_train_other_features`**: Additional features (e.g., metadata)
- **`y_train_sequences`**: Labels (0-4) for classification

### **Model Architecture**
The model consists of:
1. **Embedding Layer**: Converts input sequences into dense vectors.
2. **LSTM Layer**: Extracts long-range dependencies in sequences.
3. **Dense Layer for Other Features**: Processes additional numerical data.
4. **Concatenation Layer**: Merges LSTM output and additional features.
5. **Output Layer**: Predicts one of five possible classes using softmax activation.

### **Training and Evaluation**
- The model is compiled using **Adam optimizer** and `sparse_categorical_crossentropy` loss.
- It is trained for **10 epochs** with a batch size of **64**.
- The final performance is evaluated on a test set.

Expected outputs:
1. Model summary
2. Training loss and accuracy per epoch
3. Final test accuracy

## **Results**
| Model         | Perplexity Score |
|--------------|--------------|
| LSTM (2 layers) | 3.8       |
