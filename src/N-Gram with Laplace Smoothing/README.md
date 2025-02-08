# **N-Gram Language Model with Laplace Smoothing for DNA Sequence Prediction**

## **Method Overview**
- **Model Type**: N-Gram language model using Laplace smoothing
- **Data**: DNA sequences from the **NCBI Gene Database**
- **Goal**: Predict nucleotide sequences and evaluate perplexity scores
- **Evaluation**: Perplexity metrics

## **Code Explanation**
### **Training Process**
The training process involves:
- **Loading DNA sequences** from CSV files.
- **Converting sequences into N-grams**.
- **Training an N-Gram model with Laplace smoothing**.
- **Saving the trained model** for future use.

### **Evaluation**
The model is tested using a **test dataset**, where **perplexity** is calculated to measure prediction confidence.

### **Saving and Loading Models**
- Trained models are saved using **Pickle**.
- Models can be loaded for evaluation without retraining.

### **Visualization**
- Perplexity scores are plotted to show model performance for different N values.

## **Execution**
To train and evaluate the model, run:
```bash
python laplace_smoothing.py
```
Expected outputs:
1. Training log with time duration.
2. Perplexity scores for different N values.
3. Graph visualization of perplexity trends.

## **Results**
| N-Gram Order | Perplexity Score |
|-------------|----------------|
| 8-gram      | ~3.2           |
| 9-gram      | ~3.81           |
| 10-gram      | ~3.65           |

Lower perplexity scores indicate better model performance.