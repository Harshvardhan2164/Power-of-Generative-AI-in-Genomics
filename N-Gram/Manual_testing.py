from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends
import pickle

def load_model(n):
    with open(f'lang_model_s_{n}.obj', 'rb') as f:
        lm = pickle.load(f)
    print(f"Loaded N-gram model with n={n}")
    return lm

def preprocess_sequence(sequence, n):
    padded_seq = list(pad_both_ends(sequence, n=n))  # Add padding
    ngram_seq = list(ngrams(padded_seq, n=n))  # Generate N-grams
    return ngram_seq

def calculate_perplexity(lm, sequence, n):
    ngram_sequence = preprocess_sequence(sequence, n)
    perplexity = lm.perplexity(ngram_sequence)
    return perplexity

def generate_next_nucleotide(lm, context, n):
    context = tuple(preprocess_sequence(context, n)[4])  # Get the last N-gram
    next_nucleotide = lm.generate(1, text_seed=context)
    return next_nucleotide

n = 5
lm = load_model(n)

# Example: Test sequence
test_sequence = "ATCGGTA"
n = 5  # Use the same n as the trained model
processed_sequence = preprocess_sequence(test_sequence, n)

print("Processed N-grams:", processed_sequence)

# Example: Evaluate test sequence
perplexity = calculate_perplexity(lm, test_sequence, n)
print(f"Perplexity of the sequence: {perplexity}")

# Example: Predict next nucleotide after "ATCG"
context = "ATCG"
predicted_nucleotide = generate_next_nucleotide(lm, context, n)
print(f"Predicted next nucleotide: {predicted_nucleotide}")