import numpy as np
import sys
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.lm import MLE
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import gc
import pickle
import json
from functools import partial
from logger import logging
from exception import CustomException

def main():
    try:
        for i in range(6):
            random.seed(638)
            largest_value, WILL_TRAIN_FROM_SCRATCH  = i+1, True
            if WILL_TRAIN_FROM_SCRATCH:
                lm = get_comprehensive_lm(largest_value)
            else:
                lm = read_lm(largest_value)
            n_candidates, perp_values = list(range(largest_value, largest_value + 1)), []
            for n in n_candidates:
                print(n, ' Started.')
                perp = get_performance_for_n_gram(n, lm)
                gc.collect()
                perp_values.append(perp)
                print(n, ' is completed. Perplexity value: ', perp)
            write_results(n_candidates, perp_values, largest_value)
            plot_perp_values(perp_values, n_candidates, largest_value)
            
            logging.info("Model Prediction Completed")
    except Exception as e:
        raise CustomException(e, sys)

def read_lm(n):
    print('Language Model is being read.')
    if n >= 1:
        f = open('./lang_model_s_' + str(n) + '.obj', 'rb')
    else:
        raise Exception('N Value Given for N-gram is not defined.')
    lm = pickle.load(f)
    print('Language Model was read.')
    return lm

def get_train_text():
    print('Train Data is being prepared.')
    text_train = get_formatted_data('../cleaned_dataset/train.csv')
    print('Train Data is Ready.')

    return text_train

def get_test_text():
    print('Test Data is Being Prepared.')
    text_test = get_formatted_data('../cleaned_dataset/test.csv')
    print('Test Data is Ready.')

    return text_test

def get_formatted_data(path):
    dataframe = pd.read_csv(path)
    gene_nucleotide_sequences = dataframe['NucleotideSequence']
    gene_nucleotide_sequences_list = gene_nucleotide_sequences.tolist()
    formatted_dataset = []
    for i in range(len(gene_nucleotide_sequences_list)):
        formatted_str = format_str(gene_nucleotide_sequences_list[i])
        formatted_dataset.append(formatted_str)

    del dataframe
    del gene_nucleotide_sequences
    del gene_nucleotide_sequences_list
    del formatted_str
    gc.collect()

    return formatted_dataset

def format_str(gene_nucleotide_sequence):
    chars_list = [*gene_nucleotide_sequence[1:-1]]
    return chars_list

def get_comprehensive_lm(n):
    text_train = get_train_text()
    gc.collect()
    start_time = time.time()
    lm = get_language_model(text_train, n)
    print("--- %s seconds ---" % (time.time() - start_time))
    del text_train
    gc.collect()

    return lm

def get_performance_for_n_gram(n, lm):
    text_test = get_test_text()
    gc.collect()
    start_time = time.time()
    result = evaluate_lang_model(lm, text_test, n)
    print("--- %s seconds ---" % (time.time() - start_time))
    del text_test
    del lm
    gc.collect()

    return result

def get_language_model(X_train, n):
    padding_fn = partial(pad_both_ends, n=n)
    vocab = flatten(map(padding_fn, X_train))
    X_train_formatted = (ngrams(list(padding_fn(sent)), n) for sent in X_train)
    lm = MLE(n)
    print('Data is formatted.')
    lm.fit(X_train_formatted, vocab)
    del padding_fn
    del vocab
    del X_train_formatted
    gc.collect()
    print('Model Was Fitted.')
    save_lm(lm, n)

    return lm

def save_lm(lm, n):
    print('Language Model is being Saved.')
    f = open('lang_model_s_' + str(n) + '.obj', 'wb')
    pickle.dump(lm, f)
    print('Language Model was Saved.')

def evaluate_lang_model(lm, X_test, n):
    ranges, subresults = find_ranges(len(X_test), n_parts=50), []
    print(ranges)
    for k in range(len(ranges)):
        X_test_preprocessed = []
        for i in range(ranges[k][0], min(ranges[k][1], len(X_test))):
            X_test_preprocessed += preprocess(X_test[i], n)
        print('Formatting Completed. Perplexity Calculation Is Starting.', k, '/', len(ranges))
        perp = lm.perplexity(X_test_preprocessed)
        print('Perplexity Was Calculated.', k, '/', len(ranges))
        subresults.append((len(X_test_preprocessed), perp))
        print('N and P Values:', (len(X_test_preprocessed), perp))
        del X_test_preprocessed
        gc.collect()
    gc.collect()
    res = calculate_actual_perp(subresults)
    return res

def find_ranges(length_val, n_parts=10):
    part_size = (length_val // n_parts) + 1
    ranges_list = []
    count = 0
    while count < length_val:
        ranges_list.append((count, count+part_size))
        count += part_size
    return ranges_list

def calculate_actual_perp(subresults):
    n_vals, p_vals = [], []
    for i in range(len(subresults)):
        n_vals.append(subresults[i][0])
        p_vals.append(subresults[i][1])
    n_vals, p_vals = np.array(n_vals), np.array(p_vals)
    n_vals_sum = np.sum(n_vals)
    p_log_vals = np.log(p_vals)
    dividend = np.sum(n_vals * p_log_vals)
    return np.exp(dividend / n_vals_sum)

def preprocess(X_test, n):
    temp = pad_both_ends(X_test, n=n)
    res = list(ngrams(temp, n=n))
    if n > 2:
        res = res[:-(n-2)]
    return res

def write_results(n_candidates, perp_values, largest_value):
    jsonString1 = json.dumps(n_candidates)
    jsonFile1 = open("n_candidates_s_" + str(largest_value) + ".json", "w")
    jsonFile1.write(jsonString1)
    jsonFile1.close()

    jsonString2 = json.dumps(perp_values)
    jsonFile2 = open("perp_values_s_" + str(largest_value) + ".json", "w")
    jsonFile2.write(jsonString2)
    jsonFile2.close()

def plot_perp_values(perp_values, n_candidates, largest_value):
    INF_VALUE = 5
    print(perp_values)
    plt.xlabel('N')
    plt.ylabel('Perplexity')
    plt.ylim((0,5))
    plt.xticks(list(range(1, largest_value+1)))
    plt.axhline(y=1, color='r', linestyle='--', label = "Theoretical Lower Bound")
    plt.axhline(y=4, color='b', linestyle='--', label = "Theoretical Upper Bound")
    plt.plot(n_candidates, np.minimum(np.array(perp_values), INF_VALUE), c='g', label='Perplexity')
    plt.legend(loc='center right')
    plt.title('Perplexity for Different N Values')
    plt.savefig('./perp_values_s_' + str(largest_value))

    plt.show()

if __name__ == '__main__':
    main()