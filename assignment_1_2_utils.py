import numpy as np
from nltk.util import ngrams
import sys
import math

from assignment_utils import split, set_log_file, log, tabulate_list, count_words, count_ngrams, BiGram, TriGram, bigram_from, trigram_from

# unique_words_d_cz = {w for w in tr_d_cz}
# unique_words_count_d_cz = len(unique_words)
#print(f"Number of unique words in train cz {unique_words_count_d_cz}" )
def compute_p(dataset, unigram_data, bigram_data, trigram_data):
    # compute counts of unigrams bi-grams trigrams
    unigram_dict = count_words(unigram_data)

    bigram_dict = count_ngrams(bigram_data)

    trigram_dict = count_ngrams(trigram_data)

    # compute probabilities of unigrams bi-grams trigrams
    unigram_probs = dict()
    for uni in unigram_dict:
        unigram_probs[uni] = unigram_dict[uni] / len(dataset)
    bigram_probs = dict()

    for bigram in bigram_dict:
        bigram_probs[bigram] = bigram_dict[bigram] / unigram_dict[bigram.left]

    trigram_probs = dict()
    for trigram in trigram_dict:
        bigram = bigram_from(left=trigram.first, right=trigram.second)
        trigram_probs[trigram] = trigram_dict[trigram] / bigram_dict[bigram]
    return unigram_probs, bigram_probs, trigram_probs

def uniform_probability(vocabulary_size):
    return 1/vocabulary_size

def unigram_probability(trigram, unigram_dataset, word_counts):
    word = trigram.third
    vocabulary_size = len(word_counts.keys())
    if word not in word_counts:
        return uniform_probability(vocabulary_size)
    return word_counts[word] / len(unigram_dataset)

def bigram_probability(trigram, bigram_counts, word_counts):
    bigram = BiGram(trigram.second, trigram.third)
    vocabulary_size = len(word_counts.keys())
    if bigram not in bigram_counts:
        return uniform_probability(vocabulary_size)
    return bigram_counts[bigram] / word_counts[bigram.left]

def trigram_probability(trigram, trigram_counts, bigram_counts, word_counts):
    vocabulary_size = len(word_counts.keys())
    if trigram not in trigram_counts:
        return uniform_probability(vocabulary_size)
    return trigram_counts[trigram] / bigram_counts[BiGram(trigram.first, trigram.second)]

def smoothed_probability(trigram, l0, l1, l2, l3, unigram_dataset, trigram_counts, bigram_counts, word_counts):
    vocabulary_size = len(word_counts.keys())
    return  \
      l0*uniform_probability(vocabulary_size) \
    + l1*unigram_probability(trigram, unigram_dataset, word_counts) \
    + l2*bigram_probability(trigram, bigram_counts, word_counts) \
    + l3*trigram_probability(trigram, trigram_counts, bigram_counts, word_counts)

def equals_with(a, b, precision):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for pair in zip(a,b):
            if not math.isclose(pair[0], pair[1], abs_tol=precision):
                return False
        return True
    return math.isclose(a, b, abs_tol=precision)

# calculate smoothing params
def smothing_param(trigram_heldout_data, unigram_train_data, bigram_train_data, trigram_train_data):

    word_counts = count_words(unigram_train_data)
    vocabulary_size = len(word_counts.keys())

    bigram_counts = count_ngrams(bigram_train_data)

    trigram_counts = count_ngrams(trigram_train_data)

    l0 = 0.25
    l1 = 0.25
    l2 = 0.25
    l3 = 0.25
    c_l0 = 0
    c_l1 = 0
    c_l2 = 0
    c_l3 = 0
    e = 0.0001
    while (True):
        for trigram in trigram_heldout_data:

            smoothed = smoothed_probability(trigram, l0, l1, l2, l3, unigram_train_data, trigram_counts, bigram_counts,  word_counts)

            c_l0 += l0 * uniform_probability(vocabulary_size) / smoothed
            c_l1 += l1 * unigram_probability(trigram, unigram_train_data, word_counts) / smoothed
            c_l2 += l2 * bigram_probability(trigram, bigram_counts, word_counts) / smoothed
            c_l3 += l3 * trigram_probability(trigram, trigram_counts, bigram_counts, word_counts) / smoothed

        l0_new = c_l0 / (c_l0 + c_l1 + c_l2 + c_l3)
        l1_new = c_l1 / (c_l0 + c_l1 + c_l2 + c_l3)
        l2_new = c_l2 / (c_l0 + c_l1 + c_l2 + c_l3)
        l3_new = c_l3 / (c_l0 + c_l1 + c_l2 + c_l3)

        if equals_with([l0_new,l1_new,l2_new,l3_new],[l0,l1,l2,l3], e):
            break

        l0 = l0_new
        l1 = l1_new
        l2 = l2_new
        l3 = l3_new
  
    return [l0, l1, l2, l3]

def cross_entropy(l0, l1, l2, l3, trigram_test_data, dataset, trigram_train_data, bigram_train_data, unigram_train_data): 
    word_counts = count_words(unigram_train_data)

    bigram_counts = count_ngrams(bigram_train_data)

    trigram_counts = count_ngrams(trigram_train_data)

    H = 0
    for trigram in trigram_test_data:
        prob = smoothed_probability(trigram, l0, l1, l2, l3, unigram_dataset=unigram_train_data, trigram_counts=trigram_counts, bigram_counts=bigram_counts, word_counts=word_counts)
        assert prob >= 0
        H += np.log2(prob)
    H = -H
    return H/len(dataset)

def compute_entropy(unigram_data, bigram_data, trigram_data, unigram_probs1,
                   bigram_probs1, trigram_probs1, l0, l1,l2,l3):
    entropy = 0
    for i in range(len(unigram_data)):
        word1 = unigram_probs1[unigram_data[i]]
        if i + 2 == len(unigram_data) or i + 1 == len(unigram_data):
            break

        word2 = bigram_probs1[bigram_data[i]]
        word3 = trigram_probs1[trigram_data[i]]

        p = l1 * word1 + l2 * word2 + l3 * word3 + l0 * 1 / len(h_d_en)
        entropy = entropy + (p/np.log(p))
    return entropy


def shannon_entropy(unigram_data, bigram_data, trigram_data, unigram_probs1,
                   bigram_probs1, trigram_probs1, l0, l1,l2,l3, heldout_len):
    smoothed_prods = dict()
    for i in range(len(unigram_data)):
            word1 = unigram_probs1[unigram_data[i]]
            if i + 2 == len(unigram_data) or i + 1 == len(unigram_data):
                break

            word2 = bigram_probs1[bigram_data[i]]
            word3 = trigram_probs1[trigram_data[i]]

            smoothed_prods[unigram_data[i]] = l1 * word1 + l2 * word2 + l3 * word3 + l0 * 1 / heldout_len
    return -1 * sum([smoothed_prods[prob] * np.log2(smoothed_prods[prob]) for prob in smoothed_prods.keys()])
