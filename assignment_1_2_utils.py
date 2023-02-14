import numpy as np
from nltk.util import ngrams
import sys

from assignment_utils import split, discount_by, set_log_file, log, tabulate_list, count_words, count_ngrams, BiGram, TriGram, bigram_from, trigram_from

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

# calculate smoothing params
def smothing_param(dataset, unigram_data, bigram_data, trigram_data, unigram_probs1,
                   bigram_probs1, trigram_probs1, heldout_len):
    l2 = 0.1
    l3 = 0.1
    l1 = 0.3
    l4 = 0.3
    l0 = 1 - l4 - l2 - l2
    smoothed_prods = dict()
    m = 0
    c_l0 = 0
    c_l1 = 0
    c_l2 = 0
    c_l3 = 0
    e = 0.001
    uniform_probability = 1 / len(dataset)
    while (True):
        # if m > 100:
        #     break
        m = m + 1
        for i in range(len(unigram_data)):
            word1 = unigram_probs1[unigram_data[i]]
            if i + 2 == len(unigram_data) or i + 1 == len(unigram_data):
                break

            word2 = bigram_probs1[bigram_data[i]]
            word3 = trigram_probs1[trigram_data[i]]

            smoothed_prods[unigram_data[i]] = l1 * word1 + l2 * word2 + l3 * word3 + l0 * 1 / heldout_len
        for word1 in smoothed_prods.keys():
            c_l0 += l0 * uniform_probability / smoothed_prods[word1]
        # print("c0 " + str(c_l0))
        for word1 in unigram_probs1.keys():
            c_l1 += l1 * unigram_probs1[word1] / smoothed_prods[word1]
        # print("c1 " + str(c_l1))

        for i in range(len(bigram_data)):
            c_l2 += l2 * bigram_probs1[bigram_data[i]] / smoothed_prods[unigram_data[i]]

        for i in range(len(trigram_data)):
            c_l3 += l3 * trigram_probs1[trigram_data[i]] / smoothed_prods[unigram_data[i]]
        # print("c2 " + str(c_l2))

        l0_new = c_l0 / (c_l0 + c_l1 + c_l2 + c_l3)
        l1_new = c_l1 / (c_l0 + c_l1 + c_l2 + c_l3)
        l2_new = c_l2 / (c_l0 + c_l1 + c_l2 + c_l3)
        l3_new = c_l3 / (c_l0 + c_l1 + c_l2 + c_l3)
        #print("l0_new " + str(l0_new))
        # print("l1_new " + str(l1_new))
        # print("l2_new " + str(l2_new))
        # print("l3_new " + str(l3_new))
        #
        # print("l0 " + str(l0))
        # print("l1 " + str(l1))
        # print("l2 " + str(l2))
        # print("l3 " + str(l3))
        sum_l = l0 + l1 + l2 + l3
        # print(str(sum_l))

        if abs(l0_new - l0) < e and abs(l1_new - l1) < e and abs(l2_new - l2) < e and abs(l3_new - l3) < e:
            break
        l0 = l0_new
        l1 = l1_new
        l2 = l2_new
        l3 = l3_new
    return [l0, l1, l2, l3]

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
