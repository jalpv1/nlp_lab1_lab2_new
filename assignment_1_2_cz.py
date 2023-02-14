import numpy as np
from nltk.util import ngrams
import sys

from assignment_utils import split, discount_by, set_log_file, log, tabulate_list
from assignment_1_2_utils import * 

set_log_file("./web/docs/assignment1/assignment12_cz.md")

# %%
log("## Czech Language")
# read the file, make lowercase and strip end of lines
# texten1 = []
# with open("./inputs/TEXTCZ1.txt") as file:
#     texten1 = [line for line in file]
textcz1 = []
with open("./inputs/TEXTCZ1.txt", encoding="iso-8859-2") as file:
    textcz1 = [line for line in file]

# %%
# clean up text

# remove trailing end-of-lines
# texten1 = [line.rstrip().lower() for line in texten1]
textcz1 = [line.rstrip().lower() for line in textcz1]

# unique_words = {w for w in textcz1}
# unique_words_count = len(unique_words)
#print(f"Number of unique words in texten1 {unique_words_count}")

unique_words_cz = {w for w in textcz1}
unique_words_count_cz = len(unique_words_cz)
log(f"Number of unique words in textcz1 {unique_words_count_cz}")


test_data_cz, h_d_cz, tr_d_cz = split(textcz1)

test_data_cz = test_data_cz
h_d_cz = h_d_cz
tr_d_cz = tr_d_cz

log("###  Czech datasets")
log("Data: " + str(len(textcz1)))
log("Train data: " + str(len(tr_d_cz)))
log("Test data: " + str(len(test_data_cz)))
log("Heldout data: " + str(len(h_d_cz)))

unique_words_d_cz = {w for w in tr_d_cz}
unique_words_count_d_cz = len(unique_words_cz)

# ------------------- compute parameters from the heldout data
log("### Compute parameters from the heldout data ")
bigram_heldout_train_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(h_d_cz, 2, pad_right=True)]
trigram_heldout_train_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(h_d_cz, 3, pad_right=True)]
unigram_heldout_train_data = list(h_d_cz)
unigram_heldout_probs, bigram_heldout_probs, trigram_heldout_probs = compute_p(h_d_cz,
                                                                               unigram_heldout_train_data,
                                                                               bigram_heldout_train_data,
                                                                               trigram_heldout_train_data)

l0,l1,l2,l3 = smothing_param(h_d_cz, unigram_heldout_train_data, bigram_heldout_train_data,
               trigram_heldout_train_data, unigram_heldout_probs, bigram_heldout_probs, trigram_heldout_probs, heldout_len=len(h_d_cz))
log("Smoothing params for heldout  "+  str(l0) + ", "+  str(l1) + ", "+  str(l2) + ", "+  str(l3))
# ------------------- compute parameters from the training data
log("### Compute parameters from the training data ")
# vocab_en_train_data_size = len(vacb_en_train_data)
bigram_cz_train_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(tr_d_cz, 2, pad_right=True)]
trigram_cz_train_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(tr_d_cz, 3, pad_right=True)]
unigram_cz_train_data = list(tr_d_cz)

unigram_cz_train_probs, bigram_cz_train_probs, trigram_cz_train_probs = compute_p(tr_d_cz,
                                                                               unigram_cz_train_data,
                                                                               bigram_cz_train_data,
                                                                               trigram_cz_train_data)

l0,l1,l2,l3 = smothing_param(h_d_cz, unigram_cz_train_data, bigram_cz_train_data,
               trigram_cz_train_data,unigram_cz_train_probs, bigram_cz_train_probs, trigram_cz_train_probs, heldout_len=len(h_d_cz))
log("Smoothing params for training  "+  str(l0) + ", "+  str(l1) + ", "+  str(l2) + ", "+  str(l3))
#----------------------------------entropy
log("### Experiments  ")
vacb_cz_train_data = list(set(sorted(tr_d_cz)))
# vocab_cz_train_data_size = len(vacb_cz_train_data)
bigram_cz_test_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(test_data_cz, 2, pad_right=True)]
trigram_cz_test_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(test_data_cz, 3, pad_right=True)]
unigram_cz_test_data = list(test_data_cz)

unigram_cz_test_probs, bigram_cz_test_probs, trigram_cz_test_probs = compute_p(tr_d_cz,
                                                                               unigram_cz_test_data,
                                                                               bigram_cz_test_data,
                                                                               trigram_cz_test_data)
entr = shannon_entropy(unigram_cz_test_data, bigram_cz_test_data, trigram_cz_test_data,
                unigram_cz_test_probs, bigram_cz_test_probs, trigram_cz_test_probs, l0, l1,l2,l3, heldout_len=len(h_d_cz))

#----------------------------------------- experiments

exp1_res = []
for percent in list(range(10,100,10)) + [95,99]:
    new_l0, new_l1, new_l2, new_l3 = discount_by(l0, l1, l2, l3, percent/100)
    entr = shannon_entropy(unigram_cz_test_data, bigram_cz_test_data, trigram_cz_test_data,
                           unigram_cz_test_probs, bigram_cz_test_probs, trigram_cz_test_probs, new_l0,new_l1, new_l2, new_l3, heldout_len=len(h_d_cz))

    exp1_res.append((percent, entr))


tabulate_list(exp1_res, columns=["Percent from difference", "Entropy"])

#log("Entropy for: " +  "percent from difference: "  + str(percent) + " is " + str(entr))
exp2_res = []
for percent in range (0,100,10):
    new_l0, new_l1, new_l2, new_l3 = discount_by(l0, l1, l2, l3, percent/100)
    entr = shannon_entropy(unigram_cz_test_data, bigram_cz_test_data, trigram_cz_test_data,
                           unigram_cz_test_probs, bigram_cz_test_probs, trigram_cz_test_probs, new_l0,new_l1, new_l2, new_l3, heldout_len=len(h_d_cz))

    exp2_res.append((percent,entr))
#list12 = [(key2, val2) for key2, val2 in exp2_res]

tabulate_list(exp2_res, columns=["Percent", "Entropy"])

#  log("Entropy for: " + "percent: "  + str(percent) + " is " + str(entr))
