import numpy as np
from nltk.util import ngrams
import sys

from assignment_utils import split, discount_by, set_log_file, log, tabulate_list, count_words, count_ngrams, BiGram, TriGram, bigram_from, trigram_from

from assignment_1_2_utils import * 

set_log_file("./web/docs/assignment1/assignment12.md")

# %%
log("# Task N2. Cross-Entropy and Language Modeling")
log("## English Language")

# read the file, make lowercase and strip end of lines
texten1 = []
with open("./inputs/TEXTEN1.txt") as file:
    texten1 = [line for line in file]
# textcz1 = []
# with open("./inputs/TEXTCZ1.txt", encoding="iso-8859-1") as file:
#     textcz1 = [line for line in file]

# %%
# clean up text

# remove trailing end-of-lines
texten1 = [line.rstrip().lower() for line in texten1]
# textcz1 = [line.rstrip().lower() for line in textcz1]

unique_words = {w for w in texten1}
unique_words_count = len(unique_words)
log(f"Number of unique words in texten1 {unique_words_count}")

# unique_words_cz = {w for w in textcz1}
# unique_words_count_cz = len(unique_words_cz)
# log(f"Number of unique words in textcz1 {unique_words_count_cz}", file=output)

#create datasets

test_data_en, h_d_en, tr_d_en = split(texten1)

test_data_en = test_data_en
h_d_en = h_d_en
tr_d_en = tr_d_en

# c = textcz1[::-1]
# test_data_cz = t[:20000]
# h_d_cz = t[20000:60000]
# tr_d_cz = t[60000:]

# test_data_cz = test_data_cz[::-1]
# h_d_cz = h_d_cz[::-1]
# tr_d_cz = tr_d_cz[::-1]

log("### English datasets")
log("Data: " + str(len(texten1)))
log("Train data: " + str(len(tr_d_en)))
log("Test data: " + str(len(test_data_en)))
log("Heldout data: " + str(len(h_d_en)))



# ------------------- compute parameters from the heldout data
log("### Compute parameters from the heldout data ")
vacb_en_heldout_data = list(set(sorted(h_d_en)))
bigram_heldout_train_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(h_d_en, 2, pad_right=True)]
trigram_heldout_train_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(h_d_en, 3, pad_right=True)]
unigram_heldout_train_data = list(h_d_en)
unigram_heldout_probs, bigram_heldout_probs, trigram_heldout_probs = compute_p(h_d_en,
                                                                               unigram_heldout_train_data,
                                                                               bigram_heldout_train_data,
                                                                               trigram_heldout_train_data)

l0,l1,l2,l3 = smothing_param(h_d_en, unigram_heldout_train_data, bigram_heldout_train_data,
               trigram_heldout_train_data, unigram_heldout_probs, bigram_heldout_probs, trigram_heldout_probs, heldout_len=len(h_d_en))
log("Smoothing params for heldout  "+  str(l0) + ", "+  str(l1) + ", "+  str(l2) + ", "+  str(l3))

# ------------------- compute parameters from the training data
log("### Compute parameters from the training data ")
vacb_en_train_data = list(set(sorted(tr_d_en)))
# vocab_en_train_data_size = len(vacb_en_train_data)
bigram_en_train_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(tr_d_en, 2, pad_right=True)]
trigram_en_train_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(tr_d_en, 3, pad_right=True)]
unigram_en_train_data = list(tr_d_en)

unigram_en_train_probs, bigram_en_train_probs, trigram_en_train_probs = compute_p(tr_d_en,
                                                                               unigram_en_train_data,
                                                                               bigram_en_train_data,
                                                                               trigram_en_train_data)

l0,l1,l2,l3 = smothing_param(h_d_en, unigram_en_train_data, bigram_en_train_data,
               trigram_en_train_data,unigram_en_train_probs, bigram_en_train_probs, trigram_en_train_probs, heldout_len=len(h_d_en))
log("Smoothing params for training  "+  str(l0) + ", "+  str(l1) + ", "+  str(l2) + ", "+  str(l3))

#----------------------------------entropy
log("### Experiments  ")

vacb_en_train_data = list(set(sorted(tr_d_en)))
# vocab_en_train_data_size = len(vacb_en_train_data)
bigram_en_test_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(test_data_en, 2, pad_right=True)]
trigram_en_test_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(test_data_en, 3, pad_right=True)]
unigram_en_test_data = list(test_data_en)

unigram_en_test_probs, bigram_en_test_probs, trigram_en_test_probs = compute_p(tr_d_en,
                                                                               unigram_en_test_data,
                                                                               bigram_en_test_data,
                                                                               trigram_en_test_data)
entr = shannon_entropy(unigram_en_test_data, bigram_en_test_data, trigram_en_test_data,
                unigram_en_test_probs, bigram_en_test_probs, trigram_en_test_probs, l0, l1,l2,l3, heldout_len=len(h_d_en))

#-----------------------------------------experiments
exp1_res = []
for percent in list(range(10,100,10)) + [95,99]:
    new_l0, new_l1, new_l2, new_l3 = discount_by(l0, l1, l2, l3, percent/100)
    entr = shannon_entropy(unigram_en_test_data, bigram_en_test_data, trigram_en_test_data,
                           unigram_en_test_probs, bigram_en_test_probs, trigram_en_test_probs, new_l0,new_l1, new_l2, new_l3, heldout_len=len(h_d_en))

    exp1_res.append((percent, entr))
    #log("Entropy for: " + "percent: "  + str(percent) + " is " + str(entr))


tabulate_list(exp1_res, columns=["Percent from difference", "Entropy"])

#log("Entropy for: " +  "percent from difference: "  + str(percent) + " is " + str(entr))
log("----------------------------------------------------------------------------------------")
exp2_res = []
for percent in range (0,100,10):
    new_l0, new_l1, new_l2, new_l3 = discount_by(l0, l1, l2, l3, percent/100)
    entr = shannon_entropy(unigram_en_test_data, bigram_en_test_data, trigram_en_test_data,
                           unigram_en_test_probs, bigram_en_test_probs, trigram_en_test_probs, new_l0,new_l1, new_l2, new_l3, heldout_len=len(h_d_en))

    exp2_res.append((percent,entr))
#list12 = [(key2, val2) for key2, val2 in exp2_res]

tabulate_list(exp2_res, columns=["Percent", "Entropy"])

#  print("Entropy for: " + "percent: "  + str(percent) + " is " + str(entr))
