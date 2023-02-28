import numpy as np
from nltk.util import ngrams
import sys

from assignment_utils import split, add_percent, set_percent, set_log_file, log, tabulate_list
from assignment_1_2_utils import * 

set_log_file("./web/docs/assignment1/assignment12_cz.md")

# %%
log("## Czech Language")
# read the file, make lowercase and strip end of lines
textcz1 = []
with open("./inputs/TEXTCZ1.txt", encoding="iso-8859-2") as file:
    textcz1 = [line for line in file]

# %%
# clean up text

# remove trailing end-of-lines
# texten1 = [line.rstrip() for line in texten1]
textcz1 = [line.rstrip() for line in textcz1]

unique_words_cz = {w for w in textcz1}
unique_words_count_cz = len(unique_words_cz)
log(f"Number of unique words in textcz1 {unique_words_count_cz}")

# create datasets

test_data, heldout_data, training_data = split(textcz1)

log("### Czech datasets")
log("Data: " + str(len(textcz1)))
log("Train data: " + str(len(training_data)))
log("Test data: " + str(len(test_data)))
log("Heldout data: " + str(len(heldout_data)))

# ------------------- compute parameters from the heldout data
log("### Compute parameters from the heldout data ")
vacb_heldout_data = list(set(heldout_data))
bigram_heldout_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(heldout_data, 2)]
trigram_heldout_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(heldout_data, 3)]
unigram_heldout_data = heldout_data

bigram_train_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(training_data, 2)]
trigram_train_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(training_data, 3)]
unigram_train_data = training_data

bigram_test_data = [bigram_from(ngram[0], ngram[1]) for ngram in ngrams(test_data, 2)]
trigram_test_data = [trigram_from(ngram[0], ngram[1], ngram[2]) for ngram in ngrams(test_data, 3)]
unigram_test_data = test_data


# print(len(bigram_heldout_data))
# print(len(trigram_heldout_data))

# unigram_heldout_probs, bigram_heldout_probs, trigram_heldout_probs = compute_p(heldout_data,
#                                                                                unigram_heldout_data,
#                                                                                bigram_heldout_data,
#                                                                                trigram_heldout_data)

l0,l1,l2,l3 = smothing_param(trigram_heldout_data, unigram_train_data, bigram_train_data, trigram_train_data)
log("Smoothing params for heldout  "+  str(l0) + ", "+  str(l1) + ", "+  str(l2) + ", "+  str(l3))

# ------------------- compute parameters from the training data
# log("### Compute parameters from the training data ")

entropy = cross_entropy(l0=l0,l1=l1,l2=l2,l3=l3, dataset=test_data, trigram_test_data=trigram_test_data,trigram_train_data=trigram_train_data,bigram_train_data=bigram_train_data,unigram_train_data=unigram_train_data)
log(f"Cross entropy with lambdas {entropy}")

#----------------------------------entropy

log("### Experiments  ")

#-----------------------------------------experiments
log("Adding 10%, 20%, 30%, ..., 90%, 95% and 99% of the difference between the trigram smoothing parameter and 1.0 to its value *discounting* other parameters proportionally")
log("")
exp1_res = []
for percent in list(range(10,100,10)) + [95,99]:
    new_l0, new_l1, new_l2, new_l3 = add_percent(l0, l1, l2, l3, percent/100)
    entropy = cross_entropy(l0=new_l0, l1=new_l1, l2=new_l2, l3=new_l3, dataset=test_data, trigram_test_data=trigram_test_data,trigram_train_data=trigram_train_data,bigram_train_data=bigram_train_data,unigram_train_data=unigram_train_data)
    exp1_res.append((percent, entropy))


tabulate_list(exp1_res, columns=["Percent", "Entropy"])

log("----------------------------------------------------------------------------------------")
log("Setting trigram smoothng parameter to 90%, 80%, 70%, ... 10%, 0% of its value, *boosting* other parameters proportionally")
log("")
exp2_res = []
for percent in range (0, 100, 10):
    new_l0, new_l1, new_l2, new_l3 = set_percent(l0, l1, l2, l3, percent/100)
    entropy = cross_entropy(l0=new_l0, l1=new_l1, l2=new_l2, l3=new_l3, dataset=test_data, trigram_test_data=trigram_test_data,trigram_train_data=trigram_train_data,bigram_train_data=bigram_train_data,unigram_train_data=unigram_train_data)

    exp2_res.append((percent, entropy))

tabulate_list(exp2_res, columns=["Percent", "Entropy"])

#  print("Entropy for: " + "percent: "  + str(percent) + " is " + str(entr))
