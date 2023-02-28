#%%
import numpy as np
from tqdm import tqdm
import math
import random
from typing import Dict, List
from nltk.util import ngrams

from assignment_utils import set_log_file, log, tabulate_list, BiGram, count_bigrams_words_apart, count_words, count_ngrams, bigram_from
#%%

def entropy(text:List[str]):
    word_counts = count_words(text)
    bigram_counts = count_ngrams([bigram_from(ngram[0], ngram[1]) for ngram in ngrams(text, 2, pad_right=True)])
    # print(f"bigram counts:{len(bigram_counts)}")
    # bigram_counts = count_bigrams_words_apart(text, exclude_less=0)

    HIPipeJ = 0
    n_of_bigrams = len(bigram_counts) #len(text) - 1 # T + 1 - N, len(text) + 1 - 2
    for bigram, bigram_count in bigram_counts.items():
        Pij = bigram_count/len(text)
        if bigram.right:
            PJPipeJ = bigram_count/word_counts[bigram.right]
            HIPipeJ += - Pij * math.log(PJPipeJ, 2)
    return HIPipeJ

# Convert algorithm above to reusable function for the next task
class EntropyCalculator:

    def __init__(self, text) -> None:
        self.original_text = text       
        self.unique_words = list(set(text))
        self.characters_in_text = list(set("".join(self.unique_words)))

    def entropy(self, text):
        HIPipeJ = entropy(text)
        return HIPipeJ

    ## reusable functions
    # replaces multiple characters in the string at once
    def replace_many(self, word, chars:Dict[str,str]):
        # for c_from, c_to in chars.items():
        #     word = word.replace(c_from, c_to)
        return word.translate(str.maketrans(chars))

    def run_experiment(self, ntimes, probabilities):
        results = {p:{"min_messed_chars":0,"min_messed_words":0,"max_messed_chars":0,"max_messed_words":0,"avg_messed_chars":0,"avg_messed_words":0} for p in probabilities}
        probs_bar = tqdm(probabilities)
        for p in probs_bar:
            probs_bar.set_description(f"Probability of messing: {p}%")
            entropies_with_messed_chars = []
            entropies_with_messed_words = []
            ranges_bar = tqdm(range(ntimes), leave=False)
            for n in ranges_bar:
                ranges_bar.set_description(f"Run N: {n}")
                messed_text = self.mess_chars(self.original_text, self.characters_in_text, p/100)
                entropies_with_messed_chars.append(self.entropy(messed_text))
                messed_text = self.mess_words(self.original_text, self.unique_words, p/100)
                entropies_with_messed_words.append(self.entropy(messed_text))

            results[p]["min_messed_chars"] = min(entropies_with_messed_chars)
            results[p]["min_messed_words"] = min(entropies_with_messed_words)
            results[p]["max_messed_chars"] = max(entropies_with_messed_chars)
            results[p]["max_messed_words"] = max(entropies_with_messed_words)            
            results[p]["avg_messed_chars"] = sum(entropies_with_messed_chars)/len(entropies_with_messed_chars)
            results[p]["avg_messed_words"] = sum(entropies_with_messed_words)/len(entropies_with_messed_words)

        return results

    # messes chars in the list of words with given probability
    def mess_chars(self, text, characters_in_text, probability):
        chars_to_mess = {}
        for c in characters_in_text:
            # selects True with given probability
            probability_to_mess = random.choices([True,False],weights=[probability,1 - probability])
            if probability_to_mess:
                char_to_replace = characters_in_text[int(random.random()*len(characters_in_text))]
                chars_to_mess[c] = char_to_replace
        messed_words = [self.replace_many(w,chars_to_mess) for w in text]
        return messed_words
    
    def mess_words(self, text, unique_words, probability):
        words_to_mess = {}
        for w in unique_words:
            # selects True with given probability
            probability_to_mess = random.choices([True,False],weights=[probability,1 - probability])
            if probability_to_mess: # if we chose to mess the word w
                # find random replacement 
                word_to_replace = unique_words[int(random.random()*len(unique_words))]
                # store replacement
                words_to_mess[w] = word_to_replace
        
        # execute all stored replacements
        # words_to_mess.get(w,w) - returns word stored in dict words_to_mess and if not found then returs word itself, i.e. no replacement is made
        messed_words = [words_to_mess.get(w,w) for w in text]
        return messed_words
def print_md_table(results, columns):
    caption = f"| {'Probability, %':^24} | {'Min Entropy':^24} | {'Max Entropy':^24} | {'Avg Entropy':^24} |"
    log(caption)
    log(f"| {'-'*24} | {'-'*24} | {'-'*24} | {'-'*24} |")
    for p,result  in results.items():
        line = f"| {p:<24} "
        for c in columns:
            value = result[c]
            line += f"| {value:<24}" 
        log(line + " |")
    log("")

if __name__ == "__main__":
    set_log_file("./web/docs/assignment1/assignment1.md")

    log("# Task N1. Entropy of a Text")
    log("## Task N1.1 Compute this conditional entropy and perplexity for the file TEXTEN1.txt")

    # read the file, make lowercase and strip end of lines
    texten1 = []
    with open("./inputs/TEXTEN1.txt", encoding="iso-8859-2") as file:
        texten1 = [line for line in file]
    textcz1 = []
    with open("./inputs/TEXTCZ1.txt", encoding="iso-8859-2") as file:
        textcz1 = [line for line in file]

    # %%
    # clean up text

    # remove trailing end-of-lines
    texten1 = [line.rstrip() for line in texten1]
    textcz1 = [line.rstrip() for line in textcz1]

    word_counts = count_words(texten1)

    unique_words_en = word_counts.keys()
    unique_words_count = len(unique_words_en)
    log(f"Number of unique words in texten1 {unique_words_count}")

    words_count_cz = count_words(textcz1)
    unique_words_cz = words_count_cz.keys()
    unique_words_count_cz = len(unique_words_cz)
    log(f"Number of unique words in textcz1 {unique_words_count_cz}")

    HIPipeJ = entropy(text=texten1)

    PX = math.pow(2, HIPipeJ)

    log(f"Conditional entropy for English {HIPipeJ}")
    log(f"Perplexity for English PX = {PX}")

    HIPipeJ = entropy(text=textcz1)

    PX = math.pow(2, HIPipeJ)

    log(f"Conditional entropy for Czech {HIPipeJ}")
    log(f"Perplexity for Czech PX = {PX}")

    probabilities = [10, 5, 1, 0.1, 0.01, 0.001] 

    log("## Task N1.2 Mess up with text and measure how it affects entropy for the file TEXTEN1.txt")
    texten1_calc = EntropyCalculator(text=texten1)
    texten1_results = texten1_calc.run_experiment(10, probabilities)


    log("### Entropies of texten1 with messed CHARs")
    print_md_table(texten1_results, columns=["min_messed_chars","max_messed_chars","avg_messed_chars"])

    log("### Entropies of texten1 with messed WORDs")
    print_md_table(texten1_results, columns=["min_messed_words","max_messed_words","avg_messed_words"])

    textcz_calc = EntropyCalculator(text=textcz1)
    textcz_results = textcz_calc.run_experiment(10, probabilities)
    log("### Entropies of textcz1 with messed CHARs ---")
    print_md_table(textcz_results, columns=["min_messed_chars","max_messed_chars","avg_messed_chars"])

    log("### Entropies of textcz1 with messed WORDs")
    print_md_table(textcz_results, columns=["min_messed_words","max_messed_words","avg_messed_words"])

    import matplotlib.pyplot as plt

    min_messed_chars = [texten1_results[p]["min_messed_chars"] for p in probabilities]
    max_messed_chars = [texten1_results[p]["max_messed_chars"] for p in probabilities]
    avg_messed_chars = [texten1_results[p]["avg_messed_chars"] for p in probabilities]
    plt.figure(1)
    plt.title("Entropy when CHARs are messed in TEXTEN1")
    plt.xlabel("Probability of change")
    plt.plot(probabilities, min_messed_chars, label = "Min Entropy", linestyle="dashed")
    plt.plot(probabilities, max_messed_chars, label = "Max Entropy", linestyle="dashdot")
    plt.plot(probabilities, avg_messed_chars, label = "Avg Entropy", linestyle="solid")
    plt.legend()
    plt.savefig("./web/docs/assignment1/texten1_chars.jpg")
    log("### Entropy when CHARs are messed in TEXTEN1")
    log("![alt text for screen readers](texten1_chars.jpg)")


    min_messed_words = [texten1_results[p]["min_messed_words"] for p in probabilities]
    max_messed_words = [texten1_results[p]["max_messed_words"] for p in probabilities]
    avg_messed_words = [texten1_results[p]["avg_messed_words"] for p in probabilities]
    plt.figure(2)
    plt.title("Entropy when WORDs are messed in TEXTEN1")
    plt.xlabel("Probability of change")
    plt.plot(probabilities, min_messed_words, label = "Min Entropy", linestyle="dashed")
    plt.plot(probabilities, max_messed_words, label = "Max Entropy", linestyle="dashdot")
    plt.plot(probabilities, avg_messed_words, label = "Avg Entropy", linestyle="solid")
    plt.legend()
    plt.savefig("./web/docs/assignment1/texten1_words.jpg")
    log("### Entropy when WORDs are messed in TEXTEN1")
    log("![alt text for screen readers](texten1_words.jpg)")

    min_messed_chars = [textcz_results[p]["min_messed_chars"] for p in probabilities]
    max_messed_chars = [textcz_results[p]["max_messed_chars"] for p in probabilities]
    avg_messed_chars = [textcz_results[p]["avg_messed_chars"] for p in probabilities]
    plt.figure(3)
    plt.title("Entropy when CHARs are messed in TEXTCZ1")
    plt.xlabel("Probability of change")
    plt.plot(probabilities, min_messed_chars, label = "Min Entropy", linestyle="dashed")
    plt.plot(probabilities, max_messed_chars, label = "Max Entropy", linestyle="dashdot")
    plt.plot(probabilities, avg_messed_chars, label = "Avg Entropy", linestyle="solid")
    plt.legend()
    log("### Entropy when CHARs are messed in TEXTCZ1")
    log("![alt text for screen readers](textcz1_chars.jpg)")
    plt.savefig("./web/docs/assignment1/textcz1_chars.jpg")



    min_messed_words = [textcz_results[p]["min_messed_words"] for p in probabilities]
    max_messed_words = [textcz_results[p]["max_messed_words"] for p in probabilities]
    avg_messed_words = [textcz_results[p]["avg_messed_words"] for p in probabilities]
    plt.figure(4)
    plt.title("Entropy when WORDs are messed in TEXTCZ1")
    plt.xlabel("Probability of change")
    plt.plot(probabilities, min_messed_words, label = "Min Entropy", linestyle="dashed")
    plt.plot(probabilities, max_messed_words, label = "Max Entropy", linestyle="dashdot")
    plt.plot(probabilities, avg_messed_words, label = "Avg Entropy", linestyle="solid")
    plt.legend()
    log("### Entropy when WORDs are messed in TEXTCZ1")
    log("![alt text for screen readers](textcz1_words.jpg)")
    plt.savefig("./web/docs/assignment1/textcz1_words.jpg")

