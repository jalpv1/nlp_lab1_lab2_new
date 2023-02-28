from typing import Dict, List, Tuple
import sys
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class BiGram:
    left:str
    right:str

    def __str__(self) -> str:
        return f"{self.left} {self.right}"

@dataclass(eq=True, frozen=True)
class TriGram:
    first:str
    second:str
    third:str

    def __str__(self) -> str:
        return f"{self.first} {self.second} {self.third}"

def count_ngrams(ngrams_data:List[BiGram | TriGram]) -> Dict[BiGram | TriGram, int]:
    ngrams = dict()
    for ngram in ngrams_data:
        if ngram not in ngrams:
            ngrams[ngram] = 1
        else:
            ngrams[ngram] += 1
    return ngrams


def count_bigrams_words_apart(words:List[str], words_apart=2, exclude_less=0) -> Dict[BiGram, int]:
    ngrams = dict()

    for i, word in enumerate(words):
        j = i + words_apart - 1
        if j >= len(words):
            break
        ngram = bigram_from(word, words[j])
        if ngram not in ngrams:
            ngrams[ngram] = 0
        ngrams[ngram] += 1
    return {p:count for p,count in ngrams.items() if count >= exclude_less}

def count_words(words:List[str]) -> Dict[str,int]:
    words_count = dict()
    for w in words:
        if w not in words_count:
            words_count[w] = 0
        words_count[w] += 1
    return words_count


def bigram_from(left, right):
    return BiGram(left=left, right=right)

def trigram_from(first, second, third):
    return TriGram(first=first, second=second, third=third)

def split(text):
    # take last 20000 
    test = text[-20000:]

    # take remainder from previous step and take last 40 000 words
    heldout = text[:-20000][-40000:]
    
    # take remainder for training, since test + heldout both are 60000 we take all except  last 60000 words
    training = text[:-60000]

    # check if we did not mess with slices 
    assert len(heldout) + len(training) + len(test) == len(text), f"{len(training)}, {len(heldout)}, {len(test)}, {len(text)}"
    assert len(heldout) == 40000, len(heldout)
    assert len(training) == len(text) - 20000 - 40000
    return test, heldout, training
    # example:
    # test = [1,2,3,4,5,6,7,8,9,10][-2:], heldout = ([1,2,3,4,5,6,7,8,9,10][:-2])[-4:], training = [1,2,3,4,5,6,7,8,9,10][:-6], training + heldout + test == [1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9, 10]

def add_percent(l0, l1, l2, l3, percent):
    assert percent != 0
    delta = (1 - l3)*percent
    assert delta != 0
    new_l3 = l3 + delta
    new_l1 = l1 - l1*percent
    new_l2 = l2 - l2*percent
    new_l0 = 1 - new_l1 - new_l2 - new_l3
    assert new_l3 >= 0,  f"{new_l3}, {l3}"
    assert new_l2 >= 0,  f"{new_l2}, {l2}"
    assert new_l1 >= 0,  f"{new_l1}, {l1}"
    assert new_l0 >= 0,  f"{new_l0}, {l0}"
    return new_l0, new_l1, new_l2, new_l3

def set_percent(l0, l1, l2, l3, percent):
    """
    set the trigram smoothing parameter to 90%, 80%, 70%, ... 10%, 0% of its value,
    """
    new_l3 = l3*percent
    delta = abs(l3 - new_l3)
    new_l1 = l1 + l1*percent
    new_l2 = l2 + l2*percent
    new_l0 = 1 - new_l1 - new_l2 - new_l3
    return new_l0, new_l1, new_l2, new_l3

log_file = ""
def log(text):
    assert log_file != ""
    with open(log_file, "a") as output:
        print(text, file=output)
        print(text)

def set_log_file(file):
    global log_file
    log_file = file
    with open(log_file, "w") as output:
        output.truncate()

def tabulate_list(data, columns):
    caption = "".join([f"| {c:^24} " for c in columns]) + "|"
    log(caption)
    line = "".join([f"| {'-'*24} " for c in columns]) + "|"
    log(line)
    for (p, result)  in data:
        # print(p, result)
        if isinstance(result, List):
            result = ", ".join(result)
        log(f"| {str(p):<24} | {result:<24} |")
    log("")

def extract_tag(line):
    return line[line.find("/") + 1:]