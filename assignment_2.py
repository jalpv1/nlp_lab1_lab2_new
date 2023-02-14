import numpy as np
from tqdm import tqdm
import math
import random
from typing import Dict, List
from dataclasses import dataclass, field
import sys

from assignment_utils import tabulate_list, log, set_log_file, extract_tag, BiGram, count_bigrams_words_apart, count_words


def pointwise_mf(words, words_apart=2, exclude_less=1):
    words_count = count_words(words)
    ngrams_count = count_bigrams_words_apart(words, words_apart=words_apart, exclude_less=exclude_less)

    pmf = dict()

    for ngram in ngrams_count.keys():
        Pab = ngrams_count[ngram]/len(ngrams_count)
        Pa = words_count[ngram.left]/len(words_count)
        Pb = words_count[ngram.right]/len(words_count)
        Pmf = math.log(Pab/(Pa*Pb))
        pmf[ngram] = Pmf

    return pmf

def best_n(pmfs,n=20):
    return sorted([(ngram, pmf) for ngram, pmf in pmfs.items()], key=lambda item: -item[1])[0:n]

@dataclass
class WordNode():
    left:object = field()
    right:object = field()
    level:int = field()
    words:List[str] = field()
    class_id:str = field()

class TextClasses:

    def __init__(self, words, exclude_less=10) -> None:
        self.words = words
        self.words = ["UNK"] + self.words
        self.words_count = count_words(self.words)
        self.id_to_word = {index:word for index, word in enumerate(self.words_count.keys())}
        self.word_to_id = {word:index for index, word in enumerate(self.words_count.keys())}
        self.exclude_less = exclude_less
        self.bigram_counts = count_bigrams_words_apart(words=self.words, exclude_less=exclude_less)
        print("Bigrams with case and subject")
        found = False
        for i, w in enumerate(self.words[:-1]):
            if w == "case" and self.words[i + 1] == "subject" or w == "subject" and self.words[i + 1] == "case":
                print("Found case and subject at {i}")
        if not found:
            print("No bigram with case and subject, subject never follows case and case never follows subject")
        print(list(filter(lambda bigram: bigram.left == "case" and bigram.right == "subject" or bigram.left == "subject" and bigram.right == "case", self.bigram_counts.keys())))
        self.word_to_class = dict()
        self.class_to_words = dict()
        c_idx = 0
        self.max_class = 0
        self.classes = []
        for w in self.word_to_id.keys():
            c = f"C{c_idx}"
            self.word_to_class[w] = c
            if c not in self.class_to_words:
                self.class_to_words[c] = []
            self.class_to_words[c].append(w)
            self.classes.append(c)
            c_idx += 1
        self.max_class = c_idx
        self.class_counts = {c:self.n_of_words(in_class=c) for c in self.classes}
        # self.id_to_class = {index:cl for index, cl in enumerate(self.classes)}
        self.class_to_id = {cl:index for index, cl in enumerate(self.classes)}
        self.id_to_class = {index:cl for index, cl in enumerate(self.classes)}

        # word to its id numbered like in lectures 0001 00101 etc
        self.word_classes_hierarchy = dict()

        self.reduced_classes = [c for c in self.classes if self.words_count[self.class_to_words[c][0]] >= exclude_less]
        # initially id of the class is the id of the word
        self.history = []
        self.word_nodes = {c:WordNode(left=None, right=None,level=0, class_id=c, words=self.class_to_words[c]) for c in self.reduced_classes}
        self.root = None
    
    def new_class(self):
        self.max_class += 1
        new_idx = self.max_class
        new_id = f"C{new_idx}"
        self.id_to_class[new_idx] = new_id
        self.class_to_id[new_id] = new_idx
        return new_id

    def adjanced_classes_iter(self, classes):
        for i, d in enumerate(classes):
            if i == len(classes) - 1:
                continue
            e = classes[i + 1]
            yield d, e

    def reduce_to(self, nclasses=15):
        
        mutual_class_counts = self.make_mutual_class_counts()

        while len(self.reduced_classes) >= nclasses:
            I_current = self.mutual_info(mutual_class_counts, self.class_counts, self.reduced_classes)
            min_loss = sys.maxsize
                            
            c1_to_merge = 0 # l in the lectures
            c2_to_merge = 0 # k in the lectures
            # for d in self.reduced_classes:
            #     for e in self.reduced_classes:
            for i, d in enumerate(self.reduced_classes):
                if i == len(self.reduced_classes) - 1:
                    continue
                e = self.reduced_classes[i + 1]
                if d == e:
                    continue

                merged_mutual_class_counts = self.merge_class_counts(mutual_class_counts, left=d, right=e)
                merged_classes = [c for c in self.reduced_classes if c != e]
                merged_class_counts = self.class_counts.copy()
                merged_class_counts[d] = merged_class_counts[d] + merged_class_counts[e]

                I_merged = self.mutual_info(merged_mutual_class_counts, merged_class_counts, merged_classes)
                loss = I_merged - I_current

                if min_loss >= loss:
                    min_loss = loss
                    c1_to_merge = d
                    c2_to_merge = e
            if len(self.reduced_classes) == 1:
                # merge one remaining class
                self.new_node(self.root.class_id, self.reduced_classes[0], self.class_to_words[self.reduced_classes[0]])
                break

            print(f"Min loss: {min_loss}")
            if c1_to_merge == 0 and c2_to_merge == 0:
                break    
            mutual_class_counts = self.merge(mutual_class_counts, left=c1_to_merge, right=c2_to_merge)
    
    def merge_class_counts(self, mutual_class_counts, left, right):
        merged_class_counts = mutual_class_counts.copy()
        left_idx = self.class_to_id[left]
        right_idx = self.class_to_id[right]
        merged_class_counts[:,left_idx] = merged_class_counts[:,left_idx] + merged_class_counts[:,right_idx]
        # merged_class_counts[left_idx, :] = merged_class_counts[left_idx, :] + merged_class_counts[right_idx, :]
        return merged_class_counts

    def mutual_info(self, mutual_class_counts, class_counts, classes):
        """
        # counts - N of words in class C1 previously with class C0
# initially it is our bigram counts table
# counted as 
# for each w set +=1 for w and w

# Sum(d,e -> C) of p(d,e) log( p (d,e) / (p(d) p(e)) )
# p(d), p(e) - len(classes_to_words(d or e))/len(words)
# p(d,e) - counts table / len(classes)
        """

# define class counts as np.array classes X classes
# when merging make Mi -= sum(row and col of class that disappears)
# class_counts[:, class that remains] = class_counts[:,class that remains] + class_counts[:, class_that_disappears]
# same for rows
# probably x2 for intersections - see text 'dont forget to add this, be carefull at intersections"
# then
# Mi sum == 


        mf = 0
        
        # for d_idx, d  in enumerate(classes[:-1]):

        #     e = classes[d_idx + 1] # d_idx here is index in self.classes as array not class id

        for  d in classes:
            for e in classes:
                if mutual_class_counts[self.class_to_id[d]][self.class_to_id[e]] == 0:
                    continue

                Pd = class_counts[d]/(len(self.words) - 1)
                Pe = class_counts[e]/(len(self.words) - 1)
                
                Pde = mutual_class_counts[self.class_to_id[d]][self.class_to_id[e]] / (len(self.words))
                mf += Pde * math.log(Pde/(Pe * Pd), 2)

        return mf
    
    def n_of_words(self, in_class):
        return sum(self.words_count[word] for word in self.class_to_words[in_class])

    def log_hierarchy(self):

        max_level = self.root.level
        assert len(self.reduced_classes) == 1
        queue = [("0", self.root)]
        hierarchy = []
        while queue:
            prefix, node = queue.pop()

            if node.left:
                queue.append(("0" + prefix, node.left))
            if node.right:
                queue.append(("1" + prefix, node.right))
            words = node.words
            padded_class = prefix.ljust(max_level,"0")
            hierarchy.append((padded_class, ", ".join(words)))
        tabulate_list(hierarchy, ["Class", "words"])


    def new_node(self, left, right, words):
        left_node = self.word_nodes[left]
        right_node = self.word_nodes[right]

        node = WordNode(left=left_node, right=right_node, class_id=left, words=words, level=max(left_node.level+1, right_node.level+1))
        self.word_nodes[left] = node
        self.root = node

    def merge(self, mutual_class_counts, left, right):
        
        for w in self.class_to_words[left]:
            if not w in self.word_classes_hierarchy:
                self.word_classes_hierarchy[w] = ""

            self.word_classes_hierarchy[w] += "0"

        for w in self.class_to_words[right]:
            if not w in self.word_classes_hierarchy:
                self.word_classes_hierarchy[w] = ""

            self.word_classes_hierarchy[w] += "1"

        self.new_node(left, right, self.class_to_words[left] + self.class_to_words[right])

        if right in self.reduced_classes:
            self.reduced_classes.remove(right)

        # new class that will replace left and right
        self.history.append((self.class_to_words[left], self.class_to_words[right]))
        print(f"Merging {left}:{self.class_to_words[left]}/{self.class_counts[left]} and {right}:{self.class_to_words[right]}/{self.class_counts[right]} into {left}, {len(self.reduced_classes)} classes left")

        # now all words from class "left" and "right" merged into class "into"
        self.class_to_words[left] = self.class_to_words[left] + self.class_to_words[right]
        # class "right" does not exist anymore, merged to "into"
        self.class_to_words[right] = []


        return self.merge_class_counts(mutual_class_counts, left, right)

        # # recalculate pmfs
        # # for each class in classes calculate mf
        # for c1 in self.classes:
        #     c1_idx = self.class_to_id[c1]

        #     Pa = len(self.class_to_words[c1])/len(self.words_count)
        #     Pb = len(self.class_to_words[into])/len(self.words_count)
        #     Pab = (len(self.class_to_words[c1]) + len(self.class_to_words[into]))/len(self.words_count)
        #     if Pa != 0 and Pb != 0:
        #         Pmf = math.log(Pab/(Pa*Pb))
        #         self.pmf[into_idx][c1_idx] = Pmf
        #         self.pmf[c1_idx][into_idx] = Pmf




    def class_members(self):
        """ returns list of classes and their members"""
        return {c:self.class_to_words[c] for c in self.reduced_classes}

    # makes 2 dimensional array where class_counts[i[j] == count of cases where word of class J follows word of class I
    def make_mutual_class_counts(self) -> np.array:
        counts = np.zeros((len(self.classes),len(self.classes)))

        for i, word in enumerate(self.words):
            if i == len(self.words) - 1:
                 break 
            
            # if word in self.word_to_class and self.words[i + 1] in self.word_to_class: # i.e. not ignored
            class_idx = self.class_to_id[self.word_to_class[word]]
            following_class_idx = self.class_to_id[self.word_to_class[self.words[i + 1]]]
            # left word was followed by right word, so increment count
            counts[class_idx][following_class_idx] += 1

        return counts



# merge
# define class counts as np.array classes X classes
# when merging make Mi -= sum(row and col of class that disappears)
# class_counts[:, class that remains] = class_counts[:,class that remains] + class_counts[:, class_that_disappears]
# same for rows
# probably x2 for intersections - see text 'dont forget to add this, be carefull at intersections"
# then
# Mi sum == 

# now Q - how ot calculate Mi in the first place? 

if __name__ == "__main__":
    set_log_file("./web/docs/assignment2/assignment2.md")

    log("# Task N2. Words and The Company They Keep")
    log("## Task N2.1 Best Friends")

    # read the file, make lowercase and strip end of lines
    texten1 = []
    with open("./inputs/TEXTEN1.txt") as file:
        texten1 = [line.rstrip().lower() for line in file]
    textcz1 = []
    with open("./inputs/TEXTCZ1.txt", encoding="iso-8859-2") as file:
        textcz1 = [line.rstrip().lower() for line in file]

    en1_pmfs = pointwise_mf(texten1, exclude_less=10)
    log("### Best 20 friends in English text, pairs")
    tabulate_list(best_n(en1_pmfs,n=20), columns=["Pair","PMF"])

    cz1_pmfs = pointwise_mf(textcz1, exclude_less=10)
    log("### Best 20 friends in Czech text, pairs")
    tabulate_list(best_n(cz1_pmfs,n=20), columns=["Pair","PMF"])

    en1_pmfs = pointwise_mf(texten1, words_apart=25, exclude_less=10)
    log("### Best 20 friends in English text, words that are 25 words apart")
    tabulate_list(best_n(en1_pmfs,n=20), columns=["Pair","PMF"])

    cz1_pmfs = pointwise_mf(textcz1, words_apart=25, exclude_less=10)
    log("### Best 20 friends in Czech text, words that are 25 words apart")
    tabulate_list(best_n(cz1_pmfs,n=20), columns=["Pair","PMF"])

    log("## Task N2.2 Word Classes")

    log("### English text")
    with open("./inputs/TEXTEN1.ptg") as file:
        texten1 = [line.split("/")[0].rstrip() for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=0)
        mf = classes.mutual_info(classes.make_mutual_class_counts(), classes.class_counts, classes.classes)
        log(f"Initial mutual information: {mf}")

        classes = TextClasses(texten1, exclude_less=10)
        classes.reduce_to(1)
        log("#### Full hierarchy of words for words occuring 10 times or more, limit 8000")
        classes.log_hierarchy()
        log("\n")
        log("#### History of merges")
        for h in classes.history:
            log(f"Merged \"{', '.join(h[0])}\" and \"{', '.join(h[1])}\"\n")

        classes = TextClasses(texten1, exclude_less=10)
        classes.reduce_to(15)
        log("#### 15 Classes")
        tabulate_list(classes.class_members().items(), ["Class", "Words"])

    log("### Czech text")
    with open("./inputs/TEXTCZ1.ptg", encoding="iso-8859-2") as file:
        texten1 = [line.split("/")[0].rstrip() for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=0)
        mf = classes.mutual_info(classes.make_mutual_class_counts(), classes.class_counts, classes.classes)
        log(f"Initial mutual information: {mf}")

        classes = TextClasses(texten1, exclude_less=10)
        classes.reduce_to(1)
        log("#### Full hierarchy of words for words occuring 10 times or more, limit 8000")
        classes.log_hierarchy()
        log("\n")
        log("#### History of merges")
        for h in classes.history:
            log(f"Merged \"{', '.join(h[0])}\" and \"{', '.join(h[1])}\"\n")

        classes = TextClasses(texten1, exclude_less=10)
        classes.reduce_to(15)
        log("#### 15 Classes")
        tabulate_list(classes.class_members().items(), ["Class", "Words"])

    log("## Task N2.3 Tag Classes")

    log("### English tags")

    with open("./inputs/TEXTEN1.ptg") as file:
        texten1 = [extract_tag(line).rstrip() for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=5)
        classes.reduce_to(15)
        log("### 15 Classes")
        tabulate_list(classes.class_members().items(), ["Class", "Tags"])
        log("### History of merges")
        for h in classes.history:
            log(f"Merged \"{', '.join(h[0])}\" and \"{', '.join(h[1])}\"\n")

    log("### Czech tags")

    with open("./inputs/TEXTCZ1.ptg", encoding="iso-8859-2") as file:
        texten1 = [extract_tag(line).rstrip() for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=5)
        classes.reduce_to(15)
        log("### 15 Classes")
        tabulate_list(classes.class_members().items(), ["Class", "Tags"])
        log("### History of merges")
        for h in classes.history:
            log(f"Merged \"{', '.join(h[0])}\" and \"{', '.join(h[1])}\"\n")
