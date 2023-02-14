import pytest 
from assignment_2 import *
from assignment_utils import *
from assignment_1_1 import *

def test_count_pairs_unique_pairs():
    pairs = count_bigrams_words_apart("this is the only sentence in Las Vegas".split(), exclude_less=1)
    assert len(pairs) == 7, pairs

def test_count_pairs_repeating_pairs():
    pairs = count_bigrams_words_apart("this is the only this is sentence in Las Vegas".split(), exclude_less=0)
    assert len(pairs) == 8, pairs

def test_count_pairs_filter_less_than_one():
    pairs = count_bigrams_words_apart("this is the only sentence in Las Vegas, so Las Vegas has the only sentence".split(), exclude_less=2)
    assert len(pairs) == 2, f"Pairs are {pairs}"

def test_count_words():
    words = count_words("this is the only sentence in Las Vegas".split())
    assert len(words) == 8, words

def test_pmf():
    pmfs = pointwise_mf("this is the only sentence in Las Vegas, so Las Vegas has the only sentence".split())
    print(pmfs)
    assert pmfs, f"Pmfs were {pmfs}"

def test_best_n():
    l = {f"{x}":x for x in range(100)}
    assert set([(f"{x}", x) for x in range(95,100)]) == set(best_n(l,n=5))

def test_classes():
    classes = TextClasses("literally there is nothing in there you would be surprised of".split(" "), exclude_less=3)
    classes.reduce_to(3)
    print(classes.class_members())

def test_entropy():
    first = "червоне то любов а чорне то журба".split(" ")
    second = "red is love but black is sorrow".split(" ")
    print(f"{entropy(first)} is entropy of first")
    print(f"{entropy(first + second)} is entropy of first + second")
    assert entropy(first) == entropy(first + second)