import pytest 
import numpy as np
from assignment_2 import *
from assignment_utils import *
import re

def test_classes():
    classes = TextClasses("literally there is literally nothing in there you would be literally surprised of".split(), exclude_less=0)
    classes.reduce_to(5)
    print(classes.class_members())
    assert 0

def test_counts():
    classes = TextClasses("literally there is literally nothing in there you would be literally surprised of".split(), exclude_less=0)
    assert classes.make_class_counts() == np.array([3,2,1,1,1,1,1,1,1])

def test_english():
    texten1 = []
    with open("./inputs/TEXTEN1.ptg") as file:
        texten1 = [line.split("/")[0] for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=0)
        classes.reduce_to(15)
        print(classes.class_members())
        print(classes.history[:5])
        assert 0

def test_extract_tag():
    assert extract_tag("word/tag/tag/tag") == "tag/tag/tag"
    assert extract_tag("word/tag") == "tag"

def test_english_reduced():
    texten1 = []
    with open("./inputs/TEXTEN1.ptg") as file:
        texten1 = [line.split("/")[0] for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=10)
        classes.reduce_to(15)
        print(classes.class_members())
        print(classes.history[:5])
        assert 0

def test_english_initial_mf():
    texten1 = []
    with open("./inputs/TEXTEN1.ptg") as file:
        texten1 = [line.split("/")[0] for line in file][:8000]
        classes = TextClasses(texten1, exclude_less=0)
        mf = classes.mutual_info(classes.make_mutual_class_counts(), classes.class_counts, classes.classes)
        print(mf)
        assert mf == 4.99633675507535

def test_simple_mf():

    text = """This is the house that Jack built.
This is the malt that lay in the house that Jack built.
This is the rat that ate the malt
That lay in the house that Jack built.
This is the cat
That killed the rat that ate the malt
That lay in the house that Jack built.
This is the dog that worried the cat
That killed the rat that ate the malt
That lay in the house that Jack built. """
    text  = re.split('[ ,\n.]',text)
    text = [w.lower().rstrip() for w in text if len(w)]
    print(text)


    classes = TextClasses(text, exclude_less=0)
    classes.reduce_to(1)
    set_log_file("./tmp/out.txt")
    print(classes.class_members())
    print(classes.history[:5])

    