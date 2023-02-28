import pytest 
from assignment_2 import *
from assignment_utils import *
from assignment_1_2_utils import *
from assignment_1_1 import *

def test_equals_with():
    assert equals_with(1.0,1.0, precision=0)
    assert equals_with(1.0,1.099, precision=0.1)
    assert not equals_with(1.0,1.2, precision=0.1)
    assert equals_with(1.0, 0.9, precision=0.1)
    assert equals_with(1.0, 0.9, precision=0.3)
    assert not equals_with(1.0, 0.9, precision=0.01)
    assert equals_with(-1.0,-0.9, precision=0.1)
    assert not equals_with(-1.0, 0.9, precision=0.1)

def test_discount_by():
    l0,l1,l2,l3 = [0.25,0.25,0.25,0.25]
    l0_new,l1_new,l2_new,l3_new = discount_by(l0, l1, l2, l3, 0.1)
    print(l0_new,l1_new,l2_new,l3_new)
    assert 0.9999999 <= sum([l0_new, l1_new, l2_new, l3_new]) <= 1.0
    assert equals_with(l3_new, l3*0.1, 0.001)
    assert equals_with(l2_new, l2 + abs(l3_new - l3)/3, 0.001)
    assert equals_with(l1_new, l1 + abs(l3_new - l3)/3, 0.001)
    assert equals_with(l0_new, l0 + abs(l3_new - l3)/3, 0.001)

def test_boost_by():
    l0,l1,l2,l3 = [0.25,0.25,0.25,0.25]
    l0_new,l1_new,l2_new,l3_new = boost_by(l0, l1, l2, l3, 0.1)
    print(l0_new,l1_new,l2_new,l3_new)
    assert 0.9999999 <= sum([l0_new, l1_new, l2_new, l3_new]) <= 1.0

    # add 0.1 of its diff with 1 instead of just "set"
    assert equals_with(l3_new, l3 + (1 - l3)*0.1, 0.001)
    assert equals_with(l2_new, l2 - (l3_new - l3)/3, 0.001)
    assert equals_with(l1_new, l1 - (l3_new - l3)/3, 0.001)
    assert equals_with(l0_new, l0 - (l3_new - l3)/3, 0.001)
