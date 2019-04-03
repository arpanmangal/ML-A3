"""
The main controller
"""


import sys
import numpy as np
from read import preprocess_data, read_data
from model import DecisionTree


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'a0'):
        preprocess_data ('data/credit-card/credit-cards.val.csv', 'data/credit-card/credit-cards.val.processed')
        preprocess_data ('data/credit-card/credit-cards.test.csv', 'data/credit-card/credit-cards.test.processed')
        preprocess_data ('data/credit-card/credit-cards.train.csv', 'data/credit-card/credit-cards.train.processed')

    elif (sys.argv[1] == 'a'):
        data = read_data ('data/credit-card/credit-cards.train.processed')
        DT = DecisionTree ()
        Tree = DT.grow_tree (data)
        Tree.print_subtree (root=True)

    else:
        print ("Go Away")
