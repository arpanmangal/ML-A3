"""
The main controller
"""


import sys
import numpy as np
from read import preprocess_data, read_data
from model import DecisionTree
import time


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
        valData = read_data ('data/credit-card/credit-cards.val.processed')
        testData = read_data ('data/credit-card/credit-cards.test.processed')
        
        features = set()
        for x in range(1, 24):
            features.add(x)
            
        start_time = time.time()
        DT = DecisionTree ()
        Tree = DT.grow_tree (data, features)
        end_time = time.time()

        print ("Time Taken: %.2f secs" % (end_time - start_time))

        print ("Training Accuracy: %.2f %%" % (100 * DT.evaluate (Tree, data)))
        print ("Validation Accuracy: %.2f %%" % (100 * DT.evaluate (Tree, valData)))
        print ("Test Accuracy: %.2f %%" % (100 * DT.evaluate (Tree, testData)))
        # Tree.print_subtree (root=True)

    else:
        print ("Go Away")
