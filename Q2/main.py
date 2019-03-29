"""
Main Controller
"""


import sys
from read import *


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'a'):
        # Convert to the one-hot encoding
        gen_one_hot_data ('data/poker/poker-hand-training.data', 'data/poker/train.data')