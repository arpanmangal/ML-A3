"""
Main Controller
"""


import sys
from read import *
from nnetwork import NNetwork
from plot import make_confusion_matrix


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'a'):
        # Convert to the one-hot encoding
        gen_one_hot_data ('data/poker/poker-hand-training.data', 'data/poker/train.data')

    if (sys.argv[1] == 'b'):
        # Read the data
        X, Y = read_one_hot_data ('data/poker/train.data')
        print (X.shape, Y.shape)

        # Train the neural network
        NNet = NNetwork (85, [20, 20], 10, 100)
        # NNet.train (X, Y)
        predictions = NNet.predict (X)
        print (predictions)
        print (Y)
        accuracy = NNet.accuracy (Y, predictions)
        print ("Accuracy: ", accuracy)
        # make_confusion_matrix (Y, predictions)
        NNet.train(X, Y, 5, 0.1)