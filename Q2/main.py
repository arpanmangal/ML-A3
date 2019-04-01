"""
Main Controller
"""


import sys
from read import *
from nnetwork import NNetwork
from network import Network #!!!!!!!!!!!!!!!!!!!!!
from plot import make_confusion_matrix

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'a'):
        # Convert to the one-hot encoding
        gen_one_hot_data ('data/poker/poker-hand-training.data', 'data/poker/train.data')

    if (sys.argv[1] == 'b'):
        # Read the data
        data = read_one_hot_data ('data/poker/train.data')
        # print (X.shape, Y.shape)

        # Train the neural network
        NNet = NNetwork (85, [20, 20], 10, 100)
        predictions = NNet.predict (data)
        print (predictions)
        Y = NNet.getTrueY(data)
        print (Y[:10])
        print (Y)
        # exit(0)
        accuracy = NNet.accuracy (Y, predictions)
        print ("Accuracy: ", accuracy)
        # make_confusion_matrix (Y, predictions)
        # exit(0)
        # NNet.train(data, 10, 0.03)
        NNet.train(data, 10, 0.1)
        exit(0)

    if (sys.argv[1] == 'c'):
        # Read the data
        data = read_one_hot_data ('data/poker/train.data')
        # print (X.shape, Y.shape)
        # data = [(x.reshape(85, 1), vectorized_result(y)) for x, y in zip (X, Y)]

        # Train the neural network
        Net = Network ([85, 20, 20, 10])
        accuracy = Net.evaluate(data) / len(data)
        print (accuracy)
        Net.SGD(data, 10, 100, 0.1, data)

