"""
Module for reading in the data
"""

import numpy as np
from tqdm import tqdm


def gen_one_hot_data (datafile, onehotfile):
    """
    Convert the given data to one hot encodings
    """
    outfile = open(onehotfile, 'w')
    count = np.zeros(10)
    for data in tqdm(read_raw_data (datafile)):
        # First write all the suits
        for S in range(5):
            encoding = convert_one_hot(data[S * 2], 4)
            for e in encoding:
                outfile.write(str(e))
                outfile.write(' ')

        # Write all the ranks
        for S in range(5):
            encoding = convert_one_hot(data[S * 2 + 1], 13)
            for e in encoding:
                outfile.write(str(e))
                outfile.write(' ')

        # Write the class
        outfile.write(str(data[10]))
        outfile.write(' ')

        outfile.write('\n')
        count[data[10]] += 1

    np.set_printoptions(suppress=True)
    print (count)

def read_one_hot_data (datafile):
    """
    Read the one hot data
    """
    X = []
    Y = []
    data = []
    for line in tqdm(read_raw_data (datafile, delimeter=' ')):
        data.append(line)
        # X.append(data[:-1])
        # Y.append(data[-1])

    data = np.array(data)
    np.random.seed(0) # Deterministically random
    np.random.shuffle (data)

    X = data[:, :-1]
    Y = data[:, -1]

    print (X)
    print (Y)
    return X, Y


def read_raw_data (datafile, delimeter=','):
    """
    Read the raw decimal data
    generator
    """

    for line in open (datafile):
        data = ([int(x) for x in line.replace(' \n', '\n').strip('\n').split(delimeter)])
        yield data


def convert_one_hot (num, range):
    """
    Convert num in range 1-range to 1-hot encoding
    """
    encoding = [0]*range
    encoding[num - 1] = 1
    return encoding
