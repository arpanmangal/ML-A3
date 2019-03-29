"""
Module for reading in the data
"""

import numpy as np


def gen_one_hot_data (datafile, onehotfile):
    """
    Convert the given data to one hot encodings
    """
    outfile = open(onehotfile, 'w')
    for data in read_raw_data (datafile):
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
        encoding = convert_one_hot(data[10] + 1, 10)
        for e in encoding:
            outfile.write(str(e))
            outfile.write(' ')

        outfile.write('\n')
    return []


def read_one_hot_data (datafile):
    """
    Read the one hot data
    """

    return []


def read_raw_data (datafile):
    """
    Read the raw decimal data
    generator
    """

    for line in open (datafile):
        data = ([int(x) for x in line.strip('\n').split(',')])
        yield data


def convert_one_hot (num, range):
    """
    Convert num in range 1-range to 1-hot encoding
    """
    encoding = [0]*range
    encoding[num - 1] = 1
    return encoding
