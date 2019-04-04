"""
Reading and Preprocessing module
"""

import numpy as np
from tqdm import tqdm


def read_data (datafile):
    # Read the processed data
    data = []
    for line in open(datafile):
        entries = [int(e) for e in line.strip('\n').split(' ')]
        data.append(entries)
    data = np.array(data)
    return data
    

def read_cont_data (datafile):
    # Read the unprocessed data
    data = []
    for line in tqdm(read_raw_data(datafile)):
        data.append(line)
    data = np.array(data)
    return data

def preprocess_data (datafile, outfile):
    data = []
    for line in tqdm(read_raw_data(datafile)):
        data.append(line)

    data = np.array(data)
    X = data[:,:-1]
    Y = data[:,-1:]

    # Replace the continuous features with binary features
    X1median = np.median(X[:,0])
    X[:,0] = (X[:,0] > X1median).astype(int)

    X5median = np.median(X[:,4])
    X[:,4] = (X[:,4] > X5median).astype(int)

    for vbl in range(12, 24):
        median = np.median(X[:,vbl-1])
        X[:,vbl-1] = (X[:,vbl-1] > median).astype(int)

    ofile = open(outfile, 'w')
    for x, y in zip(X, Y):
        ofile.write(' '.join(str(v) for v in x))
        ofile.write(' ')
        ofile.write(str(int(y)))
        ofile.write('\n')


def one_hot_data (datafile):
    data = []
    for line in open(datafile):
        val = [int(v) for v in line.split(' ')]
        # print (val)
        one_hot_vals = val[:2]
        one_hot_vals += one_hot(val[2], 0, 6)
        one_hot_vals += one_hot(val[3], 0, 3)
        one_hot_vals.append (val[4])
        for x in range(6, 12):
            one_hot_vals += one_hot(val[x-1], -2, 9)
        for x in range(12, 24):
            one_hot_vals.append (val[x-1])

        # print (one_hot_vals)
        data.append(one_hot_vals)
        # exit(0)
    return np.array(data)

def one_hot(v, minv, maxv):
    size = maxv - minv + 1
    vals = [0] * size
    vals[v - minv] = 1
    return vals

def read_raw_data (datafile):
    firstTwo = 2
    for line in open(datafile):
        if (firstTwo > 0):
            firstTwo -= 1
            continue
        
        yield [int(x) for x in line.strip('\n').split(',')][1:]
