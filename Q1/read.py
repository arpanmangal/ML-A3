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


def read_raw_data (datafile):
    firstTwo = 2
    for line in open(datafile):
        if (firstTwo > 0):
            firstTwo -= 1
            continue
        
        yield [int(x) for x in line.strip('\n').split(',')][1:]
