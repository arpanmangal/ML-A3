"""
The main controller
"""


import sys
import time
import numpy as np
import itertools
from read import preprocess_data, read_data, read_cont_data, one_hot_data
from model import DecisionTree
from plot import make_acc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    part = sys.argv[1]
    if (part == 'a0'):
        preprocess_data ('data/credit-card/credit-cards.val.csv', 'data/credit-card/credit-cards.val.processed')
        preprocess_data ('data/credit-card/credit-cards.test.csv', 'data/credit-card/credit-cards.test.processed')
        preprocess_data ('data/credit-card/credit-cards.train.csv', 'data/credit-card/credit-cards.train.processed')

    elif (part == 'a' or part == 'b'):
        data = read_data ('data/credit-card/credit-cards.train.processed')
        valData = read_data ('data/credit-card/credit-cards.val.processed')
        testData = read_data ('data/credit-card/credit-cards.test.processed')
        
        features = set()
        for x in range(1, 24):
            features.add(x)
            
        trainAccuracies = []
        valAccuracies = []
        testAccuracies = []
        numNodes = []
        remainders = [r for r in range(23, -1, -1)]# [23, 20, 15, 10, 5, 0]
        pruning = False if part == 'a' else True
        fileName = "Q1/plots/accuracies.png" if part == 'a' else "Q1/plots/pruning.png"

        for r in remainders:
            start_time = time.time()
            DT = DecisionTree ()
            Tree, num_nodes, correct_preds, train_cps, test_cps = DT.grow_tree (data, valData, testData, features, pruning=pruning, remainder=r)
            end_time = time.time()

            trAcc = 100 * train_cps / len(data)
            valAcc = 100 * correct_preds / len(valData)
            ttAcc = 100 * test_cps / len(testData)

            trainAccuracies.append(trAcc)
            valAccuracies.append(valAcc)
            testAccuracies.append(ttAcc)
            numNodes.append(num_nodes)
            print (num_nodes, "%.2f %.2f %.2f" % (trAcc, valAcc, ttAcc))
            print ("Time Taken: %.2f secs" % (end_time - start_time))

        make_acc_curve (numNodes, trainAccuracies, valAccuracies, testAccuracies, fileName=fileName)


    elif (part == 'c'):
        data = read_cont_data ('data/credit-card/credit-cards.train.csv')
        valData = read_cont_data ('data/credit-card/credit-cards.val.csv')
        testData = read_cont_data ('data/credit-card/credit-cards.test.csv')

        features = set()
        for x in range(1, 24):
            features.add(x)

        trainAccuracies = []
        valAccuracies = []
        testAccuracies = []
        numNodes = []
        depths = [0, 2, 4, 6, 8, 10, 12, 16, 20]
        pruning = False
        fileName = "Q1/plots/continuous.png"

        for d in depths:
            start_time = time.time()
            DT = DecisionTree ()
            Tree, num_nodes, correct_preds, train_cps, test_cps = DT.grow_tree (data, valData, testData, features, continuous=True, depth=d)
            end_time = time.time()

            trAcc = 100 * train_cps / len(data)
            valAcc = 100 * correct_preds / len(valData)
            ttAcc = 100 * test_cps / len(testData)

            trainAccuracies.append(trAcc)
            valAccuracies.append(valAcc)
            testAccuracies.append(ttAcc)
            numNodes.append(num_nodes)
            print (num_nodes, "%.2f %.2f %.2f" % (trAcc, valAcc, ttAcc))
            print ("Time Taken: %.2f secs" % (end_time - start_time))

        make_acc_curve (numNodes, trainAccuracies, valAccuracies, testAccuracies, fileName=fileName)


    elif (part == 'd' or part == 'e'):
        if (part == 'd'):
            data = read_data ('data/credit-card/credit-cards.train.processed')
            valData = read_data ('data/credit-card/credit-cards.val.processed')
            testData = read_data ('data/credit-card/credit-cards.test.processed')
        else:
            # Sparsity <= one-hot
            data = one_hot_data ('data/credit-card/credit-cards.train.processed')
            valData = one_hot_data ('data/credit-card/credit-cards.val.processed')
            testData = one_hot_data ('data/credit-card/credit-cards.test.processed')
 
        maxAcc = 0
        maxPara = None
        max_depths = [1, 5, 6, 7, 8, 9]
        min_samples_splits = [2, 3, 4]
        min_samples_leafs = [1, 5, 20, 50, 54, 55, 56, 57, 59, 60, 61, 62]
        for max_depth, min_samples_split, min_samples_leaf in list(itertools.product(*[max_depths, min_samples_splits, min_samples_leafs])):
            clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            clf.fit (data[:,:-1],data[:,-1])
            trainScore = 100 * clf.score (data[:,:-1], data[:,-1])
            valScore = 100 * clf.score (valData[:,:-1], valData[:,-1])
            testScore = 100 * clf.score (testData[:,:-1], testData[:,-1])
            print ("%d %d %d | %.3f %.3f %.3f" % (max_depth, min_samples_split, min_samples_leaf, trainScore, valScore, testScore))
            if (valScore > maxAcc):
                maxAcc = valScore
                maxPara = (max_depth, min_samples_split, min_samples_leaf)
        
        print (maxPara)
        clf = DecisionTreeClassifier(random_state=0, max_depth=maxPara[0], min_samples_split=maxPara[1], min_samples_leaf=maxPara[2])
        clf.fit (data[:,:-1],data[:,-1])
        trainScore = 100 * clf.score (data[:,:-1], data[:,-1])
        valScore = 100 * clf.score (valData[:,:-1], valData[:,-1])
        testScore = 100 * clf.score (testData[:,:-1], testData[:,-1])
        print ("%.3f %.3f %.3f" % (trainScore, valScore, testScore))


    elif (part == 'f'):
        data = read_data ('data/credit-card/credit-cards.train.processed')
        valData = read_data ('data/credit-card/credit-cards.val.processed')
        testData = read_data ('data/credit-card/credit-cards.test.processed')
        # data = one_hot_data ('data/credit-card/credit-cards.train.processed')
        # valData = one_hot_data ('data/credit-card/credit-cards.val.processed')
        # testData = one_hot_data ('data/credit-card/credit-cards.test.processed')
        
        maxAcc = 0
        maxPara = None
        n_estimators = [5, 6, 7, 8, 10]
        max_features = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
        bootstraps = [True, False]
        max_depths = [2, 3, 4, 5, 6]

        for n_est, max_f, bootstrap, max_depth in list (itertools.product(*[n_estimators, max_features, bootstraps, max_depths])):
            clf = RandomForestClassifier (n_estimators=n_est, max_features=max_f, bootstrap=bootstrap, max_depth=max_depth, random_state=0)
            clf.fit (data[:,:-1], data[:,-1])
            trainScore = 100 * clf.score (data[:,:-1], data[:,-1])
            valScore = 100 * clf.score (valData[:,:-1], valData[:,-1])
            testScore = 100 * clf.score (testData[:,:-1], testData[:,-1])
            print ("%d %d %d %d | %.3f %.3f %.3f" % (n_est, max_f, bootstrap, max_depth, trainScore, valScore, testScore)) 
            if (valScore > maxAcc):
                maxAcc = valScore
                maxPara = (n_est, max_f, bootstrap, max_depth)

        print (maxPara)
        clf = RandomForestClassifier (n_estimators=maxPara[0], max_features=maxPara[1], bootstrap=maxPara[2], max_depth=maxPara[3], random_state=0)
        clf.fit (data[:,:-1], data[:,-1])
        trainScore = 100 * clf.score (data[:,:-1], data[:,-1])
        valScore = 100 * clf.score (valData[:,:-1], valData[:,-1])
        testScore = 100 * clf.score (testData[:,:-1], testData[:,-1])
        print ("%.3f %.3f %.3f" % (trainScore, valScore, testScore)) 

    elif (part == 't'):
        X = [0, 1, 2, 3, 4, 5]
        Y1 = [0, 0.5, 1, 1.5, 2, 2.5]
        Y2 = [0, 1, 2, 3, 4, 5]
        Y3 = [0, 1, 4, 9, 16, 25]
        make_acc_curve (X, Y1, Y2, Y3)
    else:
        print ("Go Away")
