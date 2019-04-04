"""
Code for generating the Decision-Tree
"""

import numpy as np

class Node:
    def __init__ (self, distribution, feature, feature_values):
        self.dist = distribution
        self.prediction = np.argmax(distribution)
        self.feature = feature
        self.fv = feature_values
        self.children = []

    def add_child (self, child):
        self.children.append(child)

    def get_children (self):
        return self.children

    def print_subtree (self, root=False):
        # Print this and the below node
        if root:
            print (self.feature, self.dist, ' ||| ')
        
        for c in self.children:
            print (c.feature, c.dist, ' | ')
        print (' ||| ')
        for c in self.children:
            c.print_subtree()


class DecisionTree:
    def __init__(self):
        self.features_values = self.feature_value_array()
        self.tree = None
        

    def grow_tree (self, data, valData, testData, immutable_features, pruning=False, remainder=10, continuous=False, depth=5):
        # Last col of data is Y
        # Find the distribution
        features = immutable_features.copy()
        class0 = np.sum(data[:,-1] == 0)
        class1 = np.sum(data[:,-1] == 1)
        if (class0 == 0 or class1 == 0):
            # Make this a leaf node
            if (len(data) == 0):
                distribution = [0.5, 0.5]
            elif (class0 == 0):
                distribution = [0.0, 1.0]
            else:
                distribution = [1.0, 0.0]

            node = Node (distribution, None, None)
            return node, 1, np.sum(valData[:,-1] == node.prediction), np.sum(data[:,-1] == node.prediction), np.sum(testData[:,-1] == node.prediction)

        distribution = [class0 / len(data) , class1 / len(data)]

        node = Node (distribution, None, None)
        present_correct_preds = np.sum(valData[:,-1] == node.prediction)
        present_train_cps = np.sum(data[:,-1] == node.prediction)
        present_test_cps = np.sum(testData[:,-1] == node.prediction)
        if ((not continuous and len(features) <= remainder) or (continuous and depth == 0)):
            # Stop growing the tree and make this as Leaf node
            return node, 1, present_correct_preds, present_train_cps, present_test_cps

        # Grow the tree
        correct_preds = 0
        correct_train_ps = 0
        correct_test_ps = 0
        num_nodes = 0
        best_f = self.find_best_feature (data, features, continuous)
        if not continuous:
            features.remove(best_f)
        fv = self.features_values[best_f - 1]
        seperated_data = self.partition_data (data, best_f, fv, continuous=continuous)
        seperated_val_data = self.partition_data (valData, best_f, fv, continuous=continuous)
        seperated_test_data = self.partition_data (testData, best_f, fv, continuous=continuous)

        SubTree = Node (distribution, best_f, fv)
        for sdata, svdata, stdata in zip(seperated_data, seperated_val_data, seperated_test_data):
            childTree, nn, corr_pred, c_train_pred, c_test_pred = self.grow_tree (sdata, svdata, stdata, features, pruning=pruning, remainder=remainder, continuous=continuous, depth=depth-1)
            num_nodes += nn
            correct_preds += corr_pred
            correct_train_ps += c_train_pred
            correct_test_ps += c_test_pred
            SubTree.add_child (childTree)

        if (pruning is True and present_correct_preds >= correct_preds):
            # Prune this subtree
            # print ("Pruning: ", present_correct_preds, correct_preds, best_f)
            return node, 1, present_correct_preds, present_train_cps, present_test_cps

        return SubTree, num_nodes, correct_preds, correct_train_ps, correct_test_ps

    
    def find_best_feature (self, data, features, continuous):
        class0 = np.sum(data[:,-1] == 0)
        class1 = np.sum(data[:,-1] == 1)
        distribution = [class0 / len(data) , class1 / len(data)]

        fentropies = {}
        for f in features:
            fv = self.features_values[f - 1]
            entropy = 0

            for v in fv:
                if continuous and ((f == 1) or (f == 5) or (f >= 12 and f <= 23)):
                    median = np.median(data[:,f-1])
                    dat = (data[:,f-1] > median).astype(int)
                    classv = np.sum(dat ==  v)
                    if (classv == 0):
                        continue
                    classv0 = np.sum((dat == v) & (data[:,-1] == 0))
                    classv1 = np.sum((dat == v) & (data[:,-1] == 1))
                    xprob = classv / len(data)
                    distribution = [classv0 / classv, classv1 / classv]
                else:
                    classv = np.sum(data[:,f-1] == v)
                    if (classv == 0):
                        continue
                    classv0 = np.sum((data[:,f-1] == v) & (data[:,-1] == 0))
                    classv1 = np.sum((data[:,f-1] == v) & (data[:,-1] == 1))
                    xprob = classv / len(data)
                    distribution = [classv0 / classv, classv1 / classv]

                entropy += xprob * self.calc_entropy (distribution)
            fentropies[f] = entropy

        best_f = min(fentropies, key=fentropies.get)
        return best_f


    def partition_data (self, data, best_f, fv, continuous):
        # Partition the data based on best_f for it's various fv
        seperated_data = []
        # if continuous and ((best_f == 1) or (best_f == 5) or (best_f >= 12 and best_f <= 23)):
        #     median = np.median(data[:,best_f-1])
        #     mData = (data[:,best_f-1] > median).astype(int)
        # else:
        #     mData = data
        # if (len(data) == 0):
        #     return seperated_data

        if (len(data) == 0):
            for fvals in fv:
                condition = (data[:,best_f-1] == fvals)
                sdata = data[condition]
                seperated_data.append(sdata)
            return seperated_data

        for fvals in fv:
            if continuous and ((best_f == 1) or (best_f == 5) or (best_f >= 12 and best_f <= 23)):
                median = np.median(data[:,best_f-1])
                dat = (data[:,best_f-1] > median).astype(int)
                condition = (dat == fvals)
            else:
                condition = (data[:,best_f-1] == fvals)
            sdata = data[condition]
            seperated_data.append(sdata)
        return seperated_data

    def feature_value_array (self):
        # For each feature push it's possible values
        fv = []
        fv.append ([0, 1]) # X1
        fv.append ([1, 2]) # X2
        fv.append ([v for v in range(7)]) # X3
        fv.append ([v for v in range(4)]) # X4
        fv.append ([0, 1]) # X5
        for f in range(6, 12):
            fv.append ([v for v in range(-2, 10)]) # X6-11
        for f in range(12, 24):
            fv.append ([0, 1]) # X12-X17 and X18-X23

        return fv

    def calc_entropy (self, distribution):
        entropy = 0
        for y in distribution:
            if y == 0:
                continue
            entropy += y * np.log2(y)
        return -entropy
