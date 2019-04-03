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
        

    def grow_tree (self, data, immutable_features):
        # Last col of data is Y
        # Find the distribution
        features = immutable_features.copy()
        class0 = np.sum(data[:,-1] == 0)
        class1 = np.sum(data[:,-1] == 1)
        if (class0 == 0 or class1 == 0):
            # Make this a leaf node
            if (len(data) == 0):
                return Node ([0.5, 0.5], None, None)
            elif (class0 == 0):
                return Node ([0.0, 1.0], None, None)
            else:
                return Node ([1.0, 0.0], None, None)

        distribution = [class0 / len(data) , class1 / len(data)]

        if (len(features) <= 0):
            # Stop growing the tree and make this as Leaf node
            return Node (distribution, None, None)

        # Grow the tree
        best_f = self.find_best_feature (data, features)
        features.remove(best_f)
        fv = self.features_values[best_f - 1]
        seperated_data = self.partition_data (data, best_f, fv)

        SubTree = Node (distribution, best_f, fv)
        for sdata in seperated_data:
            SubTree.add_child (self.grow_tree (sdata, features))

        return SubTree

    
    def find_best_feature (self, data, features):
        class0 = np.sum(data[:,-1] == 0)
        class1 = np.sum(data[:,-1] == 1)
        distribution = [class0 / len(data) , class1 / len(data)]

        fentropies = {}
        for f in features:
            fv = self.features_values[f - 1]
            entropy = 0

            for v in fv:
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

    def evaluate (self, Tree, data):
        # Evaluate the tree on the dataset
        num_nodes, correct_preds = self.evaluate_subtree (Tree, data)
        print (num_nodes, correct_preds)

        return correct_preds / len(data)
        

    def evaluate_subtree (self, subtree, data):
        # Return number of correct predictions
        if (subtree.feature == None):
            # This is a leaf node
            return 1, np.sum(data[:,-1] == subtree.prediction)

        correct_preds = 0
        num_nodes = 0
        seperated_data = self.partition_data (data, subtree.feature, subtree.fv)
        assert (len(seperated_data) == len(subtree.get_children()))
        for sdata, child in zip(seperated_data, subtree.get_children()):
            nn, cp = self.evaluate_subtree (child, sdata)
            num_nodes += nn
            correct_preds += cp

        return num_nodes, correct_preds


    def partition_data (self, data, best_f, fv):
        # Partition the data based on best_f for it's various fv
        seperated_data = []
        for fvals in fv:
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
