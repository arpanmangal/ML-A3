"""
Code for generating the Decision-Tree
"""

import numpy as np

class Node:
    def __init__ (self, distribution, feature, feature_values):
        self.dist = distribution
        self.feature = feature
        self.fv = feature_values
        self.children = []

    def add_child (self, child):
        self.children.append(child)

    def print_subtree (self, root=False):
        # Print this and the below node
        if root:
            print (self.feature, self.dist, ' ||| ')
        
        for c in self.children:
            print (c.feature, c.dist, ' | ')
        for c in self.children:
            c.print_subtree()

class DecisionTree:
    def __init__(self):
        # self.data = data
        self.features_values = self.feature_value_array()
        self.tree = None

        self.features = set()
        for x in range(1, 24):
            self.features.add(x)

    def grow_tree (self, data):
        # Last col of data is Y
        # Find the distribution
        class0 = np.sum(data[:,-1] == 0)
        class1 = np.sum(data[:,-1] == 1)
        distribution = [class0 / len(data) , class1 / len(data)]
        print (distribution)

        if (len(self.features) <= 23):
            # Stop growing the tree and make this as Leaf node
            return Node (distribution, None, None)

        # Grow the tree
        best_f = self.find_best_feature ()
        self.features.remove(best_f)

    
    def find_best_feature (self):
        return 0

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
