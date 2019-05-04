from util import entropy, information_gain, partition_classes
import numpy as np
import ast

class TreeNode(object):
    def __init__(self):
        self.split_attribute = None
        self.split_value = None
        self.left = None
        self.right = None

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary
        self.tree = {}

    def select_split(self, X, y):
        """
        Finds the best split attribute and value to maximize information gain.
        :param X: Numpy array of data
        :param y: Numpy array of classification
        :return: split_attr, split_val, split_data
        """
        max_info_gain = 0
        split_attr = 0
        split_val = None
        split_data = {'X_left': None, 'X_right': None,
                      'y_left': None, 'y_right': None}

        # Loop through each value and find the split that will maximize information gain.
        for row_idx in range(len(X)):
            for col_idx in range(len(X[0])):
                temp_attr = col_idx
                temp_val = X[row_idx][col_idx]
                temp_split_data = partition_classes(X, y, temp_attr, temp_val)
                temp_info_gain = information_gain(y, [temp_split_data[2], temp_split_data[3]])

                if temp_info_gain > max_info_gain:
                    max_info_gain = temp_info_gain
                    split_attr = temp_attr
                    split_val = temp_val
                    split_data['X_left'] = temp_split_data[0]
                    split_data['X_right'] = temp_split_data[1]
                    split_data['y_left'] = temp_split_data[2]
                    split_data['y_right'] = temp_split_data[3]

        return split_attr, split_val, split_data

    def learn(self, X, y):
        self.tree = self.learn_helper(X, y)
        return self.tree

    def learn_helper(self, X, y):
        if len(set(y)) == 1:
            return y[0]

        tree = {}
        split_attr, split_val, split_data = self.select_split(X, y)
        left_node = self.learn_helper(split_data['X_left'], split_data['y_left'])
        right_node = self.learn_helper(split_data['X_right'], split_data['y_right'])
        tree[split_attr] = {'split_val': split_val, 'left': left_node, 'right': right_node}

        return tree

    def classify(self, record):
        node = self.tree

        # If node is a dict, then it is not a leaf node
        while isinstance(node, dict):
            feat = list(node)[0]
            is_categorical = isinstance(record[feat], str)

            # Split record based off split value of tree node
            if is_categorical:
                if record[feat] == node[feat]['split_val']:
                    node = node[feat]['left']
                else:
                    node = node[feat]['right']
            else:
                if record[feat] <= node[feat]['split_val']:
                    node = node[feat]['left']
                else:
                    node = node[feat]['right']

        return node


if __name__=="__main__":

    X = [[3, 'aa', 10],
         [1, 'bb', 22],
         [2, 'cc', 28],
         [5, 'bb', 32],
         [4, 'cc', 32]]

    y = [1, 1, 0, 0, 1]

    t = DecisionTree()
    t.learn(X, y)
    a = [3, 'aa', 23]
    t.classify(a)
    print(t.classify(a))
