from scipy import stats
import numpy as np


# This method computes entropy for information gain
def entropy(class_y):
    # Input:
    #   class_y         : list of class labels (0's and 1's)

    # Example:
    #    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

    entropy = 0

    if len(class_y) == 0:
        return 0

    class_y = list(class_y)

    # count zeroes and ones
    zeroes = class_y.count(0)
    ones = class_y.count(1)

    if zeroes == 0 or ones == 0:
        return 0

    # implement entropy formula
    total = len(class_y)
    p0 = float(zeroes) / total
    p1 = float(ones) / total
    entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))

    return entropy


def partition_classes(X, y, split_attribute, split_val):
    # Inputs:
    #   X               : data containing all attributes
    #   y               : labels
    #   split_attribute : column index of the attribute to split on
    #   split_val       : either a numerical or categorical value to divide the split_attribute
    X_left = []
    X_right = []

    y_left = []
    y_right = []

    # Check if numerical or categorical
    is_categorical = isinstance(split_val, str)

    if is_categorical:
        for index, row in enumerate(X):
            if row[split_attribute] == split_val:
                X_left.append(X[index])
                y_left.append(y[index])
            else:
                X_right.append(X[index])
                y_right.append(y[index])
    else:
        for index, row in enumerate(X):
            if row[split_attribute] <= split_val:
                X_left.append(X[index])
                y_left.append(y[index])
            else:
                X_right.append(X[index])
                y_right.append(y[index])

    return X_left, X_right, y_left, y_right


def information_gain(previous_y, current_y):
    # Inputs:
    #   previous_y: the distribution of original labels (0's and 1's)
    #   current_y:  the distribution of labels after splitting based on a particular
    #               split attribute and split value
    """
    Example:

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]

    info_gain = 0.45915
    """
    num_labels = len(previous_y)

    prev_entropy = entropy(previous_y)
    average_entropy = 0

    for row in current_y:
        cond_entropy = entropy(row)
        prob = len(row) / num_labels
        average_entropy += cond_entropy * prob

    info_gain = prev_entropy - average_entropy

    return info_gain


if __name__=="__main__":

    print(entropy([0, 0, 0, 1, 1, 1, 1, 1, 1]))

    X = [[3, 'aa', 10],
         [2, 'cc', 28],
         [1, 'bb', 22],
         [5, 'bb', 32],
         [4, 'cc', 32]]

    y = [1, 1, 0, 0, 1]
    X_left, X_right, y_left, y_right = partition_classes(X, y, 0, 3)
    print(X_left, X_right)
    print(y_left, y_right)

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]
    print(information_gain(previous_y, current_y))
