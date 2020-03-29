"""
Robert Wyckoff
CS 4435
04/01/2020
Project 1
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import os
from math import floor

# TODO: Consider placing all data-splitting code in a separate module so this main module just has the DTvsRFvsNB and
#  DT, RF, and NB functions. When in doubt, email the prof.

# Initialize the paths to the original Census Income and Credit Approval datasets.
project_dir = os.path.dirname(os.path.abspath(__file__))
census_raw_data_path = os.path.join(project_dir, 'Datasets', 'adult.data')
credit_raw_data_path = os.path.join(project_dir, 'Datasets', 'crx.data')

# Generate numpy arrays from the two datasets.
census_arr = np.genfromtxt(fname=census_raw_data_path, delimiter=',', dtype='U')
credit_arr = np.genfromtxt(fname=credit_raw_data_path, delimiter=',', dtype='U')

# Randomly shuffle both datasets in-place, then split them at 80% for the training sets and 20% for the test sets,
# effectively performing the splits using random sampling without replacement.
# TODO: Make this into a function?
np.random.shuffle(census_arr)
np.random.shuffle(credit_arr)
split = 0.8
census_split_idx = floor(len(census_arr) * split)
credit_split_idx = floor(len(credit_arr) * split)
census_trainset_arr = census_arr[:census_split_idx]
census_testset_arr = census_arr[census_split_idx:]
credit_trainset_arr = credit_arr[:credit_split_idx]
credit_testset_arr = credit_arr[credit_split_idx:]

# Save the train and test sets to comma-delimited .txt files for later use.
np.savetxt(os.path.join('Split Data', 'census_trainset.txt'), census_trainset_arr, fmt='%s', delimiter=',')
np.savetxt(os.path.join('Split Data', 'census_testset.txt'), census_testset_arr, fmt='%s', delimiter=',')
np.savetxt(os.path.join('Split Data', 'credit_trainset.txt'), credit_trainset_arr, fmt='%s', delimiter=',')
np.savetxt(os.path.join('Split Data', 'credit_testset.txt'), credit_testset_arr, fmt='%s', delimiter=',')


def decision_tree(training_file, test_file):
    """
    Uses a decision tree classifier to train a model using the samples of the training file, then applies the model to
    classify the samples from the test file.
    :param training_file: The comma-delimited text file holding the training data. The class labels for each example
    are the last element in each vector.
    :param test_file: The comma-delimited text file holding the test data.
    :return:
    """
    # Generate numpy arrays from the training and test files.
    training_arr = np.genfromtxt(fname=training_file, delimiter=',', dtype='U')
    test_arr = np.genfromtxt(fname=test_file, delimiter=',', dtype='U')

    # Slice the arrays to separate the attributes from the class labels.
    X_train = training_arr[:, :-1]  # Define the array of training set attributes.
    X_test = test_arr[:, :-1]  # Define the array of test set attributes.
    y_train = training_arr[:, -1]  # Define the array of training set labels (the last elements in each vector).
    y_test = test_arr[:, -1]  # Define the array of test set labels (the last elements in each vector).

    # One-hot-encode the categorical values in the training set.
    # TODO: Consider one-hot-encoding (OHE) and then PCA to reduce dimensionality.
    # TODO: sklearn's PCA may have an issue with sparse data. Use TruncatedSVD if it's an issue.
    ohe_arr = OneHotEncoder().fit_transform(X_train).toarray()
    print(ohe_arr.shape)

    # Build the decision tree classifier clf from the training set.
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    print(f"X-train: \n{X_train}")
    print(f"X-test: \n{X_test}")
    print(f"y-train: \n{y_train}")
    print(f"y-test: \n{y_test}")


def random_forest(training_file, test_file):
    pass


def naive_bayes(training_file, test_file):
    pass


def DTvsRFvsNB(training_file, test_file):
    pass


# ****************************************
# Test code:

decision_tree(os.path.join('Split Data', 'census_trainset.txt'), os.path.join('Split Data', 'census_testset.txt'))
decision_tree(os.path.join('Split Data', 'credit_trainset.txt'), os.path.join('Split Data', 'credit_testset.txt'))

# test print
# np.set_printoptions(precision=2, suppress=True)
# print(fake_arr)
