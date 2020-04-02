"""
Robert Wyckoff
CS 4435
04/01/2020
Project 1

This file contains function definitions at the top, and then calls the DTvsRFvsNB function at the bottom.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os
from math import floor
from functools import wraps
from time import time


def time_func(func):
    """
    A decorator function that wraps other functions to return both the result of that function and the run-time
    of that function.
    :param func: The function to be wrapped and time-measured.
    :return: The result of the wrapped function and the time it took to run that function.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        return result, time_end - time_start

    return wrap


def split_into_arrays(training_file, test_file):
    """
    Takes two comma-delimited text files: one training set and one test set, and generates numpy arrays from them,
    then splits those arrays into training and testing class labels and data.
    :param training_file: The comma-delimited text file holding the training data. The class labels for each example
    are the last element in each vector.
    :param test_file: The comma-delimited text file holding the test data.
    :return: The sliced arrays consisting of the attributes of the training (X_train) and test (X_test) files and the
    class labels of the training (y_train) and test (y_test) files.
    """
    # Generate numpy arrays from the training and test files.
    training_arr = np.genfromtxt(fname=training_file, delimiter=',', dtype='U')
    test_arr = np.genfromtxt(fname=test_file, delimiter=',', dtype='U')

    # Slice the arrays to separate the attributes from the class labels.
    X_train = training_arr[:, :-1]  # Define the array of training set attributes.
    X_test = test_arr[:, :-1]  # Define the array of test set attributes.
    y_train = training_arr[:, -1]  # Define the array of training set labels (the last elements in each vector).
    y_test = test_arr[:, -1]  # Define the array of test set labels (the last elements in each vector).

    return X_train, X_test, y_train, y_test


def print_results(y_pred, y_test):
    """
    Prints the predictions and results from the model being applied on the test set. (Task 4)
    :param y_pred: The array of predictions made by the calling classifier.
    :param y_test: The array of true values.
    """
    for i in range(len(y_pred)):
        object_id = i + 1
        predicted_class = y_pred[i]
        true_class = y_test[i]
        if predicted_class == true_class:
            accuracy = 1
        else:
            accuracy = 0
        # Print according to the formatting specified in the project instructions.
        print('ID= {0:5d}, predicted= {1:3d}, true= {2:3d}, accuracy= {3:4.2f}\n'
              .format(object_id, predicted_class, true_class, accuracy))


def gen_train_test_sets():
    """
    Generates training and testing comma-delimited .txt files from original full datasets. Specifically for the
    splitting of the adult.data and crx.data datasets from the UCI ML repository.
    """
    # Initialize the paths to the original Census Income and Credit Approval datasets.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    census_raw_data_path = os.path.join(project_dir, 'Datasets', 'adult.data')
    credit_raw_data_path = os.path.join(project_dir, 'Datasets', 'crx.data')

    # Generate numpy arrays from the two datasets.
    census_arr = np.genfromtxt(fname=census_raw_data_path, delimiter=',', dtype='U')
    credit_arr = np.genfromtxt(fname=credit_raw_data_path, delimiter=',', dtype='U')

    # Randomly shuffle both datasets in-place, then split them at 80% for the training sets and 20% for the test sets,
    # effectively performing the splits using random sampling without replacement.
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


@time_func
def decision_tree(training_file, test_file):
    """
    Uses a decision tree classifier to train a model using the samples of the training file, then applies the model to
    classify the samples from the test file.
    :param training_file: The comma-delimited text file holding the training data. The class labels for each example
    are the last element in each vector.
    :param test_file: The comma-delimited text file holding the test data.
    :return: The accuracy, precision, and recall scores of the model as applied to the test data.
    """
    # Generate training and testing numpy arrays from the training and test files.
    X_train, X_test, y_train, y_test = split_into_arrays(training_file, test_file)

    # Encode the y class labels.
    encoded_y_test = LabelEncoder().fit_transform(y_test)
    encoded_y_train = LabelEncoder().fit_transform(y_train)

    # One-hot-encode the categorical values in the training set and the test set.
    ohe_X_train = OneHotEncoder().fit_transform(X_train).toarray()
    ohe_X_test = OneHotEncoder().fit_transform(X_test).toarray()

    # Reduce the dimensionality of the sets using truncated SVD, which is similar to PCA but does not center the data
    # before computing the SVD.
    reduced_X_train = TruncatedSVD(n_components=100).fit_transform(ohe_X_train)
    reduced_X_test = TruncatedSVD(n_components=100).fit_transform(ohe_X_test)

    # Build the decision tree classifier clf from the training set.
    model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=20)
    model.fit(reduced_X_train, encoded_y_train)

    # Predict the classes of the test set.
    y_predictions = model.predict(reduced_X_test)

    # Print the predictions and results from the model being applied on the test set.
    print_results(y_predictions, encoded_y_test)

    # Calculate accuracy, precision, and recall of the decision tree classifier.
    accuracy = accuracy_score(encoded_y_test, y_predictions)
    precision = precision_score(encoded_y_test, y_predictions)
    recall = recall_score(encoded_y_test, y_predictions)

    return accuracy, precision, recall


@time_func
def random_forest(training_file, test_file):
    """
    Uses a random forest ensemble classifier (an ensemble made of many decision trees where the final result is the
    mode of the results of all the individual trees) to train a model using the samples of the training file, then
    applies the model to classify the samples from the test file.
    :param training_file: The comma-delimited text file holding the training data. The class labels for each example
    are the last element in each vector.
    :param test_file: The comma-delimited text file holding the test data.
    :return: The accuracy, precision, and recall scores of the model as applied to the test data.
    """
    # Generate training and testing numpy arrays from the training and test files.
    X_train, X_test, y_train, y_test = split_into_arrays(training_file, test_file)

    # Encode the y class labels.
    encoded_y_test = LabelEncoder().fit_transform(y_test)
    encoded_y_train = LabelEncoder().fit_transform(y_train)

    # One-hot-encode the categorical values in the training set and the test set.
    ohe_X_train = OneHotEncoder().fit_transform(X_train).toarray()
    ohe_X_test = OneHotEncoder().fit_transform(X_test).toarray()

    # Reduce the dimensionality of the sets using truncated SVD, which is similar to PCA but does not center the data
    # before computing the SVD.
    reduced_X_Train = TruncatedSVD(n_components=100).fit_transform(ohe_X_train)
    reduced_X_test = TruncatedSVD(n_components=100).fit_transform(ohe_X_test)

    # Build the random forest ensemble classifier from the training set.
    model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=10, min_samples_split=20)
    model.fit(reduced_X_Train, encoded_y_train)

    # Predict the classes of the test set.
    y_predictions = model.predict(reduced_X_test)

    # Print the predictions and results from the model being applied on the test set.
    print_results(y_predictions, encoded_y_test)

    # Calculate accuracy, precision, and recall of the random forest classifier.
    accuracy = accuracy_score(encoded_y_test, y_predictions)
    precision = precision_score(encoded_y_test, y_predictions)
    recall = recall_score(encoded_y_test, y_predictions)

    return accuracy, precision, recall


@time_func
def naive_bayes(training_file, test_file):
    """
    Uses a Naive Bayes classifier to train a model using the samples of the training file, then applies the model
    to classify the samples from the test file.
    :param training_file: The comma-delimited text file holding the training data. The class labels for each example
    are the last element in each vector.
    :param test_file: The comma-delimited text file holding the test data.
    :return: The accuracy, precision, and recall scores of the model as applied to the test data.
    """
    # Generate training and testing numpy arrays from the training and test files.
    X_train, X_test, y_train, y_test = split_into_arrays(training_file, test_file)

    # Encode the y class labels.
    encoded_y_test = LabelEncoder().fit_transform(y_test)
    encoded_y_train = LabelEncoder().fit_transform(y_train)

    # Label-encode the categorical data into integers by transforming all the data with LabelEncoder.
    le_X_train = np.apply_along_axis(LabelEncoder().fit_transform, 0, X_train)
    le_X_test = np.apply_along_axis(LabelEncoder().fit_transform, 0, X_test)

    model = GaussianNB()
    model.fit(le_X_train, encoded_y_train)

    # Predict the classes of the test set
    y_predictions = model.predict(le_X_test)

    # Calculate accuracy of the NB classifier
    accuracy = accuracy_score(y_test, y_predictions)

    # Print the predictions and results from the model being applied on the test set.
    print_results(y_predictions, encoded_y_test)

    # Calculate accuracy, precision, and recall of the Naive Bayes classifier.
    accuracy = accuracy_score(encoded_y_test, y_predictions)
    precision = precision_score(encoded_y_test, y_predictions)
    recall = recall_score(encoded_y_test, y_predictions)

    return accuracy, precision, recall


def handle_missing_ca_data(training_file, test_file):
    # Read the training and test files into a pandas dataframe to make it easy to clean the data.
    training_df = pd.read_csv(training_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    # Combine the dataframes into one.
    df = pd.concat([training_df, test_df])

    drop_list = []
    for idx, row in df.iterrows():
        for i in range(0, 16):
            if row[i] is '?':
                drop_list.append(idx)

    df = df.drop(drop_list)

    # Randomly shuffle both datasets in-place, then split them at 80% for the training sets and 20% for the test sets,
    # effectively performing the splits using random sampling without replacement.
    training_df, test_df = train_test_split(df, test_size=0.2)

    # Save the new dataframes into new files.
    np.savetxt(os.path.join('No-Missing CA Data', 'Task6_credit_trainset.txt'), training_df, fmt='%s', delimiter=',')
    np.savetxt(os.path.join('No-Missing CA Data', 'Task6_credit_testset.txt'), test_df, fmt='%s', delimiter=',')

    # Conduct all three algorithms on the new datasets and get the accuracies, precisions, recalls, and run-times.
    (dt_accuracy, dt_precision, dt_recall), dt_running_time = \
        decision_tree(os.path.join('No-Missing CA Data', 'Task6_credit_trainset.txt'),
                      os.path.join('No-Missing CA Data', 'Task6_credit_testset.txt'))
    (rf_accuracy, rf_precision, rf_recall), rf_running_time = \
        random_forest(os.path.join('No-Missing CA Data', 'Task6_credit_trainset.txt'),
                      os.path.join('No-Missing CA Data', 'Task6_credit_testset.txt'))
    (nb_accuracy, nb_precision, nb_recall), nb_running_time = \
        naive_bayes(os.path.join('No-Missing CA Data', 'Task6_credit_trainset.txt'),
                    os.path.join('No-Missing CA Data', 'Task6_credit_testset.txt'))

    print("\n**************************\nResults after handling missing data:\n")
    print('\n\nDecision Tree (DT):\nAccuracy = {0:4.2f}\nPrecision = {1:4.2f}\nRecall = {2:4.2f}'
          .format(dt_accuracy, dt_precision, dt_recall))
    print('\n\nRandom Forest(RF):\nAccuracy = {0:4.2f}\nPrecision = {1:4.2f}\nRecall = {2:4.2f}'
          .format(rf_accuracy, rf_precision, rf_recall))
    print('\n\nNaive Bayes (NB):\nAccuracy = {0:4.2f}\nPrecision = {1:4.2f}\nRecall = {2:4.2f}'
          .format(nb_accuracy, nb_precision, nb_recall))

    print('\n\nRunning times:\nDecision Tree (DT) = {0:4.2f}\nRandom Forest(RF) = {1:4.2f}\nNaive Bayes(NB) = {2:4.2f}'
          .format(dt_running_time, rf_running_time, nb_running_time))


def DTvsRFvsNB(training_file, test_file):
    """
    The main function, which calls all three classification algorithms (Decision Tree, Random Forest, and Naive Bayes)
    on the given training and tes files and prints their overall results.
    :param training_file: The path name of the training file, where the training data is stored.
    :param test_file: The path name of the test file, where the test data is stored.
    """
    # Pass the training and test files to all three algorithms, getting their scores and running times.
    (dt_accuracy, dt_precision, dt_recall), dt_running_time = decision_tree(training_file, test_file)
    (rf_accuracy, rf_precision, rf_recall), rf_running_time = random_forest(training_file, test_file)
    (nb_accuracy, nb_precision, nb_recall), nb_running_time = naive_bayes(training_file, test_file)

    # Print the scores of all three algorithms.
    print('\n\nDecision Tree (DT):\nAccuracy = {0:4.2f}\nPrecision = {1:4.2f}\nRecall = {2:4.2f}'
          .format(dt_accuracy, dt_precision, dt_recall))
    print('\n\nRandom Forest(RF):\nAccuracy = {0:4.2f}\nPrecision = {1:4.2f}\nRecall = {2:4.2f}'
          .format(rf_accuracy, rf_precision, rf_recall))
    print('\n\nNaive Bayes (NB):\nAccuracy = {0:4.2f}\nPrecision = {1:4.2f}\nRecall = {2:4.2f}'
          .format(nb_accuracy, nb_precision, nb_recall))
    # Print the running time of the three algorithms
    print('\n\nRunning times:\nDecision Tree (DT) = {0:4.2f}\nRandom Forest(RF) = {1:4.2f}\nNaive Bayes(NB) = {2:4.2f}'
          .format(dt_running_time, rf_running_time, nb_running_time))


# ****************************************************************************************************************
# Program:

# The main function, where the parameters are the paths to the training set and the test set, respectively.
DTvsRFvsNB(os.path.join('Split Data', 'credit_trainset.txt'), os.path.join('Split Data', 'credit_testset.txt'))

# Handle the missing data in the Credit Approval dataset by ignoring tuples that have missing data, generating
# new training and test files, then running all three algorithms on those new training and test files.
handle_missing_ca_data(os.path.join('Split Data', 'credit_trainset.txt'),
                       os.path.join('Split Data', 'credit_testset.txt'))
