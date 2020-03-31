"""
Robert Wyckoff
CS 4435
04/01/2020
Project 1
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from math import floor


# TODO: Probably print the results along with the true values, etc, *after* training and predicting.
# TODO: Consider encapsulating the categorical label preprocessing into a separate function.
# TODO: Also consider encapsulating the printing (Task 4) functionality.

def gen_train_test_sets():
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

    # Encode the y class labels.
    encoded_y_test = LabelEncoder().fit_transform(y_test)
    encoded_y_train = LabelEncoder().fit_transform(y_train)

    # One-hot-encode the categorical values in the training set and the test set.
    ohe_X_train = OneHotEncoder().fit_transform(X_train).toarray()
    ohe_X_test = OneHotEncoder().fit_transform(X_test).toarray()
    print("One-hot-encoded!")
    print(ohe_X_train.shape)

    # Reduce the dimensionality of the sets using truncated SVD, which is similar to PCA but does not center the data
    # before computing the SVD.
    reduced_X_train = TruncatedSVD(n_components=100).fit_transform(ohe_X_train)
    reduced_X_test = TruncatedSVD(n_components=100).fit_transform(ohe_X_test)
    print("Reduced!")
    print(reduced_X_train.shape)

    # Build the decision tree classifier clf from the training set.
    model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=20)
    model.fit(reduced_X_train, encoded_y_train)
    print(f"Model: {model}")

    # Predict the classes of the test set.
    y_predictions = model.predict(reduced_X_test)
    print(f"Predictions: {y_predictions}")

    # Print the predictions and results from the model being applied on the test set.
    for i in range(len(y_predictions)):
        object_id = i + 1
        predicted_class = y_predictions[i]
        true_class = encoded_y_test[i]
        if predicted_class == true_class:
            accuracy = 1
        else:
            accuracy = 0
        print('ID= {0:5d}, predicted= {1:3d}, true= {2:3d}, accuracy= {3:4.2f}\n'
              .format(object_id, predicted_class, true_class, accuracy))

    # Calculate accuracy, precision, and recall of the decision tree classifier.
    accuracy = accuracy_score(encoded_y_test, y_predictions)
    precision = precision_score(encoded_y_test, y_predictions)
    recall = recall_score(encoded_y_test, y_predictions)
    print(f"Overall accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def random_forest(training_file, test_file):
    """
    Uses a random forest ensemble classifier (an ensemble made of many decision trees where the final result is the
    mode of the results of all the individual trees) to train a model using the samples of the training file, then
    applies the model to classify the samples from the test file.
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

    # Encode the y class labels.
    encoded_y_test = LabelEncoder().fit_transform(y_test)
    encoded_y_train = LabelEncoder().fit_transform(y_train)

    # One-hot-encode the categorical values in the training set and the test set.
    ohe_X_train = OneHotEncoder().fit_transform(X_train).toarray()
    ohe_X_test = OneHotEncoder().fit_transform(X_test).toarray()
    print("One-hot-encoded!")
    print(ohe_X_train.shape)

    # Reduce the dimensionality of the sets using truncated SVD, which is similar to PCA but does not center the data
    # before computing the SVD.
    reduced_X_Train = TruncatedSVD(n_components=100).fit_transform(ohe_X_train)
    reduced_X_test = TruncatedSVD(n_components=100).fit_transform(ohe_X_test)
    print("Reduced!")
    print(reduced_X_Train.shape)

    # Build the random forest ensemble classifier from the training set.
    model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=10, min_samples_split=20)
    model.fit(reduced_X_Train, encoded_y_train)
    print(f"Model: {model}")

    # Predict the classes of the test set.
    y_predictions = model.predict(reduced_X_test)
    print(f"Predictions: {y_predictions}")

    # Print the predictions and results from the model being applied on the test set.
    for i in range(len(y_predictions)):
        object_id = i + 1
        predicted_class = y_predictions[i]
        true_class = encoded_y_test[i]
        if predicted_class == true_class:
            accuracy = 1
        else:
            accuracy = 0
        print('ID= {0:5d}, predicted= {1:3d}, true= {2:3d}, accuracy= {3:4.2f}\n'
              .format(object_id, predicted_class, true_class, accuracy))

    # Calculate accuracy, precision, and recall of the random forest classifier.
    accuracy = accuracy_score(encoded_y_test, y_predictions)
    precision = precision_score(encoded_y_test, y_predictions)
    recall = recall_score(encoded_y_test, y_predictions)
    print(f"Overall accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def naive_bayes(training_file, test_file):
    """
    Uses a Naive Bayes classifier to train a model using the samples of the training file, then applies the model
    to classify the samples from the test file.
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

    # Encode the y class labels.
    encoded_y_test = LabelEncoder().fit_transform(y_test)
    encoded_y_train = LabelEncoder().fit_transform(y_train)

    # Label-encode the categorical data into integers by transforming all the data with LabelEncoder.
    le_X_train = np.apply_along_axis(LabelEncoder().fit_transform, 0, X_train)
    le_X_test = np.apply_along_axis(LabelEncoder().fit_transform, 0, X_test)
    print(le_X_train.shape)

    model = GaussianNB()
    model.fit(le_X_train, encoded_y_train)
    print(f"Model: {model}")

    # Predict the classes of the test set
    y_predictions = model.predict(le_X_test)
    print(f"Predictions: {y_predictions}")

    # Calculate accuracy of the NB classifier
    accuracy = accuracy_score(y_test, y_predictions)
    print(f"Accuracy: {accuracy}")

    # Print the predictions and results from the model being applied on the test set.
    for i in range(len(y_predictions)):
        object_id = i + 1
        predicted_class = y_predictions[i]
        true_class = encoded_y_test[i]
        if predicted_class == true_class:
            accuracy = 1
        else:
            accuracy = 0
        print('ID= {0:5d}, predicted= {1:3d}, true= {2:3d}, accuracy= {3:4.2f}\n'
              .format(object_id, predicted_class, true_class, accuracy))

    # Calculate accuracy, precision, and recall of the Naive Bayes classifier.
    accuracy = accuracy_score(encoded_y_test, y_predictions)
    precision = precision_score(encoded_y_test, y_predictions)
    recall = recall_score(encoded_y_test, y_predictions)
    print(f"Overall accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def DTvsRFvsNB(training_file, test_file):
    pass


# ****************************************
# Test code:

# decision_tree(os.path.join('Split Data', 'census_trainset.txt'), os.path.join('Split Data', 'census_testset.txt'))
# decision_tree(os.path.join('Split Data', 'credit_trainset.txt'), os.path.join('Split Data', 'credit_testset.txt'))
# random_forest(os.path.join('Split Data', 'census_trainset.txt'), os.path.join('Split Data', 'census_testset.txt'))
# random_forest(os.path.join('Split Data', 'credit_trainset.txt'), os.path.join('Split Data', 'credit_testset.txt'))
naive_bayes(os.path.join('Split Data', 'census_trainset.txt'), os.path.join('Split Data', 'census_testset.txt'))
naive_bayes(os.path.join('Split Data', 'credit_trainset.txt'), os.path.join('Split Data', 'credit_testset.txt'))
