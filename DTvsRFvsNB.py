"""
Robert Wyckoff
CS 4435
04/01/2020
Project 1
"""

import numpy as np
import pandas as pd
import sklearn
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

# Save the trainsets and testsets to comma-delimited .txt files for later use.
np.savetxt(os.path.join('Split Data', 'census_trainset.txt'), census_trainset_arr, fmt='%s', delimiter=',')
np.savetxt(os.path.join('Split Data', 'census_testset.txt'), census_testset_arr, fmt='%s', delimiter=',')
np.savetxt(os.path.join('Split Data', 'credit_trainset.txt'), credit_trainset_arr, fmt='%s', delimiter=',')
np.savetxt(os.path.join('Split Data', 'credit_testset.txt'), credit_testset_arr, fmt='%s', delimiter=',')

# Split data into train and test arrays using random sampling without replacement
#np.savetxt(fname='')

# test print
# np.set_printoptions(precision=2, suppress=True)
# print(fake_arr)


# noinspection PyPep8Naming
def DTvsRFvsNB(training_file, test_file):
    # Do the below, but specify the data types as a tuple in the dtype parameter. I think.
    # test_arr = np.genfromtxt(fname=ci_raw_data_path, delimiter=',', dtype='U')
    pass
