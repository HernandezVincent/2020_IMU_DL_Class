import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn import preprocessing
import random
from sklearn import decomposition
from collections import OrderedDict

cwd = os.getcwd() # Get the current working directory
path_save = cwd + '\\Database\\'

# List of subjects
subject_names = ['001']
# List of exercises
exercises_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']

def get_data():

    DB_X = np.load(path_save + "DB_X.npy")
    DB_Y = np.load(path_save + "DB_Y.npy")
    info = np.load(path_save + "info.npy")

    p = np.array([i for i in range(DB_X.shape[0])])
    random.shuffle(p)

    n = DB_X.shape[0]
    n_sample_train = int(n * 0.8)


    train_X = DB_X[p][:n_sample_train]
    train_X = np.transpose(train_X, [0, 2, 1])
    train_X = train_X.reshape([train_X.shape[0], -1])

    test_X = DB_X[p][n_sample_train:]
    test_X = np.transpose(test_X, [0, 2, 1])
    test_X = test_X.reshape([test_X.shape[0], -1])

    train_Y = DB_Y[p][:n_sample_train]
    test_Y = DB_Y[p][n_sample_train:]

    info_train = info[p][:n_sample_train]
    info_test = info[p][n_sample_train:]

    return train_X, test_X, train_Y, test_Y, info_train, info_test

train_X, test_X, train_Y, test_Y, info_train, info_test = get_data()
