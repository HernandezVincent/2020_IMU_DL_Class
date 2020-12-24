import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from matplotlib import pyplot as plt
import random
from keras import optimizers, Model, metrics, backend, losses
from keras.layers import Dense, Activation, Dropout, Input
import os
import itertools
from tabulate import tabulate
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as backend_keras

# gpu_options = tf.GPUOptions()
# config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

cwd = os.getcwd()  # Get the current working directory
path_save = cwd + '\\Database\\'

# List of subjects
subject_names = ['001']
# List of exercises
exercices_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']

num_classes = 5

def get_data(num_classes):

    DB_X = np.load(path_save + "DB_X.npy")
    DB_Y = np.load(path_save + "DB_Y.npy")

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

    train_Y = np_utils.to_categorical(train_Y, num_classes)
    test_Y = np_utils.to_categorical(test_Y, num_classes)

    return train_X, test_X, train_Y, test_Y
def rank_and_display(hyperparameters_list, accuracy_list):

    # Sort metric and generation
    idx = np.argsort(accuracy_list)
    hyperparameters_list_sorted = np.array(hyperparameters_list)[idx].tolist()
    hyperparameter_best = hyperparameters_list_sorted[-1]

    accuracy_list_sorted = np.array(accuracy_list)[idx].tolist()

    df = pd.DataFrame.from_dict(hyperparameters_list_sorted)
    df["Accuray"] = accuracy_list_sorted

    print(tabulate(df, headers='keys', tablefmt='psql'))

    return hyperparameter_best

train_X, test_X, train_Y, test_Y = get_data(num_classes)
