import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn import preprocessing

def get_data(path_data):

    df = pd.read_csv(path_data, sep=",")
    time = df['TimeStamp'].values  # Extract columns named Time

    columns = list(df.columns.values)[1:]  # Get a list of columns name
    time = time - time[0]

    data = df.values[:, 1:]

    data_dict = {}

    for i, m in enumerate(columns):
        data_dict[m] = data[:, i]

    return data, data_dict
def spline_data(data, spline=60):

    x = np.array([x for x in range(data.shape[0])])
    x_new = np.linspace(x.min(), x.max(), spline)
    data_spline = interp1d(x, data, kind='cubic', axis=0)(x_new)

    return data_spline

path_root = os.getcwd() # Get the current working directory

# List of subjects
subject_names = ['001']
# List of exercises
exercises_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']

subject_ind = 0
exercice_ind = 0
