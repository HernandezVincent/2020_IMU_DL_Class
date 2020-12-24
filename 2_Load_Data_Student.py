import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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

data_cut_X = []
data_cut_Y = []

# Each repetition may have different lengths so we spline each repetition so they all have the same length for the Neural Network
spline = 50

# Path to the IMU data
path_data = path_root + '\\Data\\' + subject_names[subject_ind] + "\\" + exercises_names[exercice_ind] + ".csv"

# Loop through each exercices
for exercice_ind in range(len(exercises_names)):

    path_data = path_root + '\\Data\\' + subject_names[subject_ind] + "\\" + exercises_names[exercice_ind] + ".csv"
    path_point_save = path_root + '\\Points\\' + subject_names[subject_ind] + "\\"

    # Load the data
    data, data_dict = get_data(path_data=path_data)

    # Load the points
    points = np.load(path_point_save + exercises_names[exercice_ind] + "_points.npy")

    n = 0

    # Cut the data
    for point in points:

        print("Repetitions " + str(n+1) + " - Begin: " + str(point[0]) + " to " + str(point[1]))

        # Select the data for the current repetitions from begin to end (from point[0] to point[1])
        temp = data[point[0]:point[1]]

        # Spline the repetition to the same number of frame
        temp = spline_data(temp, spline=spline)
        data_cut_X.append(temp)

        n += 1

    # Create an array of label.
    # For instance [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] represent the labels for 10 repetition of activity number 2
    temp = np.array([exercice_ind for i in range(len(points))])

    data_cut_Y.append(temp)
    print(temp.shape)

# Gather the data in a matrix of size (50 * 50 * 9) (sample * frame * DOF)
database_X = np.concatenate([d[np.newaxis, :, :] for d in data_cut_X], axis=0)

# Gather the data in a matrix of size (50) (label)
database_Y = np.concatenate([d for d in data_cut_Y], axis=0)
database_X_std = np.empty(shape=database_X.shape)

# Standardize the input data
for i in range(database_X.shape[2]):
    mean = np.mean(database_X[:, :, i])
    std = np.std(database_X[:, :, i])
    database_X_std[:, :, i] = (database_X[:, :, i] - mean) / std

# Path to the final database that will be used for training and testing our model
path_save = path_root + '\\Database\\'

# Create the folder if does not exist
if not os.path.exists(path_save):
    os.makedirs(path_save)

# Save the database
np.save(path_save + "DB_X.npy", database_X_std)
np.save(path_save + "DB_Y.npy", database_Y)