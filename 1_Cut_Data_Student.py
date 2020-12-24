import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

class Plotter:

    def __init__(self, data, title):

        self.points = []
        self.data = data

        plt.ion()
        self.fig = plt.figure(1, figsize=(16, 5))
        self.ax1 = plt.subplot(1, 1, 1)

        self.fig.suptitle(title + '\n' + 'z + right click')

        self.ax1.plot(np.sum(data[:, 1:5], axis=1), 'k')
        self.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.92, wspace=0.05, hspace=0.1)

        self.ax1.grid()

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):

        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        if event.key == 'z':
            self.points.append([int(event.xdata)])
            self.ax1.plot(event.xdata, event.ydata, 'o', color='r', markersize=5)
            self.ax1.axvline(x=event.xdata, color='b')
            self.fig.canvas.draw()

    def get_points(self):

        return self.points

    def plot_point(self, points):

        for point in points:

            self.ax1.axvline(x=point[0], color='r', linewidth=3)
            self.ax1.axvline(x=point[1], color='b', linewidth=3)

path_root = os.getcwd() # Get the current working directory

# List of subjects
subject_names = ['001']
# List of exercises
exercises_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']

subject_ind = 0
exercice_ind = 4

# Create a path to the data that we are going to use
path_data = path_root + "\\Data\\" +  subject_names[subject_ind] + "\\" +  exercises_names[exercice_ind] + ".csv"

path_point_save = path_root + '\\Points\\' +  subject_names[subject_ind] + "\\"

# Create the folder where to save the points data
if not os.path.exists(path_point_save):
    os.makedirs(path_point_save)

# Get the data
data, data_dict = get_data(path_data=path_data)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# 1) Call the plotter                                                       #
# 2) Click the points                                                       #
# 3) After you have finish to click all the points, close the figure        #
# 4) Use points = plotter.get_points() to recover the point                 #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# Instantiate the Plotter class use to plot the IMU accelerometer data
plotter = Plotter(data, title="")










### This part of the code must be use only after you have finish to click !!! ###

# Get the click points
points = plotter.get_points()

# Reshape the list of points to a 2D numpy array
points = np.array(points) # change to numpy array of size (20, 1)
points = points.reshape(points.shape[0]//2, 2) # Reshape to numpy array of size (10, 2)

# Instantiate the Plotter class use to plot the IMU accelerometer data and the points
plotter = Plotter(data, title="")
plotter.plot_point(points=points)

# Save the points as a numpy array
np.save(path_point_save + exercises_names[exercice_ind] + "_points", points)


