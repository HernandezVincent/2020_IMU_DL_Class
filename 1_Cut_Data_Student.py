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
exercice_ind = 0





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# 1) Call the plotter
# 2) Click the points
# 3) After you have finish to click all the points, close the figure
# 4) Use points = plotter.get_points() to recover the point
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# This part of the code must be use only after you have finish to click !!! #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #



