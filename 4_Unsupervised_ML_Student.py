import os
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import OrderedDict

cwd = os.getcwd() # Get the current working directory
path_save = cwd + '\\Database\\'

# List of subjects
subject_names = ['001']
# List of exercises
exercises_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']

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

    return train_X, test_X, train_Y, test_Y

train_X, test_X, train_Y, test_Y = get_data(num_classes)

model = decomposition.PCA(n_components=2)
model.fit(train_X) # Learning

print(model.components_)

z_train = model.transform(train_X)  # Transform
z_test = model.transform(test_X)

unique_class = np.unique(train_Y)

cmap = plt.get_cmap('nipy_spectral')
colors = cmap(np.linspace(0, 1, len(unique_class)))
colors = dict(zip(unique_class, colors))

plt.figure()

for ind in unique_class:

    indices = [train_Y == ind]

    X = z_train[:, 0][indices]
    Y = z_train[:, 1][indices]

    plt.scatter(X, Y, color=colors[ind], label=exercises_names[ind], s=300)

    indices = [test_Y == ind]

    X = z_test[:, 0][indices]
    Y = z_test[:, 1][indices]

    plt.scatter(X, Y, color=colors[ind], label=exercises_names[ind], s=300, marker="+")

    plt.legend()

# Classification on the latent space
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X=z_train, y=train_Y)

pred_Y = classifier.predict(X=z_test)

print("True test class are:")
print(str(test_Y))
print("Predicted test class are:")
print(str(pred_Y))

# Confusion matrix
print('Confusion Matrix')
print(confusion_matrix(test_Y, pred_Y))
