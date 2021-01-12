import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Only use CPU
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

# Uncomment the following lines if you plan to use your GPU at 90 %
# gpu_options = tf.GPUOptions()
# config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

# Function area
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
def get_hyperparameters(num_classes):

    choices = dict(input_shape=[450], num_classes=[5],
                   learning_rate=[0.1, 0.01, 0.001],
                   layers_units=[[64, 64], [128, 128], [256, 256],
                                 [64, 64, 64], [128, 128, 128], [256, 256, 256]],
                   activation_function=['relu', 'sigmoid', 'tanh'])

    keys, values = zip(*choices.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    df = pd.DataFrame.from_dict(combs)

    print(tabulate(df, headers="keys", tablefmt="psql"))

    return combs
def MLP(hyper):

    num_classes = hyper['num_classes']
    input_shape = hyper['input_shape']
    learning_rate = hyper['learning_rate']
    layers_units = hyper['layers_units']
    activation_function = hyper['activation_function']

    # Create graph and session
    tf.reset_default_graph()
    backend_keras.clear_session()

    graph = tf.get_default_graph()
    sess = tf.Session()

    backend_keras.set_session(sess)

    # Input layers
    x = Input(shape=(input_shape,), name="X")
    l = x

    for i in range(0, len(layers_units)):

        l = Dense(units=layers_units[i], name="Dense_" + str(i))(l)
        l = Activation(activation=activation_function,
                       name="Activation_dense_" + str(i))(l)

    y_ = Dense(units=num_classes, name="output", activation="softmax")(l)

    # Scope
    loss = losses.categorical_crossentropy
    metric = metrics.categorical_accuracy
    optimizer = optimizers.sgd()

    # Compile
    model = Model(inputs=[x], outputs=[y_])
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.summary()

    return model
def report(test_Y_true, test_Y_pred, labels):

    print('Confusion Matrix')
    cm = confusion_matrix(test_Y_true, test_Y_pred)
    print(cm)

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.show()

cwd = os.getcwd()  # Get the current working directory
path_save = cwd + '\\Database\\'

# List of subjects
subject_names = ['001']
# List of exercises
exercices_names = ['AbductionRight', 'BicepsRight', 'Squat', 'AbductionLeft', 'BicepsLeft']
num_classes = len(exercices_names)

# Get the split train/test data
train_X, test_X, train_Y, test_Y = get_data(num_classes)

#################################################################################
# Bad architectures here. Only used for example purpose on the confusion matrix #
#################################################################################
hyper = {'input_shape': 450, 'num_classes': num_classes, 'learning_rate': 0.1,
         'layers_units': [64, 64], 'activation_function': 'sigmoid'}
max_epoch = 20

model = MLP(hyper)

for i in range(max_epoch):
    # Training
    model.fit(x=train_X, y=train_Y, epochs=1, verbose=0)

    # Evaluation of the model
    loss, accuracy = model.evaluate(x=train_X, y=train_Y, verbose=0)

# Get prediction
test_Y_pred = model.predict(x=test_X)
test_Y_pred = np.argmax(test_Y_pred, axis=1)
test_Y_true = np.argmax(test_Y, axis=1)

# Confusion matrix
report(test_Y_true, test_Y_pred, labels=exercices_names)

# example results:
# Confusion Matrix
# [[0 0 0 1 0]
#  [0 2 0 0 0]
#  [0 0 0 2 0]
#  [0 0 0 2 0]
#  [0 2 0 1 0]]
# We can see which labels are mistaken (Some numbers appears outside of the diagonal)
# Check more here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

##################################################
# Hyperparameters search (Train a lot of models) #
##################################################
combs = get_hyperparameters(num_classes=num_classes)

hyper_accuracy_list = []

for hyper in combs:

    model = MLP(hyper)

    loss_hist = []
    accuracy_hist = []

    for i in range(max_epoch):

        # Training
        model.fit(x=train_X, y=train_Y, epochs=1, verbose=0)

        # Evaluation of the model
        loss, accuracy = model.evaluate(x=train_X, y=train_Y, verbose=0)

        loss_hist.append(loss)
        accuracy_hist.append(accuracy)

        print("Loss: " + str(loss) + " - Accuracy: " + str(accuracy))

    # Get the max accuracy for the current training
    accuracy_max = np.nanmax(accuracy_hist)
    hyper_accuracy_list.append(accuracy_max)

hyperparameter_best = rank_and_display(hyperparameters_list=combs, accuracy_list=hyper_accuracy_list)

############################
# Train/test with the best #
############################
path_model = 'Model\\'  # Path to the model
if not os.path.exists(path_model): # Create folder if not exist
    os.makedirs(path_model)

loss_train_hist = []
loss_test_hist = []
accuracy_train_hist = []
accuracy_test_hist = []

model = MLP(hyperparameter_best)
max_epoch = 100

for i in range(max_epoch):

    # Training
    model.fit(x=train_X, y=train_Y, epochs=1, verbose=0)

    # Evaluation of the model
    loss_train, accuracy_train = model.evaluate(x=test_X, y=test_Y, verbose=0)
    loss_test, accuracy_test = model.evaluate(x=test_X, y=test_Y, verbose=0)

    loss_train_hist.append(loss_train)
    accuracy_train_hist.append(accuracy_train)
    loss_test_hist.append(loss_test)
    accuracy_test_hist.append(accuracy_test)

    print("Train - Loss: " + str(loss_train) + " - Accuracy: " + str(accuracy_train))
    print("Test - Loss: " + str(loss_test) + " - Accuracy: " + str(accuracy_test))

    # Save the model if test accuracy for current epoch is higher than all the previous ones
    if i > 1:
        # Save the model if condition is met
        if accuracy_test >= np.max(accuracy_test_hist):
            model.save(path_model + "model.h5")
            print("Model saved in " + path_model)

# Plot accuracy train and test
plt.figure()
plt.plot(accuracy_train_hist, 'b', label='Train accuracy')
plt.plot(accuracy_test_hist, 'r', label='Test accuracy')
plt.legend()

# Load model that was saved
model = load_model(path_model + 'model.h5')

# Summarize model
model.summary()

# Evaluate model
loss_test, accuracy_test = model.evaluate(x=test_X, y=test_Y, verbose=0)

# Get prediction
test_Y_pred = model.predict(x=test_X)
test_Y_pred = np.argmax(test_Y_pred, axis=1)
test_Y_true = np.argmax(test_Y, axis=1)

# Confusion matrix
print('Confusion Matrix')
report(test_Y_true, test_Y_pred, labels=exercices_names)

# Confusion Matrix example (Your results may be different)
# [[4 0 0 0]
#  [0 3 0 0]
#  [0 0 1 0]
#  [0 0 0 2]]
# Everything is classified (Only number along the diagonal of the matrix)