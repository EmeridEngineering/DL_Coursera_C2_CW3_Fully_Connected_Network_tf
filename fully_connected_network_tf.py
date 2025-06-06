import h5py
import numpy as np
import tensorflow as tf

from dnn_lib_tf import *
from tensorflow.python.framework.ops import EagerTensor

# Load Datasets
x_train_initial, y_train_initial, x_test_initial, y_test_initial = load_dataset()

# Data visualization
display_data(x_train_initial, y_train_initial)

# Generate possible categorizations
unique_labels = get_unique_labels(y_train_initial)
# print(unique_labels)

# Preprocess the datasets
x_train = x_train_initial.map(unroll_and_normalize)
x_test = x_test_initial.map(unroll_and_normalize)

# print(x_train.element_spec)
# print(next(iter(x_train)))

# Preprocess the labels datasets
y_train = y_train_initial.map(one_hot_encoding)
y_test = y_test_initial.map(one_hot_encoding)

# print(next(iter(y_test)))

parameters, costs, train_accuracies, test_accuracies = train_deep_fully_connected_model_tf(x_train, y_train, x_test, y_test, num_epochs = 100)