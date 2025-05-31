import h5py
import numpy as np
import tensorflow as tf

from dnn_lib_tf import *

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