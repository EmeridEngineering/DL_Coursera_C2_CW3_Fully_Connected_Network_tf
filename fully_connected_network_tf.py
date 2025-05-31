import h5py
import numpy as np
import tensorflow as tf

from dnn_lib_tf import *

# Load Datasets
x_train, y_train, x_test, y_test = load_dataset()

# Data visualization
display_data(x_train, y_train)

# Generate possible categorizations
unique_labels = get_unique_labels(y_train)
# print(unique_labels)



