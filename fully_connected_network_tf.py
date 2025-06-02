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
def one_hot_matrix_test(target):
    label = tf.constant(1)
    C = 4
    result = target(label, C)
    print("Test 1:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 1., 0., 0.]), "Wrong output. Use tf.one_hot"
    label_2 = [2]
    C = 5
    result = target(label_2, C)
    print("Test 2:", result)
    assert result.shape[0] == C, "Use the parameter C"
    assert np.allclose(result, [0., 0., 1., 0., 0.]), "Wrong output. Use tf.reshape as instructed"

    print("\033[92mAll test passed")


one_hot_matrix_test(one_hot_encoding)