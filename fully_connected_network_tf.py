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
def compute_total_loss_test(target, Y):
    pred = tf.constant([[2.4048107, 5.0334096],
                        [-0.7921977, -4.1523376],
                        [0.9447198, -0.46802214],
                        [1.158121, 3.9810789],
                        [4.768706, 2.3220146],
                        [6.1481323, 3.909829]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break

    print("Test 1: ", result)
    assert (type(result) == EagerTensor), "Use the TensorFlow API"
    assert (np.abs(result - (
                0.50722074 + 1.1133534) / 2.0) < 1e-7), "Test 1 does not match. Did you get the reduce sum of your loss functions?"

    ### Test 2
    labels = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    logits = tf.constant([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])

    result = compute_total_loss(logits, labels)
    print("Test 2: ", result)
    assert np.allclose(result, 3.295837), "Test 2 does not match."

    print("\033[92mAll test passed")


compute_total_loss_test(compute_total_loss, y_train_initial)