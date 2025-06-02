import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.src.backend import shape

USE_RANDOM_SEED = True

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    test_dataset = h5py.File('datasets/test_signs.h5', "r")

    x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    """
    # Type check (TensorSliceDataset)
    type(x_train)

    # Shape and data type check of the elements
    print(x_train.element_spec)

    # Accessing the elements by iter (element contains value, shape and dtype
    print(next(iter(x_train)))

    # Accessing the values of the elements by iter
    print(next(iter(x_train)).numpy())

    # Accessing the values of the elements by for loop (uses iterators internally) and creating a set of catagorizations
    for element in y_train:
        print(element.numpy())
    """

    return x_train, y_train, x_test, y_test


def get_unique_labels(y_train):
    unique_labels = set()
    for element in y_train:
        unique_labels.add(element.numpy())

    return unique_labels

def display_data(image_dataset, label_dataset):
    images_iter = iter(image_dataset)
    labels_iter = iter(label_dataset)

    plt.figure(figsize=(7,7)) # size in inches
    for i in range(25):
        plt.subplot(5, 5, i + 1) # 5x5 grid, picture i
        plt.imshow(next(images_iter))
        plt.title(next(labels_iter).numpy())
        plt.axis("off")
    plt.show()


def unroll_and_normalize(image_dataset):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.

    Arguments
    image - Tensor.

    Returns:
    result -- Transformed tensor
    """
    result = tf.reshape(image_dataset, [-1,])
    result = tf.cast(result, tf.float32) / 255.0

    return result

def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    initializer = tf.keras.initializers.GlorotNormal()

    W1 = tf.Variable(initializer(shape = (25, 12288)))
    b1 = tf.Variable(initializer(shape = (25, 1)))
    W2 = tf.Variable(initializer(shape = (12, 25)))
    b2 = tf.Variable(initializer(shape = (12, 1)))
    W3 = tf.Variable(initializer(shape = (6, 12)))
    b3 = tf.Variable(initializer(shape = (6, 1)))

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2,
        "W3" : W3,
        "b3" : b3
    }

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    Z1 = tf.math.add(tf.linalg.matmul(parameters["W1"], X), parameters["b1"])
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(parameters["W2"], A1), parameters["b2"])
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(parameters["W3"], A2), parameters["b3"])

    return Z3


def one_hot_encoding(label, C=6):
    """
    Computes the one hot encoding for a single label

    Arguments:
        label --  (int) Categorical labels
        C --  (int) Number of different classes that label can take

    Returns:
        one_hot -- tf.Tensor A one-dimensional tensor (array) with the one hot encoding.
    """

    return tf.reshape(tf.one_hot(label, C), shape=[C,]) # reshape for confidence as one_hot shape depends on the inputs


def compute_total_loss(logits, labels):
    """
    Computes the total loss

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3

    Returns:
    total_loss - Tensor of the total loss value
    """

    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits = True))

    return total_loss