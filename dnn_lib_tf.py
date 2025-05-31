import h5py
import tensorflow as tf

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

