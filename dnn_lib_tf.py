import h5py
import matplotlib.pyplot as plt
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