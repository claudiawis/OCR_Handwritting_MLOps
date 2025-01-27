import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs

import numpy as np
from tensorflow.keras.utils import to_categorical

def one_hot_encode_labels(y_train_path, y_test_path, output_train_path, output_test_path):
    # Load the training and testing labels
    Y_train = np.load(y_train_path)
    Y_test = np.load(y_test_path)

    # Determine the number of unique classes
    num_classes = max(len(np.unique(Y_train)), len(np.unique(Y_test)))

    # One-hot encode the labels
    y_train = to_categorical(Y_train, num_classes=num_classes)
    y_test = to_categorical(Y_test, num_classes=num_classes)

    # Save the one-hot encoded labels
    np.save(output_train_path, y_train)
    np.save(output_test_path, y_test)

    # Print verification details
    print("Shape of unique Y_train:", np.unique(Y_train).shape)
    print("Shape of one-hot encoded y_train:", y_train.shape)
    print("Example of class label in y_train:", np.argmax(y_train[0]))
    print("Number of unique classes in Y_train:", len(np.unique(Y_train)))

if __name__ == "__main__":
    y_train_path = "data/processed/Y_train.npy"  # Path to the training labels
    y_test_path = "data/processed/Y_test.npy"  # Path to the testing labels
    output_train_path = "data/processed/y_train_one_hot.npy"  # Path to save one-hot encoded training labels
    output_test_path = "data/processed/y_test_one_hot.npy"  # Path to save one-hot encoded testing labels

    one_hot_encode_labels(y_train_path, y_test_path, output_train_path, output_test_path)
