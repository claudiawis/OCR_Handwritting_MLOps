import numpy as np
import os

def reshape_data(input_path, output_path_train, output_path_test, width=28, height=28):
    # Load the train and test data
    X_train = np.load(os.path.join(input_path, "X_train.npy"))
    X_test = np.load(os.path.join(input_path, "X_test.npy"))

    # Reshape the data to include the channel dimension (for grayscale images)
    X_train_reshaped = X_train.reshape((-1, height, width, 1))
    X_test_reshaped = X_test.reshape((-1, height, width, 1))

    # Save the reshaped data
    np.save(os.path.join(output_path_train, "X_train_reshaped.npy"), X_train_reshaped)
    np.save(os.path.join(output_path_test, "X_test_reshaped.npy"), X_test_reshaped)

    print("Reshaped training data shape:", X_train_reshaped.shape)
    print("Reshaped testing data shape:", X_test_reshaped.shape)

if __name__ == "__main__":
    input_data_path = "data/processed"  # Directory where X_train.npy and X_test.npy are stored
    output_train_path = "data/processed"  # Directory to save reshaped training data
    output_test_path = "data/processed"  # Directory to save reshaped testing data

    # Call the reshape function
    reshape_data(input_data_path, output_train_path, output_test_path)
