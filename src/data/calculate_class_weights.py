import numpy as np
import os

def calculate_class_weights(y_train_path, output_path, smooth_factor=1e-6):
    # Load the training labels
    Y_train = np.load(y_train_path)

    # Count occurrences of each class
    unique_classes, class_counts = np.unique(Y_train, return_counts=True)
    max_count = np.max(class_counts)

    # Calculate class weights to penalize majority classes and prioritize minority classes
    class_weights_manual = class_counts / max_count
    class_weights_manual += smooth_factor  # Add smooth factor to avoid zero weights

    # Normalize weights and create a dictionary
    class_weights_dict_manual = {
        int(label): weight / np.sum(class_weights_manual)
        for label, weight in zip(unique_classes, class_weights_manual)
    }

    # Save the class weights as a .npy file
    np.save(output_path, class_weights_dict_manual)
    print("Class Weights Dictionary: ", class_weights_dict_manual)

if __name__ == "__main__":
    y_train_path = "data/processed/Y_train.npy"  # Path to the training labels
    output_path = "data/processed/class_weights.npy"  # Path to save the class weights

    calculate_class_weights(y_train_path, output_path)
