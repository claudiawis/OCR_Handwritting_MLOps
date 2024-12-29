import numpy as np
import pandas as pd
from PIL import Image
import os

def prepare_features(input_file, output_features_file, output_labels_file, width=28, height=28):
    """
    Prepares input features (X) and labels (Y) for training a CNN.
    
    Args:
    - input_file (str): Path to the input CSV file with image paths and encoded labels.
    - output_features_file (str): Path to save the numpy array of features (X).
    - output_labels_file (str): Path to save the numpy array of labels (Y).
    - width (int): Target width of the resized images.
    - height (int): Target height of the resized images.
    """
    # Load the filtered DataFrame
    df_filtered = pd.read_csv(input_file)

    X, Y = [], []

    # Loop through each row to process images
    for _, row in df_filtered.iterrows():
        try:
            # Open image, convert to grayscale, and resize
            image = Image.open(row['image_path']).convert('L').resize((width, height))
            X.append(np.array(image))
            Y.append(row['transcription_encoded'])
        except Exception as e:
            print(f"Error processing image {row['image_path']}: {e}")

    # Normalize image data and convert to numpy arrays
    X = np.array(X) / 255.0  # Normalize to range [0, 1]
    Y = np.array(Y)

    # Save the arrays as .npy files
    np.save(output_features_file, X)
    np.save(output_labels_file, Y)

    print(f"Features (X) saved to {output_features_file}")
    print(f"Labels (Y) saved to {output_labels_file}")
    print(f"Min pixel value in X: {X.min():.3f}, Max pixel value in X: {X.max():.3f}")
    print(f"X and Y have the same length: {len(X) == len(Y)}")


if __name__ == "__main__":
    # File paths
    input_file = "data/processed/encoded_data.csv"  # Input CSV with filtered data
    output_features_file = "data/processed/features.npy"  # Output numpy file for features
    output_labels_file = "data/processed/labels.npy"  # Output numpy file for labels

    # Call the function
    prepare_features(input_file, output_features_file, output_labels_file)
