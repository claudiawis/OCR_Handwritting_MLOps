import numpy as np
from sklearn.model_selection import train_test_split

# Load preprocessed features and labels
features_path = "data/processed/features.npy"
labels_path = "data/processed/labels.npy"

X = np.load(features_path)
Y = np.load(labels_path)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,  # 20% of the data will be used for testing
    random_state=42,  # Set random seed for reproducibility
    stratify=Y  # Ensure class distribution remains balanced between train and test sets
)

# Save the split datasets
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/Y_train.npy", Y_train)
np.save("data/processed/Y_test.npy", Y_test)

print("Data successfully split into training and testing sets.")
print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples")
