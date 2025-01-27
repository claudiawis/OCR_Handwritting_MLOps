import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and model
X_test = np.load("data/processed/X_test_reshaped.npy")
y_test = np.load("data/processed/y_test_one_hot.npy")
model = load_model("models/CNN.h5")

# Get predictions
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Classification report
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("metrics/classification_report.csv", index=True)
print("Classification report saved as metrics/classification_report.csv.")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.savetxt("metrics/confusion_matrix.csv", cm, delimiter=",")
print("Confusion matrix saved as metrics/confusion_matrix.csv.")

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("metrics/confusion_matrix.png")
print("Confusion matrix plot saved as metrics/confusion_matrix.png.")
