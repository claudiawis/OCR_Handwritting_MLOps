import numpy as np
import os
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

def build_model(input_shape, num_classes):
    """
    Build and compile a CNN model.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (28, 28, 1)).
        num_classes (int): Number of classes for classification.

    Returns:
        model: Compiled CNN model.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def save_model_summary(model, output_path="models/model_architecture_summary.txt"):
    """
    Save the model architecture summary to a text file.

    Parameters:
        model: The Keras model object.
        output_path (str): Path to save the summary file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Model architecture summary saved to {output_path}")

def load_callbacks(callback_path):
    """
    Load callbacks from a pickle file.

    Parameters:
        callback_path (str): Path to the callbacks pickle file.

    Returns:
        list: List of callbacks.
    """
    with open(callback_path, "rb") as f:
        callbacks = pickle.load(f)
    print(f"Callbacks loaded from {callback_path}.")
    return callbacks

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("data/processed/X_train_reshaped.npy")
    y_train = np.load("data/processed/y_train_one_hot.npy")
    X_test = np.load("data/processed/X_test_reshaped.npy")
    y_test = np.load("data/processed/y_test_one_hot.npy")

    # Load class weights
    class_weights = np.load("data/processed/class_weights.npy", allow_pickle=True).item()

    # Load callbacks
    callbacks = load_callbacks("models/callbacks.keras")

    # Define input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    # Set up MLflow tracking
    mlflow.set_tracking_uri("https://dagshub.com/KazemZh/OCR_Handwritting_MLOps.mlflow")
    mlflow.set_experiment("OCR_CNN_Training")

    # Start an MLflow run
    with mlflow.start_run():
        # Build the model
        model_cnn = build_model(input_shape, num_classes)

        # Log model summary
        save_model_summary(model_cnn, output_path="models/model_architecture_summary.txt")

        # Log parameters
        mlflow.log_param("input_shape", input_shape)
        mlflow.log_param("num_classes", num_classes)

        # Train the model
        history = model_cnn.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        
        # Log metrics
        for epoch, metrics in enumerate(history.history["accuracy"]):
            mlflow.log_metric("training_accuracy", metrics, step=epoch)
            mlflow.log_metric("validation_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Save the trained model locally and log it to MLflow
        model_cnn.save("models/CNN.h5")
        mlflow.keras.log_model(model_cnn, "model")

        # Save the training history
        history_df = pd.DataFrame(history.history)
        os.makedirs("metrics", exist_ok=True)
        history_df.to_csv("metrics/training_history.csv", index=False)
        print("Training history saved as metrics/training_history.csv.")

        # Plot and save accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig("metrics/training_accuracy.png")
        print("Training accuracy plot saved as metrics/training_accuracy.png.")
        mlflow.log_artifact("metrics/training_accuracy.png")
        mlflow.log_artifact("metrics/training_history.csv")
