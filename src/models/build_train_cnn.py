import numpy as np
import pickle
import dagshub
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt

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
    # Initialize DAGsHub MLflow Connection
    dagshub.init(repo_owner="KazemZh", repo_name="OCR_Handwritting_MLOps", mlflow=True)

    # Enable MLflow Autologging for TensorFlow
    mlflow.tensorflow.autolog()

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

    mlflow.set_experiment("OCR_CNN_Training")
 
    with mlflow.start_run():  
        # Log training start using `set_tag()`
        mlflow.set_tag("training_status", "started")

        # Build the model
        model_cnn = build_model(input_shape, num_classes)

        # Log model architecture
        model_summary_path = "models/model_architecture_summary.txt"
        save_model_summary(model_cnn, model_summary_path)
        mlflow.log_artifact(model_summary_path)

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

        # Log training completion using `set_tag()`
        mlflow.set_tag("training_status", "completed")

        # Save the trained model in `.keras` format (fixing warning)
        model_path = "models/CNN.keras"
        model_cnn.save(model_path)
        print(f"Model training complete. Model saved as {model_path}")

        # Log the model in MLflow with input example and signature
        input_example = X_train[:1]  # Example input for inference
        mlflow.tensorflow.log_model(
            model_cnn,
            "cnn_model",
            input_example=input_example
        )

        # # Log training history
        # history_csv_path = "metrics/training_history.csv"
        # os.makedirs("metrics", exist_ok=True)
        # history_df = pd.DataFrame(history.history)
        # history_df.to_csv(history_csv_path, index=False)
        # mlflow.log_artifact(history_csv_path)

        # Plot and save accuracy
        plot_path = "metrics/training_accuracy.png"
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Training accuracy plot saved as {plot_path}")

        # Automatically register the model 
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/cnn_model"
        mlflow.register_model(model_uri, name="cnn_model")

    print("Training complete! Check DAGsHub MLflow for details.")