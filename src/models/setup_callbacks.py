from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle

def setup_callbacks(callbacks_path="models/callbacks.keras", model_checkpoint_path="models/CNN_best_model.keras"):
    """
    Set up early stopping and model checkpoint callbacks and save them to a file.

    Parameters:
        callbacks_path (str): Path to save the serialized callbacks.
        model_checkpoint_path (str): Path to save the best model checkpoint.
    
    Returns:
        list: List of callbacks for training.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(callbacks_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    
    # EarlyStopping callback
    early_stopping = EarlyStopping(
        patience=30,  # Wait 30 epochs for improvement
        min_delta=0.01,  # Minimum change to qualify as improvement
        verbose=1,  # Print messages when stopping
        mode="min",  # Minimize the monitored metric
        monitor="val_loss",  # Monitor validation loss
    )
    
    # ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,  # Save the best model
        monitor="val_accuracy",  # Monitor validation accuracy
        save_best_only=True,  # Save only when performance improves
        mode="max",  # Maximize the monitored metric
        verbose=1,  # Print when a model is saved
    )
    
    # List of callbacks
    callbacks = [early_stopping, model_checkpoint]
    
    # Save the callbacks using pickle
    with open(callbacks_path, "wb") as f:
        pickle.dump(callbacks, f)

    print(f"Callbacks saved to {callbacks_path}.")
    return callbacks

if __name__ == "__main__":
    # Define paths
    callbacks_path = "models/callbacks.keras"
    model_checkpoint_path = "models/CNN_best_model.keras"
    
    # Setup and save callbacks
    setup_callbacks(callbacks_path, model_checkpoint_path)
