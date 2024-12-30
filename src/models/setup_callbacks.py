from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def setup_callbacks(model_checkpoint_path="../model_checkpoint/CNN.keras"):
    """
    Set up early stopping and model checkpoint callbacks.

    Parameters:
        model_checkpoint_path (str): Path to save the best model checkpoint.
    
    Returns:
        list: List of callbacks for training.
    """
    early_stopping = EarlyStopping(
        patience=30,  # Wait 30 epochs for improvement
        min_delta=0.01,  # Minimum change to qualify as improvement
        verbose=1,  # Print messages when stopping
        mode='min',  # Minimize the monitored metric
        monitor='val_loss',  # Monitor validation loss
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,  # Save best model
        monitor="val_accuracy",  # Monitor validation accuracy
        save_best_only=True,  # Save only when performance improves
        mode="max"  # Maximize the monitored metric
    )
    
    return [early_stopping, model_checkpoint]
