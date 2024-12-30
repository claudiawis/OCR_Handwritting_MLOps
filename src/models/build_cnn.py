import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import os

def build_model(input_shape, num_classes):
    """
    Build and compile a CNN model.
    
    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (28, 28, 1)).
        num_classes (int): Number of classes for classification.

    Returns:
        model: Compiled CNN model.
    """
    # Input layer for grayscale images of size 28x28
    inputs = Input(shape=input_shape)

    # Convolutional Layer: Apply 32 filters of size 5x5 with ReLU activation
    x = Conv2D(32, (5, 5), activation='relu')(inputs)

    # Pooling Layer: Apply MaxPooling with a 2x2 window
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Dropout Layer: Apply dropout with rate 0.2 to reduce overfitting
    x = Dropout(0.2)(x)

    # Flatten the feature maps to prepare for the fully connected layer
    x = Flatten()(x)

    # Fully Connected Layer: 128 neurons with ReLU activation
    x = Dense(128, activation='relu')(x)

    # Output Layer: Apply softmax activation to predict the class
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build and compile the model
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
    with open(output_path, "w", encoding="utf-8") as f:  # Set encoding to 'utf-8'
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Model architecture summary saved to {output_path}")

if __name__ == "__main__":
    # Load preprocessed data to determine input shape and number of classes
    X_train = np.load("data/processed/X_train_reshaped.npy")
    y_train = np.load("data/processed/y_train_one_hot.npy")
    
    input_shape = X_train.shape[1:]  # Shape of the input data (e.g., (28, 28, 1))
    num_classes = y_train.shape[1]  # Number of classes

    # Build the model
    model_cnn = build_model(input_shape, num_classes)

    # Save model architecture summary
    save_model_summary(model_cnn, output_path="models/model_architecture_summary.txt")
