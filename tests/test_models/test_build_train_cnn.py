import unittest
import os
import numpy as np
import logging
import contextlib
import sys
from tensorflow.keras.models import Model
from src.models.build_train_cnn import build_model, save_model_summary

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs
import tensorflow as tf  # Import TensorFlow after setting the environment variable
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to ERROR level

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context manager to suppress STDERR
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull):
            yield

class TestBuildTrainCNN(unittest.TestCase):

    def setUp(self):
        # Temporary directories and file paths
        self.test_output_path = "tests/test_models/tmp_model_summary.txt"
        os.makedirs(os.path.dirname(self.test_output_path), exist_ok=True)
        logger.info("Temporary output path created: %s", self.test_output_path)

    def tearDown(self):
        # Remove temporary files
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)
            logger.info("Removed temporary file: %s", self.test_output_path)

    def test_build_model(self):
        # Define dummy input shape and number of classes
        input_shape = (28, 28, 1)
        num_classes = 10
        logger.info("Building model with input shape: %s and num_classes: %d", input_shape, num_classes)

        # Build the model
        model = build_model(input_shape, num_classes)

        # Check if the model is a Keras Model instance
        self.assertIsInstance(model, Model, "The model is not an instance of Keras Model.")
        logger.info("Successfully built model: %s", model.summary())

        # Verify the output layer shape
        self.assertEqual(model.output_shape, (None, num_classes), "Output shape does not match expected shape.")
        logger.info("Output layer shape verified: %s", model.output_shape)

    def test_save_model_summary(self):
        # Define dummy input shape and number of classes
        input_shape = (28, 28, 1)
        num_classes = 10
        logger.info("Building model for saving summary with input shape: %s and num_classes: %d", input_shape, num_classes)

        # Build the model
        model = build_model(input_shape, num_classes)

        # Save the model summary
        with suppress_stderr():  # Suppress STDERR during the saving process
            save_model_summary(model, output_path=self.test_output_path)

        # Verify the summary file was created
        self.assertTrue(os.path.exists(self.test_output_path), "The model summary file was not created.")
        logger.info("Model summary file created: %s", self.test_output_path)

        # Check if the file contains the summary content
        with open(self.test_output_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Layer (type)", content, "Summary does not contain 'Layer (type)'.")
        self.assertIn("Output Shape", content, "Summary does not contain 'Output Shape'.")

if __name__ == "__main__":
    with suppress_stderr():  # Suppress STDERR during the entire test run
        unittest.main()
