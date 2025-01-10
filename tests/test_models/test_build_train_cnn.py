import unittest
import os
import numpy as np
from tensorflow.keras.models import Model
from src.models.build_train_cnn import build_model, save_model_summary

class TestBuildTrainCNN(unittest.TestCase):

    def setUp(self):
        # Temporary directories and file paths
        self.test_output_path = "tests/test_models/tmp_model_summary.txt"
        os.makedirs(os.path.dirname(self.test_output_path), exist_ok=True)

    def tearDown(self):
        # Remove temporary files
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    def test_build_model(self):
        # Define dummy input shape and number of classes
        input_shape = (28, 28, 1)
        num_classes = 10

        # Build the model
        model = build_model(input_shape, num_classes)

        # Check if the model is a Keras Model instance
        self.assertIsInstance(model, Model)

        # Verify the output layer shape
        self.assertEqual(model.output_shape, (None, num_classes))

    def test_save_model_summary(self):
        # Define dummy input shape and number of classes
        input_shape = (28, 28, 1)
        num_classes = 10

        # Build the model
        model = build_model(input_shape, num_classes)

        # Save the model summary
        save_model_summary(model, output_path=self.test_output_path)

        # Verify the summary file was created
        self.assertTrue(os.path.exists(self.test_output_path))

        # Check if the file contains the summary content
        with open(self.test_output_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Layer (type)", content)
        self.assertIn("Output Shape", content)

if __name__ == "__main__":
    unittest.main()
