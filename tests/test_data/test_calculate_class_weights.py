import unittest
import numpy as np
import os
from src.data.calculate_class_weights import calculate_class_weights

class TestCalculateClassWeights(unittest.TestCase):

    def setUp(self):
        # Set up temporary paths for test files
        self.test_y_train_path = "tests/test_data/tmp_Y_train.npy"
        self.test_output_path = "tests/test_data/tmp_class_weights.npy"

        # Create a dummy training labels array
        self.y_train_data = np.array([0, 1, 0, 1, 1, 2, 2, 2])
        np.save(self.test_y_train_path, self.y_train_data)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_y_train_path):
            os.remove(self.test_y_train_path)
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    def test_calculate_class_weights(self):
        # Call the function to calculate class weights
        calculate_class_weights(self.test_y_train_path, self.test_output_path)

        # Load the output file
        class_weights = np.load(self.test_output_path, allow_pickle=True).item()

        # Assert the output is a dictionary
        self.assertIsInstance(class_weights, dict)

        # Assert the class weights are normalized (sum to ~1)
        self.assertAlmostEqual(sum(class_weights.values()), 1, places=6)

        # Assert each class has a corresponding weight
        expected_classes = [0, 1, 2]
        self.assertCountEqual(class_weights.keys(), expected_classes)

if __name__ == "__main__":
    unittest.main()
