import unittest
import numpy as np
import os
from src.data.reshape_data import reshape_data

class TestDataReshaping(unittest.TestCase):

    def test_reshape_data(self):
        """
        Test that the reshape_data function works correctly.
        """
        input_data_path = 'data/processed'
        output_train_path = 'data/processed'
        output_test_path = 'data/processed'

        # File paths for reshaped data
        output_train_file = os.path.join(output_train_path, "X_train_reshaped.npy")
        output_test_file = os.path.join(output_test_path, "X_test_reshaped.npy")

        # Ensure the output files do not exist before testing
        for path in [output_train_file, output_test_file]:
            if os.path.exists(path):
                os.remove(path)

        # Call the reshape function
        reshape_data(input_data_path, output_train_path, output_test_path)

        # Check if the output reshaped files were created
        self.assertTrue(os.path.exists(output_train_file), f"{output_train_file} was not created.")
        self.assertTrue(os.path.exists(output_test_file), f"{output_test_file} was not created.")

        # Load the reshaped data
        X_train_reshaped = np.load(output_train_file)
        X_test_reshaped = np.load(output_test_file)

        # Check if the reshaped data has the correct shape (should be 28x28x1 for each image)
        self.assertEqual(X_train_reshaped.shape[1:], (28, 28, 1), "X_train_reshaped has an incorrect shape.")
        self.assertEqual(X_test_reshaped.shape[1:], (28, 28, 1), "X_test_reshaped has an incorrect shape.")

        # Check that the reshaped data is not empty
        self.assertGreater(X_train_reshaped.shape[0], 0, "Reshaped training data is empty.")
        self.assertGreater(X_test_reshaped.shape[0], 0, "Reshaped testing data is empty.")

if __name__ == '__main__':
    unittest.main()
