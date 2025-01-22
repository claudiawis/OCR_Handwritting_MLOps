import unittest
import numpy as np
import os
from src.data.one_hot_encode_labels import one_hot_encode_labels

class TestOneHotEncodeLabels(unittest.TestCase):

    def setUp(self):
        # Temporary file paths
        self.y_train_path = "tests/test_data/tmp_Y_train.npy"
        self.y_test_path = "tests/test_data/tmp_Y_test.npy"
        self.output_train_path = "tests/test_data/tmp_y_train_one_hot.npy"
        self.output_test_path = "tests/test_data/tmp_y_test_one_hot.npy"

        # Create dummy labels
        Y_train = np.array([0, 1, 2, 0, 1, 2])
        Y_test = np.array([1, 2, 0, 2, 1, 0])
        np.save(self.y_train_path, Y_train)
        np.save(self.y_test_path, Y_test)

    def tearDown(self):
        # Remove temporary files
        for path in [self.y_train_path, self.y_test_path, self.output_train_path, self.output_test_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_one_hot_encode_labels(self):
        # Call the one-hot encoding function
        one_hot_encode_labels(self.y_train_path, self.y_test_path, self.output_train_path, self.output_test_path)

        # Load the saved one-hot encoded arrays
        y_train_one_hot = np.load(self.output_train_path)
        y_test_one_hot = np.load(self.output_test_path)

        # Verify the shapes of the one-hot encoded arrays
        self.assertEqual(y_train_one_hot.shape, (6, 3))
        self.assertEqual(y_test_one_hot.shape, (6, 3))

        # Verify the one-hot encoding logic
        self.assertTrue((y_train_one_hot[0] == [1, 0, 0]).all())
        self.assertTrue((y_train_one_hot[1] == [0, 1, 0]).all())
        self.assertTrue((y_train_one_hot[2] == [0, 0, 1]).all())

if __name__ == "__main__":
    unittest.main()
