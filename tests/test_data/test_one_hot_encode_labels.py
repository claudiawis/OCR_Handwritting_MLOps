import os
import unittest
import numpy as np
import logging
import contextlib

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs
import tensorflow as tf  # Import TensorFlow after setting the environment variable
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to ERROR level

# Configure logging for the test
logging.basicConfig(level=logging.WARNING)  # Set to WARNING to reduce verbosity
logger = logging.getLogger(__name__)

from src.data.one_hot_encode_labels import one_hot_encode_labels

# Context manager to suppress STDERR
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull):
            yield

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
        logger.info("Starting one-hot encoding process.")  # Optional: Keep this if needed

        with suppress_stderr():  # Suppress STDERR during the encoding process
            one_hot_encode_labels(self.y_train_path, self.y_test_path, self.output_train_path, self.output_test_path)

        # Load the saved one-hot encoded arrays
        y_train_one_hot = np.load(self.output_train_path)
        y_test_one_hot = np.load(self.output_test_path)

        # Verify the shapes of the one-hot encoded arrays
        self.assertEqual(y_train_one_hot.shape, (6, 3), "Unexpected shape for y_train_one_hot")
        self.assertEqual(y_test_one_hot.shape, (6, 3), "Unexpected shape for y_test_one_hot")

        # Verify the one-hot encoding logic
        self.assertTrue((y_train_one_hot[0] == [1, 0, 0]).all(), "First label not encoded correctly")
        self.assertTrue((y_train_one_hot[1] == [0, 1, 0]).all(), "Second label not encoded correctly")
        self.assertTrue((y_train_one_hot[2] == [0, 0, 1]).all(), "Third label not encoded correctly")

        logger.info("One-hot encoding logic verified for Y_train.")

if __name__ == "__main__":
    unittest.main()
