import unittest
import numpy as np
import os
import logging
import subprocess
from src.data.split_data import *  # Import everything from split_data.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataSplitting(unittest.TestCase):

    def test_split_data(self):
        """
        Test that the split_data function works correctly.
        """
        # Define file paths
        X_train_path = 'data/processed/X_train.npy'
        X_test_path = 'data/processed/X_test.npy'
        Y_train_path = 'data/processed/Y_train.npy'
        Y_test_path = 'data/processed/Y_test.npy'

        # Ensure the output files do not exist before testing
        for path in [X_train_path, X_test_path, Y_train_path, Y_test_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Removed existing file: %s", path)

        # Call the data splitting process (runs the code in split_data.py)
        logger.info("Running the split_data script...")
        result = subprocess.run(['python3', 'src/data/split_data.py'], capture_output=True, text=True)

        # Check if the script ran successfully
        self.assertEqual(result.returncode, 0, "Script execution failed.")
        logger.info("Script executed successfully.")

        # Check if the files are created
        for path in [X_train_path, X_test_path, Y_train_path, Y_test_path]:
            self.assertTrue(os.path.exists(path), f"{path} was not created.")
            logger.info("Output file created: %s", path)

        # Load the split data files
        X_train = np.load(X_train_path)
        X_test = np.load(X_test_path)
        Y_train = np.load(Y_train_path)
        Y_test = np.load(Y_test_path)
        logger.info("Loaded data: X_train shape=%s, X_test shape=%s, Y_train shape=%s, Y_test shape=%s",
                    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

        # Check if the data is split into training and testing sets
        self.assertGreater(len(X_train), 0, "Training data is empty.")
        self.assertGreater(len(X_test), 0, "Test data is empty.")
        logger.info("Training and test data loaded successfully with non-zero lengths.")

        # Check that the split ratio is roughly 80% train and 20% test
        total_data = len(X_train) + len(X_test)
        self.assertAlmostEqual(len(X_train) / total_data, 0.8, delta=0.1,
                               msg="Training data is not approximately 80% of the total data.")
        self.assertAlmostEqual(len(X_test) / total_data, 0.2, delta=0.1,
                               msg="Test data is not approximately 20% of the total data.")
        logger.info("Data split ratio verified: Training=%d, Testing=%d", len(X_train), len(X_test))

if __name__ == '__main__':
    unittest.main()
