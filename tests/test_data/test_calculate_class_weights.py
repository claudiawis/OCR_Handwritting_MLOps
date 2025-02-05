import unittest
import numpy as np
import os
import tempfile
import logging
from src.data.calculate_class_weights import calculate_class_weights

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set to WARNING or ERROR to reduce output during tests
logger = logging.getLogger(__name__)

class TestCalculateClassWeights(unittest.TestCase):

    def setUp(self):
        """
        Set up temporary files for testing.
        """
        # Create a temporary input file with dummy data
        self.input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        self.output_file = 'data/processed/class_weights.npy'  # This will be created by the script

        # Create a dummy training labels array
        self.y_train_data = np.array([0, 1, 0, 1, 1, 2, 2, 2])
        np.save(self.input_file.name, self.y_train_data)

    def tearDown(self):
        """
        Clean up temporary files after tests.
        """
        os.remove(self.input_file.name)  # Clean up the temporary input file
        if os.path.exists(self.output_file):
            os.remove(self.output_file)  # Clean up the output file if it exists

    def test_calculate_class_weights(self):
        """
        Test the calculate_class_weights function for correct output.
        """
        # Call the function to calculate class weights
        logger.info("Calculating class weights from input file: %s", self.input_file.name)
        calculate_class_weights(self.input_file.name, self.output_file)

        # Load the output file
        class_weights = np.load(self.output_file, allow_pickle=True).item()

        # Assert the output is a dictionary
        self.assertIsInstance(class_weights, dict)

        # Assert the class weights are normalized (sum to ~1)
        self.assertAlmostEqual(sum(class_weights.values()), 1, places=6)

        # Assert each class has a corresponding weight
        expected_classes = [0, 1, 2]
        self.assertCountEqual(class_weights.keys(), expected_classes)

        # Assert specific expected weights (example values)
        expected_weights = {0: 0.25, 1: 0.375, 2: 0.375}
        for cls, weight in expected_weights.items():
            self.assertAlmostEqual(class_weights[cls], weight, places=6)
            logger.info("Class %d: Expected weight = %.3f, Actual weight = %.3f", cls, weight, class_weights[cls])

if __name__ == "__main__":
    unittest.main()
