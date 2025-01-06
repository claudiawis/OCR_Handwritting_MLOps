import unittest
import numpy as np
import os
from src.data.prepare_features import prepare_features

class TestFeaturePreparation(unittest.TestCase):

    def test_prepare_features(self):
        """
        Test that the prepare_features function works correctly.
        """
        input_file = 'data/processed/encoded_data.csv'
        output_features_file = 'data/processed/features.npy'
        output_labels_file = 'data/processed/labels.npy'

        # Ensure the output files don't exist before testing
        if os.path.exists(output_features_file):
            os.remove(output_features_file)
        if os.path.exists(output_labels_file):
            os.remove(output_labels_file)

        # Call the feature preparation function
        prepare_features(input_file, output_features_file, output_labels_file, width=28, height=28)

        # Check if the output files were created
        self.assertTrue(os.path.exists(output_features_file), f"{output_features_file} was not created.")
        self.assertTrue(os.path.exists(output_labels_file), f"{output_labels_file} was not created.")

        # Load the saved feature and label files
        X = np.load(output_features_file)
        Y = np.load(output_labels_file)

        # Check if the features and labels have the correct shape
        self.assertEqual(X.shape[0], Y.shape[0], "Number of samples in features and labels don't match.")
        self.assertGreater(X.shape[0], 0, "No samples in features.")
        self.assertGreater(Y.shape[0], 0, "No samples in labels.")

        # Verify the range of pixel values (0 to 1 due to normalization)
        self.assertTrue((X >= 0).all() and (X <= 1).all(), "Feature values are not normalized properly.")

if __name__ == '__main__':
    unittest.main()
