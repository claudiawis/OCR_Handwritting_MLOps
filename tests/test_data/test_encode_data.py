import unittest
import os
import pandas as pd
import logging
import subprocess
from sklearn.preprocessing import LabelEncoder
from src.data.encode_data import *  # Import everything from encode_data.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataEncoding(unittest.TestCase):

    def setUp(self):
        self.input_file = 'data/processed/cleaned_data.csv'
        self.output_file = 'data/processed/encoded_data.csv'

    def test_encode_data(self):
        """ Test that the encode_data function works correctly. """

        # Ensure the output file doesn't exist before testing
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
            logger.info("Removed existing output file: %s", self.output_file)

        logger.info("Starting data encoding process...")
        # Call the data encoding process (runs the code in encode_data.py)
        result = subprocess.run(['python3', 'src/data/encode_data.py'], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, "Script execution failed.")
        logger.info("Data encoding script executed successfully.")

        # Check if the output file was created
        self.assertTrue(os.path.exists(self.output_file), f"{self.output_file} was not created.")
        logger.info("Output file created: %s", self.output_file)

        # Load the encoded data and verify it's correct
        df_encoded = pd.read_csv(self.output_file)

        # Verify that the 'transcription_encoded' column exists
        self.assertIn('transcription_encoded', df_encoded.columns, "Encoded column is missing.")
        logger.info("Encoded column 'transcription_encoded' is present.")

        # Verify that the encoding process created a valid transformation
        le = LabelEncoder()
        le.fit(df_encoded['transcription'])
        self.assertEqual(list(df_encoded['transcription_encoded']), list(le.transform(df_encoded['transcription'])), "Encoded values do not match the expected encoding.")
        logger.info("Encoded values match the expected encoding.")

if __name__ == '__main__':
    unittest.main()
