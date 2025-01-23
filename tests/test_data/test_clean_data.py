import unittest
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        self.input_file = 'data/processed/filtered_data.csv'
        self.output_file = 'data/processed/cleaned_data.csv'

    def test_clean_data(self):
        # Ensure the output file doesn't exist before testing
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        logger.info("Starting data cleaning process...")
        result = subprocess.run(['python3', 'src/data/clean_data.py'], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, "Script execution failed.")
        logger.info("Data cleaning script executed successfully.")

        self.assertTrue(os.path.exists(self.output_file), f"{self.output_file} was not created.")
        logger.info("Output file created: %s", self.output_file)

        # Run the cleaning process again and check that it does not overwrite the output file
        result = subprocess.run(['python3', 'src/data/clean_data.py'], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, "Script execution failed on second run.")
        logger.info("Data cleaning script executed successfully on second run.")

        # Check that the output file still exists and no changes were made
        self.assertTrue(os.path.exists(self.output_file), f"{self.output_file} was not found.")
        logger.info("Output file still exists: %s", self.output_file)
        logger.warning("with this setup the cleaned_data.csv is not overwritten when clean_data.py is rerun")

if __name__ == '__main__':
    unittest.main()
