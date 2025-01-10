import unittest
import subprocess
import os

class TestDataCleaning(unittest.TestCase):

    def test_clean_data(self):
        """
        Test the clean_data function for correct data cleaning.
        """
        # Define input and output file paths for testing
        input_file = 'data/processed/filtered_data.csv'
        output_file = 'data/processed/cleaned_data.csv'

        # Ensure the output file doesn't exist before testing
        if os.path.exists(output_file):
            os.remove(output_file)

        # Run the original script using subprocess
        result = subprocess.run(['python3', 'src/data/clean_data.py'], capture_output=True, text=True)

        # Check if the script ran successfully
        self.assertEqual(result.returncode, 0, "Script execution failed.")

        # Check if the output file was created
        self.assertTrue(os.path.exists(output_file), f"{output_file} was not created.")

        # Load the cleaned data and verify it's correct
        import pandas as pd
        df_cleaned = pd.read_csv(output_file)
        self.assertGreater(df_cleaned['transcription'].nunique(), 0, "No unique transcriptions found in the cleaned data.")

        # Assert that no stopwords or unwanted symbols exist in the cleaned data
        stopwords = {'the', 'and', 'is', ')', ':', '...', "'s"}
        for value in df_cleaned['transcription']:
            self.assertNotIn(value, stopwords, f"Stopword '{value}' found in cleaned data.")

if __name__ == '__main__':
    unittest.main()
