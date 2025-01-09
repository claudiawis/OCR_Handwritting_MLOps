import unittest
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.encode_data import *  # Import everything from encode_data.py

class TestDataEncoding(unittest.TestCase):

    def test_encode_data(self):
        """
        Test that the encode_data function works correctly.
        """
        input_file = 'data/processed/cleaned_data.csv'
        output_file = 'data/processed/encoded_data.csv'

        # Ensure the output file doesn't exist before testing
        if os.path.exists(output_file):
            os.remove(output_file)

        # Call the data encoding process (runs the code in encode_data.py)
        # Since the script is procedural, we will invoke it using subprocess
        import subprocess
        result = subprocess.run(['python3', 'src/data/encode_data.py'], capture_output=True, text=True)

        # Check if the script ran successfully
        self.assertEqual(result.returncode, 0, "Script execution failed.")

        # Check if the output file was created
        self.assertTrue(os.path.exists(output_file), f"{output_file} was not created.")

        # Load the encoded data and verify it's correct
        df_encoded = pd.read_csv(output_file)

        # Verify that the 'transcription_encoded' column exists
        self.assertIn('transcription_encoded', df_encoded.columns, "Encoded column is missing.")

        # Verify that the encoding process created a valid transformation
        le = LabelEncoder()
        le.fit(df_encoded['transcription'])
        self.assertEqual(list(df_encoded['transcription_encoded']), list(le.transform(df_encoded['transcription'])),
                         "Encoded values do not match the expected encoding.")

if __name__ == '__main__':
    unittest.main()
