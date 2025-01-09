import unittest
import pandas as pd
import os
from src.data.filter_data import filter_data

class TestFilterData(unittest.TestCase):

    def setUp(self):
        # Create a temporary input file
        self.input_file = "tests/test_data/tmp_words.csv"
        self.output_file = "tests/test_data/tmp_filtered_data.csv"

        # Create a sample DataFrame
        data = {
            'transcription': ['word1', 'word1', 'word2', 'word3', 'word3', 'word3', 
                              'word4', 'word4', 'word4', 'word4'],
        }
        self.df = pd.DataFrame(data)
        self.df.to_csv(self.input_file, index=False)

    def tearDown(self):
        # Remove temporary files
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_filter_data(self):
        # Call the filter_data function
        min_samples = 2
        max_samples = 3

        filter_data(self.input_file, self.output_file, min_samples, max_samples)

        # Read the filtered output file
        filtered_df = pd.read_csv(self.output_file)

        # Check if the filtered file contains only the expected rows
        expected_transcriptions = ['word1', 'word3']
        self.assertCountEqual(filtered_df['transcription'].unique(), expected_transcriptions)

        # Check if the index is reset
        self.assertEqual(filtered_df.index[0], 0)

if __name__ == "__main__":
    unittest.main()
