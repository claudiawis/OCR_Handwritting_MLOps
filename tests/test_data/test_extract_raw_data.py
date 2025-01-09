import unittest
from unittest.mock import patch
import os
from src.data.extract_raw_data import extract_tar_gz

class TestDataExtraction(unittest.TestCase):

    def test_data_extraction(self):
        """
        Test the extract_tar_gz function for correct extraction of .tar.gz data.
        """
        # Input for the extraction function
        tar_gz_path = 'data/raw/raw_data.tar.gz'  # Path to the .tar.gz file
        extract_path = 'data/raw/raw_data'  # Directory where data will be extracted

        # Ensure the extraction path is clean before testing
        if os.path.exists(extract_path):
            for file in os.listdir(extract_path):
                file_path = os.path.join(extract_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Call the function to extract data
        extract_tar_gz(tar_gz_path, extract_path)

        # Assert that the extracted files exist in the directory
        extracted_files = os.listdir(extract_path)
        self.assertGreater(len(extracted_files), 0, "No files were extracted.")

    @patch('builtins.print')
    def test_invalid_file_path(self, mock_print):
        """
        Test the extract_tar_gz function with an invalid file path.
        """
        # Invalid path for testing
        invalid_path = 'data/raw/nonexistent_file.tar.gz'
        extract_path = 'data/raw/raw_data'

        # Call the function with the invalid path and check the printed output
        extract_tar_gz(invalid_path, extract_path)

        # Check if the error message is printed
        mock_print.assert_called_with(f"Error: The file {invalid_path} does not exist.")

if __name__ == '__main__':
    unittest.main()
