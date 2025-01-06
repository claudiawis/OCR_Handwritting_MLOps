import unittest
from unittest import mock
import tarfile
import os
import logging
from src.data.extract_raw_data import extract_tar_gz  # Adjust the import path as necessary

class TestExtractTarGz(unittest.TestCase):

    @mock.patch('os.path.exists')
    @mock.patch('os.makedirs')
    def test_file_not_exist(self, mock_makedirs, mock_exists):
        # Arrange
        mock_exists.return_value = False  # Simulate that the file does not exist
        tar_gz_path = "invalid_path.tar.gz"
        extract_path = "extract_dir"

        # Capture the logging output
        with self.assertLogs('root', level='ERROR') as log:
            # Act
            extract_tar_gz(tar_gz_path, extract_path)
            # Assert
            self.assertIn(f"ERROR:root: The file {tar_gz_path} does not exist.", log.output)

    @mock.patch('os.path.exists')
    @mock.patch('os.makedirs')
    @mock.patch('tarfile.open')
    def test_directory_creation(self, mock_open, mock_makedirs, mock_exists):
        # Arrange
        mock_exists.side_effect = [True, False]  # Simulate file exists, dir does not
        tar_gz_path = "valid_path.tar.gz"
        extract_path = "extract_dir"

        # Act
        extract_tar_gz(tar_gz_path, extract_path)

        # Assert
        mock_makedirs.assert_called_once_with(extract_path, exist_ok=True)
        mock_open.assert_called_once_with(tar_gz_path, 'r:gz')

    @mock.patch('os.path.exists')
    @mock.patch('os.makedirs')
    @mock.patch('tarfile.open')
    def test_successful_extraction(self, mock_open, mock_makedirs, mock_exists):
        # Arrange
        mock_exists.return_value = True  # Simulate that the file exists
        mock_tar = mock.Mock()
        mock_open.return_value.__enter__.return_value = mock_tar  # Mock the tarfile object

        tar_gz_path = "valid_path.tar.gz"
        extract_path = "extract_dir"

        # Act
        extract_tar_gz(tar_gz_path, extract_path)

        # Assert
        mock_makedirs.assert_called_once_with(extract_path, exist_ok=True)
        mock_open.assert_called_once_with(tar_gz_path, 'r:gz')
        mock_tar.extractall.assert_called_once_with(path=extract_path)

if __name__ == '__main__':
    unittest.main()
