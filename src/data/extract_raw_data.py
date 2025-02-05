'''
In this section, we extract the contents of a .tar.gz archive from a specified path to a designated directory. 
This process involves verifying the existence of the archive file and ensuring that the target extraction directory is created if it does not already exist.

- Archive Path: We define the path to the .tar.gz file that needs to be extracted.
- Extraction Directory: We specify the directory where the contents of the archive will be placed. If this directory does not exist, it will be created automatically.
- Extraction Process: We utilize the tarfile module to open the archive and extract all files into the specified directory, 
  ensuring that the files are properly placed for further use.
- Error Handling: The function includes checks to confirm the presence of the archive file, and it outputs an error message if the file is missing, 
  preventing any extraction attempts.
'''

import tarfile
import os
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR)


def extract_tar_gz(tar_gz_path, extract_path):
    # Check if the .tar.gz file exists
    if not os.path.exists(tar_gz_path):
        print(f"Error: The file {tar_gz_path} does not exist.")
        return

    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # Extract the .tar.gz file
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted {tar_gz_path} to {extract_path}")


if __name__ == "__main__":
    tar_gz_path = 'data/raw/raw_data.tar.gz'
    extract_path = 'data/raw/raw_data'
    extract_tar_gz(tar_gz_path, extract_path)