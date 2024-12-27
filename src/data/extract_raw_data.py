import tarfile
import os

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
    # Define the path to the .tar.gz file and the extraction directory
    tar_gz_path = 'data/raw/raw_data.tar.gz'
    extract_path = 'data/raw/raw_data'

    # Call the extraction function
    extract_tar_gz(tar_gz_path, extract_path)