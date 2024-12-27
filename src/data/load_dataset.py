import os
import pandas as pd
from tqdm import tqdm

def load_images(base_path, words_path):
    data = []
    inexistent_or_corrupted = 0

    # Read words from the specified text file
    with open(words_path, "r") as f:
        words = f.readlines()

    # Iterate through each line in the words file
    for line in tqdm(words):
        if line.startswith("#"):
            continue  # Skip comment lines
        parts = line.strip().split()

        # Extract fixed parts and transcription
        fixed_part = parts[:8]
        transcription_part = ' '.join(parts[8:])

        # Derive folder and file names from the first part of the line
        line_split = line.split(" ")
        folder_parts = line_split[0].split('-')
        folder1 = folder_parts[0]
        folder2 = folder_parts[0] + '-' + folder_parts[1]
        file_name = line_split[0] + ".png"
        rel_path = os.path.join(base_path, folder1, folder2, file_name)

        # Check if the image file exists and is not empty
        if os.path.exists(rel_path) and os.path.getsize(rel_path) > 0:
            data.append(fixed_part + [transcription_part, rel_path])
        else:
            inexistent_or_corrupted += 1

    print('Inexistent or corrupted files:', inexistent_or_corrupted)
    return pd.DataFrame(data, columns=['line_id', 'result', 'graylevel', 'x', 'y', 'w', 'h', 'annotation', 'transcription', 'image_path'])

if __name__ == "__main__":
    # Define paths
    dataset_path = 'data/raw/raw_data/data/raw/words'  # Update the path as necessary
    words_file_path = "data/raw/raw_data/data/raw/ascii/words.txt"  # Path to words.txt

    # Load images and create DataFrame
    df = load_images(dataset_path, words_file_path)

    # Save the DataFrame to a CSV file
    output_csv_path = 'data/raw/words.csv'
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # Create the directory if it doesn't exist
    df.to_csv(output_csv_path, index=False)  # Save the DataFrame to CSV without the index
    print(f'Data saved to {output_csv_path}')