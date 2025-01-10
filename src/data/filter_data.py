'''
To improve the quality and balance of the dataset, we filter the transcriptions based on their frequency. 
Transcriptions that occur either too frequently or too rarely can lead to imbalance or overfitting during model training. In this section, we:

1. Set a range for transcription frequency.
2. Filter the dataset to keep only transcriptions that fall within the desired range.
3. Reset the index to ensure the filtered DataFrame has a clean, continuous index.
4. Print the unique transcriptions in the filtered dataset to verify the diversity.
'''

import pandas as pd

def filter_data(input_file, output_file, min_samples, max_samples):
    """
    Filters the dataset based on the frequency of transcriptions.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the filtered CSV file.
        min_samples (int): Minimum number of occurrences for a transcription to be kept.
        max_samples (int): Maximum number of occurrences for a transcription to be kept.
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Filter transcriptions based on the specified count thresholds
    class_counts = df['transcription'].value_counts()
    classes_to_keep = class_counts[(class_counts >= min_samples) & (class_counts <= max_samples)].index
    df_filtered = df[df['transcription'].isin(classes_to_keep)].copy()

    # Reset index after filtering to ensure a clean, continuous index
    df_filtered.reset_index(drop=True, inplace=True)

    # Output the filtered data to a new CSV file
    df_filtered.to_csv(output_file, index=False)

    # Print unique transcriptions remaining in the filtered dataset
    print("Successfully filtered data and saved to", output_file)
    print("Unique transcriptions remaining in the filtered dataset:")
    print(df_filtered['transcription'].unique())

if __name__ == "__main__":
    input_file = 'data/raw/words.csv'
    output_file = 'data/processed/filtered_data.csv'
    min_samples = 100
    max_samples = 200
    filter_data(input_file, output_file, min_samples, max_samples)
