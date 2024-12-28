import pandas as pd

# Set the minimum and maximum sample counts for filtering transcriptions
min_samples = 100  # Minimum number of occurrences for a transcription to be included
max_samples = 200  # Maximum number of occurrences for a transcription to be included

# Read the input CSV file
input_file = 'data/raw/words.csv'
df = pd.read_csv(input_file)

# Filter transcriptions based on the specified count thresholds
class_counts = df['transcription'].value_counts()  # Get the count of each transcription
classes_to_keep = class_counts[(class_counts >= min_samples) & (class_counts <= max_samples)].index  # Identify transcriptions to retain
df_filtered = df[df['transcription'].isin(classes_to_keep)].copy()  # Create a filtered DataFrame containing only the selected transcriptions

# Reset index after filtering to ensure a clean, continuous index
df_filtered.reset_index(drop=True, inplace=True)

# Output the filtered data to a new CSV file
output_file = 'data/processed/filtered_data.csv'
df_filtered.to_csv(output_file, index=False)

# Print unique transcriptions remaining in the filtered dataset
print("Unique transcriptions remaining in the filtered dataset:")
print(df_filtered['transcription'].unique())  # Helps to verify the diversity of retained transcriptions
