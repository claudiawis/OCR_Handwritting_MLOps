import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download stopwords list from NLTK
nltk.download('stopwords')

# Create a set of English stopwords
stop_words = set(stopwords.words('english'))

# Remove specific unwanted transcriptions by adding symbols to the stopwords set
stop_words.update([')', ':', '...', "'s"])  # Adding symbols and suffixes to the stopwords set for further cleaning

# Read the input CSV file
input_file = 'data/processed/filtered_data.csv'
df_filtered = pd.read_csv(input_file)

# Filter out transcriptions that are in the stopwords list
df_cleaned = df_filtered[~df_filtered['transcription'].isin(stop_words)].copy()  # Remove transcriptions that match stopwords

# Reset index after filtering to ensure a clean, continuous index
df_cleaned.reset_index(drop=True, inplace=True)

# Output the cleaned data to a new CSV file
output_file = 'data/processed/cleaned_data.csv'
df_cleaned.to_csv(output_file, index=False)

# Print the number of unique transcriptions remaining in the cleaned dataset
print('Number of remained unique values: ', df_cleaned['transcription'].nunique())  # Display the count of unique values to verify filtering

# Print the unique transcriptions remaining in the dataset
print('Remained unique values: ', df_cleaned['transcription'].unique())  # Display the remaining unique transcriptions for verification