'''
To use categorical data such as transcriptions in a machine learning model, we need to convert them into a numerical format. 
In this step, we use LabelEncoder to transform each unique transcription into a unique integer label. This encoded format makes 
the labels usable in the deep learning model.
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the input CSV file
input_file = 'data/processed/cleaned_data.csv'
df_cleaned = pd.read_csv(input_file)

# Initialize the LabelEncoder
le = LabelEncoder()

# Encode the transcriptions and add to the DataFrame
df_cleaned['transcription_encoded'] = le.fit_transform(
    df_cleaned['transcription'])

# Create a mapping between transcriptions and their encoded labels
transcription_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Output the encoded data to a new CSV file
output_file = 'data/processed/encoded_data.csv'
df_cleaned.to_csv(output_file, index=False)

print('Successfully encoded data and saved to', output_file)
