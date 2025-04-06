import pandas as pd

input_file = "OFFICIAL_extracted_features_limit70_sorted.csv"  # file path
output_file = "first_10k_participants.csv"

df = pd.read_csv(input_file)

unique_participants = df['PARTICIPANT_ID'].drop_duplicates().head(10000)

df_10k_participants = df[df['PARTICIPANT_ID'].isin(unique_participants)]

df_10k_participants.to_csv(output_file, index=False)

print(f"Saved data for first 10,000 unique participants to '{output_file}'")