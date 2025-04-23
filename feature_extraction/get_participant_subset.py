import pandas as pd

# This is similar to get_first_10k_participants.py but allows for a range of participants to be selected.

# Input file path
input_file = "../dataset_dont_commit/OFFICIAL_extracted_features_limit70_sorted.csv"  # Use your actual file path
output_base = "../dataset_dont_commit/participants_subset_"  # Base name for output files

# Parameters (CHANGE THIS TO SELECT DIFFERENT PARTICIPANTS)
start_participant_index = 0  # 0-based index, so 58001st is index 58000
num_participants_to_extract = 68000  # Change this to 1000, 10000, or 100000

# Load data
df = pd.read_csv(input_file)

# Get the list of unique participant IDs
unique_participants = df['PARTICIPANT_ID'].drop_duplicates()

# Get the target range of participant IDs
selected_participants = unique_participants.iloc[start_participant_index : start_participant_index + num_participants_to_extract]

# Filter the original dataframe to only include rows from the selected participants
df_subset = df[df['PARTICIPANT_ID'].isin(selected_participants)]

# Output file
output_file = f"{output_base}{start_participant_index + 1}_to_{start_participant_index + num_participants_to_extract}.csv"
df_subset.to_csv(output_file, index=False)

print(f"Saved data for participants {start_participant_index + 1} to {start_participant_index + num_participants_to_extract} into '{output_file}'")
