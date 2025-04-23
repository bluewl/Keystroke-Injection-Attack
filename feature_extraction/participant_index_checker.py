import pandas as pd

# Input file path
input_file = "../dataset_dont_commit/OFFICIAL_extracted_features_limit70_sorted.csv"  # Use your actual file path
participant_id_to_check = 517947  # Replace with the actual ID

# Load data
df = pd.read_csv(input_file)

# Get unique participant IDs in the order they appear
unique_participants = df['PARTICIPANT_ID'].drop_duplicates().reset_index(drop=True)

# Check if the participant ID exists
if participant_id_to_check in unique_participants.values:
    order_number = unique_participants[unique_participants == participant_id_to_check].index[0] + 1  # 1-based index
    print(f"Participant ID '{participant_id_to_check}' is the {order_number}th unique participant.")
else:
    print(f"Participant ID '{participant_id_to_check}' was not found in the dataset.")
