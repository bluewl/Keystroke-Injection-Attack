import pandas as pd

# === CONFIG ===
input_file = './10k_subjects/first_10k_participants_normalized.csv'       # Replace with your actual input file
output_file = './10k_subjects/first_10k_participants_normalized_padded.csv'   # Output file to save the result
M = 70                              # Desired sequence length

# === LOAD DATA ===
df = pd.read_csv(input_file)

# === FUNCTION TO PAD A SINGLE SEQUENCE ===
def pad_sequence(group, M):
    current_length = len(group)
    if current_length < M:
        padding_needed = M - current_length
        pad = pd.DataFrame({
            'PARTICIPANT_ID': [group['PARTICIPANT_ID'].iloc[0]] * padding_needed,
            'TEST_SECTION_ID': [group['TEST_SECTION_ID'].iloc[0]] * padding_needed,
            'PRESS_TIME': [0.0] * padding_needed,
            'RELEASE_TIME': [0.0] * padding_needed,
            'KEYCODE': [0.0] * padding_needed,
            'HL': [0.0] * padding_needed,
            'IL': [0.0] * padding_needed,
            'PL': [0.0] * padding_needed,
            'RL': [0.0] * padding_needed,
        })
        group = pd.concat([group, pad], ignore_index=True)
    elif current_length > M:
        group = group.iloc[:M]
    return group

# === APPLY PADDING GROUPED BY PARTICIPANT + TEST_SECTION ===
grouped = df.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'], group_keys=False)
df_padded = grouped.apply(lambda x: pad_sequence(x, M))

# === SAVE OUTPUT ===
df_padded.to_csv(output_file, index=False)
print(f"Padded CSV saved to: {output_file}")
