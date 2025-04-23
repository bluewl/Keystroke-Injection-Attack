import pandas as pd

input_file = "../dataset_dont_commit/participants_subset_1_to_68000.csv"  # Use your actual file path
output_file = "../dataset_dont_commit/participants_subset_1_to_68000_normalized.csv"

# Load the data
df = pd.read_csv(input_file)

# Normalize keycode
df["KEYCODE"] = df["KEYCODE"] / 255.0

# Convert timing-related columns from ms to seconds
timing_columns = ["PRESS_TIME", "RELEASE_TIME", "HL", "IL", "PL", "RL"]
df[timing_columns] = df[timing_columns] / 1000.0

# Save to new CSV
df.to_csv(output_file, index=False)

print(f"Normalized data saved to '{output_file}'")
