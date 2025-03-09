# feature_extraction.py

# NOTE: if keystrokes.csv starts with keyup (no keydown pair), the extraction will fail

import pandas as pd
import numpy as np

# Load keystroke data
df = pd.read_csv("keystrokes.csv", names=["key", "keyCode", "type", "timestamp"])

# Convert timestamps to seconds
df["timestamp"] = df["timestamp"] / 1000.0  # Convert ms → seconds

# Separate keydown and keyup events
keydowns = df[df["type"] == "keydown"].reset_index(drop=True)
keyups = df[df["type"] == "keyup"].reset_index(drop=True)

print(keydowns.head())
print(keyups.head())

# Ensure both keydown and keyup have the same length
if len(keydowns) != len(keyups):
    min_len = min(len(keydowns), len(keyups))
    keydowns = keydowns[:min_len]
    keyups = keyups[:min_len]

features = []

for i in range(1, len(keydowns)):
    HL = keyups["timestamp"][i] - keydowns["timestamp"][i]  # Hold Latency
    IL = keydowns["timestamp"][i] - keyups["timestamp"][i-1]  # Inter-key Latency
    PL = keydowns["timestamp"][i] - keydowns["timestamp"][i-1]  # Press Latency
    RL = keyups["timestamp"][i] - keyups["timestamp"][i-1]  # Release Latency
    keycode = keydowns["keyCode"][i] / 255.0  # Normalize keycode (0-1)

    features.append([HL, IL, PL, RL, keycode])

# Convert to DataFrame
feature_df = pd.DataFrame(features, columns=["HL", "IL", "PL", "RL", "Keycode"])

# Save extracted features
feature_df.to_csv("keystroke_features.csv", index=False)

print("Keystroke feature extraction done.")
