import os
import glob
import pandas as pd
import numpy as np

# dataset is shared in cloud  

M = 70  # Max keystrokes per sentence

def parse_and_extract_features(file_path):
    """
    Reads one *_keystrokes.txt file line by line, extracting columns:
      - PARTICIPANT_ID
      - TEST_SECTION_ID
      - PRESS_TIME
      - RELEASE_TIME
      - KEYCODE

    For each (PARTICIPANT_ID, TEST_SECTION_ID) group:
      1) Keep only the first M keystrokes.
      2) Compute TypeNet latencies:
         HL = RELEASE_TIME[i] - PRESS_TIME[i]
         IL = PRESS_TIME[i+1]  - RELEASE_TIME[i]
         PL = PRESS_TIME[i+1]  - PRESS_TIME[i]
         RL = RELEASE_TIME[i+1] - RELEASE_TIME[i]
      3) Replace NaN latencies with 0.
    """

    valid_rows = []
    try:
        with open(file_path, 'r', encoding='cp1252', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Cannot open {file_path} due to: {e}")
        return pd.DataFrame()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 9:
            continue

        participant_id   = parts[0]
        test_section_id  = parts[1]
        press_time_str   = parts[5]
        release_time_str = parts[6]
        keycode_str      = parts[8]

        try:
            press_t   = float(press_time_str)
            release_t = float(release_time_str)
            keycode   = float(keycode_str)
        except ValueError:
            continue

        valid_rows.append({
            "PARTICIPANT_ID": participant_id,
            "TEST_SECTION_ID": test_section_id,
            "PRESS_TIME": press_t,
            "RELEASE_TIME": release_t,
            "KEYCODE": keycode
        })

    df = pd.DataFrame(valid_rows)
    if df.empty:
        return df

    df.sort_values(["PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    feat_list = []

    
    for (pid, secid), grp in df.groupby(["PARTICIPANT_ID", "TEST_SECTION_ID"], sort=False):
        g = grp.iloc[:M+1].copy()
        if g.empty:
            continue

        g["HL"] = g["RELEASE_TIME"] - g["PRESS_TIME"]

        shifted_press   = g["PRESS_TIME"].shift(-1)
        shifted_release = g["RELEASE_TIME"].shift(-1)

        g["IL"] = shifted_press - g["RELEASE_TIME"]
        g["PL"] = shifted_press - g["PRESS_TIME"]
        g["RL"] = shifted_release - g["RELEASE_TIME"]

        g[["IL", "PL", "RL"]] = g[["IL", "PL", "RL"]].fillna(0)

        g = g.iloc[:M]
        feat_list.append(g)

    if not feat_list:
        return pd.DataFrame()

    return pd.concat(feat_list, ignore_index=True)

def process_all_txt(folder_path, output_csv="typinet_features_merged.csv"):
    """
    - Processes all *_keystrokes.txt files in a folder
    - Limits to 70 keystrokes per sentence
    - Computes HL, IL, PL, RL
    - Sorts by PARTICIPANT_ID, TEST_SECTION_ID, PRESS_TIME (numerically)
    - Outputs a merged CSV
    """
    pattern = os.path.join(folder_path, "*_keystrokes.txt")
    files = glob.glob(pattern)
    print(f"Found {len(files)} keystroke txt files in: {folder_path}")

    big_dfs = []
    for fpath in files:
        df_feat = parse_and_extract_features(fpath)
        if not df_feat.empty:
            big_dfs.append(df_feat)

    if not big_dfs:
        print("No valid data extracted from any file.")
        return

    merged_df = pd.concat(big_dfs, ignore_index=True)

    # Convert IDs to int for strict numeric sorting
    merged_df["PARTICIPANT_ID"] = merged_df["PARTICIPANT_ID"].astype(int)
    merged_df["TEST_SECTION_ID"] = merged_df["TEST_SECTION_ID"].astype(int)

    # Final global sort
    merged_df.sort_values(["PARTICIPANT_ID", "TEST_SECTION_ID", "PRESS_TIME"], inplace=True)

    merged_df.to_csv(output_csv, index=False)
    print(f"Saved {len(merged_df)} rows to {output_csv}")

if __name__ == "__main__":
    folder = "./files"
    outcsv = "OFFICIAL_extracted_features_limit70_sorted.csv"
    process_all_txt(folder, outcsv)
