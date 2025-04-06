import pandas as pd
import numpy as np

# dataset is shared in cloud  

def extract_typenet_features(csv_path, output_csv=None):
    """
    Reads a keystroke CSV with columns at least:
       PARTICIPANT_ID, PRESS_TIME, RELEASE_TIME, KEYCODE
    Returns a new DataFrame with columns:
       PARTICIPANT_ID, INDEX_IN_SEQ,
       HL, IL, PL, RL, KEYCODE
    If output_csv is provided, saves to that file.
    """
    # 1) Load the CSV
    df = pd.read_csv(csv_path, encoding="cp1252")


    # 2) Sort by user and press_time so that consecutive keystrokes line up
    df.sort_values(by=["PARTICIPANT_ID", "PRESS_TIME"], inplace=True)

    # 3) Group by participant so we can compute pairwise latencies
    #    If you also want to break data by "test_section_id" or "SENTENCE",
    #    you can group by multiple columns, e.g. ["PARTICIPANT_ID","TEST_SECTION_ID"]
    grouped = df.groupby("PARTICIPANT_ID", sort=False)

    all_rows = []
    for user_id, group in grouped:
        # Convert to numpy arrays for easy shifted indexing
        press_times   = group["PRESS_TIME"].values
        release_times = group["RELEASE_TIME"].values
        keycodes      = group["KEYCODE"].values

        # We'll compute latencies for each keystroke i
        # HL[i] = release[i] - press[i]
        HL = release_times - press_times

        # IL[i] = press[i+1] - release[i], but we can’t compute it for the last keystroke
        IL = np.full_like(HL, np.nan)  # same length
        IL[:-1] = press_times[1:] - release_times[:-1]

        # PL[i] = press[i+1] - press[i]
        PL = np.full_like(HL, np.nan)
        PL[:-1] = press_times[1:] - press_times[:-1]

        # RL[i] = release[i+1] - release[i]
        RL = np.full_like(HL, np.nan)
        RL[:-1] = release_times[1:] - release_times[:-1]

        # Prepare a smaller DataFrame with these columns
        # We'll store the index in the sequence too, for reference
        sub_df = pd.DataFrame({
            "PARTICIPANT_ID": user_id,
            "INDEX_IN_SEQ": range(len(group)),
            "HL": HL,
            "IL": IL,
            "PL": PL,
            "RL": RL,
            "KEYCODE": keycodes,
        })

        all_rows.append(sub_df)

    # Combine all sub-dfs for each user
    feat_df = pd.concat(all_rows, ignore_index=True)

    # 4) (Optional) You might want to drop the last row per user if it has NaN in IL/PL/RL
    #    because we can’t compute those for the final keystroke.
    #    For illustration, we’ll just keep them but you can do:
    # feat_df.dropna(subset=["IL","PL","RL"], inplace=True)

    # 5) Save or return
    if output_csv is not None:
        feat_df.to_csv(output_csv, index=False)

    return feat_df

if __name__ == "__main__":
    # Example usage:
    csv_input_path = "all_keystrokes.csv"  # your big CSV
    csv_output_path = "extracted_features.csv"
    df_features = extract_typenet_features(csv_input_path, csv_output_path)
    print(f"Features saved to {csv_output_path}. Rows: {len(df_features)}")
