import pandas as pd

def fix_zero_timings(input_file, output_file, tolerance=1e-6):
    """
    Replace IL, PL, RL with 0.001 where:
    - PRESS_TIME, RELEASE_TIME, KEYCODE, and HL are NOT zero
    - IL, PL, RL are approximately zero (within given tolerance)
    """

    # Load the CSV
    df = pd.read_csv(input_file)

    # Build the condition
    condition = (
        (df["PRESS_TIME"].abs() > tolerance) &
        (df["RELEASE_TIME"].abs() > tolerance) &
        (df["KEYCODE"].abs() > tolerance) &
        (df["HL"].abs() > tolerance) &
        (df["IL"].abs() < tolerance) &
        (df["PL"].abs() < tolerance) &
        (df["RL"].abs() < tolerance)
    )

    # Apply the fix
    df.loc[condition, ["IL", "PL", "RL"]] = 0.001

    # Save the modified data
    df.to_csv(output_file, index=False)
    print(f"Fixed file saved to: {output_file}")

# Example usage:
fix_zero_timings("../dataset_dont_commit/first_10k_participants_normalized_padded.csv", "../dataset_dont_commit/first_10k_participants_normalized_padded_MODIFIED.csv")
