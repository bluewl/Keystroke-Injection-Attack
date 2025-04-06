import os
import csv

def merge_keystroke_files(input_dir, output_file):
    """
    Merges multiple keystroke log files into a single CSV file.

    :param input_dir: Directory that contains the *.txt keystroke files
    :param output_file: Path to the final CSV file
    """

    # We expect the columns to match what is specified in the readme.txt:
    # [PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE]
    # If your files have a slightly different format or columns, adjust as needed.

    # A place to store the header once we read it from the first file
    header = None

    # Open the combined CSV for writing (in UTF-8, but you can switch to cp1252 if you prefer)
    with open(output_file, mode="w", newline="", encoding="utf-8") as csv_out:
        writer = None  # We’ll create the writer after we know the columns

        # Walk through all files in the directory (and subdirectories if you like)
        for root, dirs, files in os.walk(input_dir):
            for filename in files:
                if not filename.endswith("_keystrokes.txt"):
                    # Skip anything not ending in `_keystrokes.txt`
                    continue

                full_path = os.path.join(root, filename)

                # Read each file using Windows-1252 (cp1252)
                # errors="replace" will let Python continue on any invalid bytes
                with open(full_path, mode="r", encoding="cp1252", errors="replace") as f_in:
                    lines = f_in.readlines()
                    # If the file is empty or too short, skip
                    if not lines:
                        continue

                    # The first line could be a header — let's detect or assume it.
                    possible_header = lines[0].strip()
                    columns_in_file = possible_header.split("\t")

                    # We check if columns_in_file matches the expected columns or not.
                    # If it does, skip it in this file but keep it to set up the CSV writer if needed.
                    if (header is None and 
                        "PARTICIPANT_ID" in columns_in_file and 
                        "KEYSTROKE_ID" in columns_in_file):
                        # This is a header line
                        header = columns_in_file

                        # Initialize CSV writer with the discovered header
                        writer = csv.writer(csv_out, delimiter=",")
                        writer.writerow(header)  # Write the header into the final CSV

                        # Now process the rest of the file as data rows
                        data_lines = lines[1:]
                    elif (header is not None and
                          "PARTICIPANT_ID" in columns_in_file and
                          "KEYSTROKE_ID" in columns_in_file):
                        # This file also has a header. We skip the first line
                        data_lines = lines[1:]
                    else:
                        # No recognized header: treat all lines as data
                        data_lines = lines

                    # If we still don’t have a writer, it means the very first file didn’t have a header at all
                    # so we must define some placeholder columns. Adjust as appropriate:
                    if writer is None and header is None:
                        # If columns are unknown, define them manually or from the first line
                        header = [
                            "PARTICIPANT_ID",
                            "TEST_SECTION_ID",
                            "SENTENCE",
                            "USER_INPUT",
                            "KEYSTROKE_ID",
                            "PRESS_TIME",
                            "RELEASE_TIME",
                            "LETTER",
                            "KEYCODE"
                        ]
                        writer = csv.writer(csv_out, delimiter=",")
                        writer.writerow(header)
                        # Since the first line was not a recognized header,
                        # we treat the entire file's lines as data
                        data_lines = lines

                    # Write each data line to the CSV
                    for line in data_lines:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue  # skip empty lines
                        row = line_stripped.split("\t")
                        writer.writerow(row)

    print(f"Merged keystroke data has been written to: {output_file}")

if __name__ == "__main__":
    # Example usage:
    # Suppose your 500k *_keystrokes.txt files are in "/path/to/keystrokes_data"
    # And you want to combine them into "all_keystrokes.csv"
    merge_keystroke_files(
        input_dir="./files",
        output_file="all_keystrokes.csv"
    )
