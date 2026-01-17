import json
import pandas as pd
import numpy as np
import ast
import os
import csv

# =========================
# CONFIGURATION
# =========================
SRS_INPUT_JSON  = "DataSet/SRS.json"
SRS_OUTPUT_CSV  = "pp_srs.csv"
E2_INPUT_CSV    = "P:/SP Challenge/DataSet/E2.csv"
TIME_STEP       = 0.01

def srs_preprocessing():
    """Logic from SRS_preprocessing.py"""
    print("--- Step 1: SRS Preprocessing ---")
    if not os.path.exists(SRS_INPUT_JSON):
        print(f"Error: {SRS_INPUT_JSON} not found. Skipping Step 1.")
        return False

    print("Reading JSON...")
    with open(SRS_INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    print("Cleaning and parsing SRS data...")
    drop_cols = ['_id', 'frame', 'slot', 'tx_port', 'rnti']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    def parse_srs(x):
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=np.float32)
        return np.array(x, dtype=np.float32)

    df['srs_ch'] = df['srs_ch'].apply(parse_srs)
    SRS_SHAPE = df['srs_ch'].iloc[0].shape  # Usually (2, 1536)

    print(f"Filling timestamp gaps (Step: {TIME_STEP})...")
    t_min = round(df['timestamp'].min(), 2)
    t_max = round(df['timestamp'].max(), 2)
    
    full_timestamps = np.round(np.arange(t_min, t_max + TIME_STEP, TIME_STEP), 2)
    rx_ants = [0, 1]

    full_index = pd.MultiIndex.from_product(
        [full_timestamps, rx_ants],
        names=['timestamp', 'rx_ant']
    )

    df = df.drop_duplicates(subset=['timestamp', 'rx_ant'])
    df = df.set_index(['timestamp', 'rx_ant']).reindex(full_index)

    # Fill missing with zero arrays
    zero_srs = np.zeros(SRS_SHAPE, dtype=np.float32)
    df['srs_ch'] = df['srs_ch'].apply(
        lambda x: zero_srs if not isinstance(x, np.ndarray) else x
    )
    df = df.reset_index()

    print("Stacking RX antennas...")
    df = df.sort_values(by=['timestamp', 'rx_ant'], ascending=[True, True])

    stacked_rows = []
    # Grouping ensures we combine antenna 0 and 1 for each timestamp
    for timestamp, group in df.groupby('timestamp'):
        srs_0 = group.iloc[0]['srs_ch']
        srs_1 = group.iloc[1]['srs_ch']

        # Vertical stack: (2, 1536) + (2, 1536) -> (4, 1536)
        stacked_srs = np.vstack([srs_0, srs_1])

        stacked_rows.append({
            'timestamp': timestamp,
            'srs_ch': stacked_srs.tolist()
        })

    out_df = pd.DataFrame(stacked_rows)
    out_df.to_csv(SRS_OUTPUT_CSV, index=False)
    print(f"✅ SRS Preprocessing complete. Saved to: {SRS_OUTPUT_CSV}")
    return True

def adjust_timeframe(file_paths):
    """Logic from adjust_timeframe.py (Modified to ask for both limits)"""
    print("\n--- Step 2: Adjust Timeframe ---")
    try:
        start_limit = int(input("Enter the integer part of the START timestamp (rows BEFORE this will be deleted): "))
        end_limit = int(input("Enter the integer part of the END timestamp (rows AFTER this will be deleted): "))
    except ValueError:
        print("Invalid input. Skipping timeframe adjustment.")
        return

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Filter rows to be within the range [start_limit, end_limit]
            filtered_rows = [
                row for row in rows 
                if start_limit <= int(float(row['timestamp'])) <= end_limit
            ]

            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)

            print(f"✅ Filtered {file_path}: Kept rows between {start_limit} and {end_limit}.")
        except KeyError:
            print(f"Error: {file_path} is missing the 'timestamp' column.")
        except Exception as e:
            print(f"An error occurred with {file_path}: {e}")

def delete_unnecessary_columns():
    """Logic from delete_columns.py"""
    print("\n--- Step 3: Delete Columns ---")
    
    tasks = [
        (SRS_OUTPUT_CSV, ["timestamp"]),
        (E2_INPUT_CSV, ["_id", "ue_id", "timestamp", "cellid", "rnti", "pmi"])
    ]

    for file_path, cols in tasks:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping column deletion.")
            continue

        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                new_fieldnames = [f for f in reader.fieldnames if f not in cols]

            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=new_fieldnames)
                writer.writeheader()
                for row in rows:
                    filtered_row = {k: v for k, v in row.items() if k in new_fieldnames}
                    writer.writerow(filtered_row)
            
            print(f"✅ Deleted columns {cols} from {file_path}.")
        except Exception as e:
            print(f"An error occurred with {file_path}: {e}")

def remove_header_row(file_path):
    """Removes the first (header) row from the CSV file."""
    print(f"\n--- Step 4: Removing Header from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping header removal.")
        return

    try:
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Write lines back starting from the second line (index 1)
        if len(lines) > 0:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.writelines(lines[1:])
            print(f"✅ Successfully removed header row from {file_path}.")
        else:
            print(f"Warning: {file_path} is empty.")
    except Exception as e:
        print(f"An error occurred while removing header from {file_path}: {e}")

def main():
    # 1. Preprocess SRS (creates pp_srs.csv)
    success = srs_preprocessing()
    
    # 2. Adjust timeframe for both files
    # We only proceed if files exist
    adjust_timeframe([SRS_OUTPUT_CSV, E2_INPUT_CSV])
    
    # 3. Clean up columns
    delete_unnecessary_columns()

    # 4. Remove header row from pp_srs.csv as the final step
    remove_header_row(SRS_OUTPUT_CSV)
    
    print("\n" + "="*30)
    print("✅ Full Pipeline Completed Successfully!")
    print("="*30)

if __name__ == "__main__":
    main()