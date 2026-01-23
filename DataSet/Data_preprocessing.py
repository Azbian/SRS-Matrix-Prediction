import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Union
import re
import ast
import argparse

def get_time_frames(file_path: Union[str, Path]) -> Dict[int, Tuple[int, int]]:
    frames = {}
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return {}

    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    serial = int(parts[0].strip())
                    start_time = int(parts[1].strip())
                    end_time = int(parts[2].strip())
                    frames[serial] = (start_time, end_time)
                except ValueError:
                    continue
    return frames

def get_files_as_dict(folder_path: Union[str, Path], marker: str) -> Dict[int, Path]:
    target_dir = Path(folder_path)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory '{target_dir}' does not exist.")
    
    files_found = list(target_dir.glob(f"*{marker}*"))
    file_map = {}
    pattern = re.compile(rf"{re.escape(marker)}[_\-]?(\d+)", re.IGNORECASE)

    for file_path in files_found:
        match = pattern.search(file_path.stem)
        if match:
            file_num = int(match.group(1))
            file_map[file_num] = file_path
    
    return dict(sorted(file_map.items()))

def process_srs_to_df(json_path, start_time, end_time):
    try:
        df = pd.read_json(json_path, convert_dates=False)
        
        if df.empty: return None

        TIME_STEP = 0.01
        t_min = np.floor(df['timestamp'].min())
        t_max = np.floor(df['timestamp'].max()) + 1.0
        
        full_timestamps = np.round(np.arange(t_min, t_max, TIME_STEP), 2)
        rx_ants = [0, 1]
        full_index = pd.MultiIndex.from_product([full_timestamps, rx_ants], names=['timestamp', 'rx_ant'])
        
        df = df.drop_duplicates(subset=['timestamp', 'rx_ant'])
        df = df.set_index(['timestamp', 'rx_ant']).reindex(full_index)
        
        # Determine valid shape from first non-NA entry
        valid_entry = df['srs_ch'].dropna().iloc[0]
        
        # Define shape corrector
        def safe_parse_and_transpose(x):
            if isinstance(x, list):
                arr = np.array(x, dtype=np.float32)
                # FIX: If shape is (1536, 2), Transpose to (2, 1536)
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]: 
                     # Heuristic: If rows > cols (e.g. 1536 > 2), assume it needs transpose
                    return arr.T
                return arr
            # Return zero array of shape (2, 1536) if missing
            return np.zeros((2, 1536), dtype=np.float32)

        # Apply parsing and Transposing
        df['srs_ch'] = df['srs_ch'].apply(safe_parse_and_transpose)
        
        df = df.reset_index()
        df = df.sort_values(by=['timestamp', 'rx_ant'])
        
        pivoted = df.pivot(index='timestamp', columns='rx_ant', values='srs_ch')
        
        # Now r[0] is (2, 1536) and r[1] is (2, 1536). vstack makes (4, 1536).
        stacked_srs = [np.vstack([r[0], r[1]]).tolist() for r in pivoted.itertuples(index=False)]
        
        result_df = pd.DataFrame({
            'timestamp': pivoted.index,
            'srs_ch': stacked_srs
        })

        ts_int = result_df['timestamp'].astype(int)
        result_df = result_df[(ts_int >= start_time) & (ts_int <= end_time)]

        if 'timestamp' in result_df.columns:
            result_df = result_df.drop(columns=['timestamp'])

        return result_df

    except Exception as e:
        print(f"   [Error] SRS Processing failed: {e}")
        return None

def process_e2_to_df(csv_path, start_time, end_time):
    try:
        df = pd.read_csv(csv_path)
        
        if 'timestamp' in df.columns:
            ts_int = df['timestamp'].astype(float).astype(int)
            df = df[(ts_int >= start_time) & (ts_int <= end_time)]
        
        cols_to_del = ["_id", "timestamp", "ue_id", "cellid", "rnti", "dlMcs", "dlBler", "ri", "pmi", "dlQm", "ulQm"]
        existing = [c for c in cols_to_del if c in df.columns]
        df = df.drop(columns=existing)
        
        return df
        
    except Exception as e:
        print(f"   [Error] E2 Processing failed: {e}")
        return None

def normalize_file(file_path):
    if not os.path.exists(file_path): return
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            means = df[numeric_cols].mean()
            stds = df[numeric_cols].std()
            stds[stds == 0] = 1
            df[numeric_cols] = (df[numeric_cols] - means) / stds
            
        df.to_csv(file_path, index=False)
        print(f"   [Normalize] Completed for {os.path.basename(file_path)}")
    except Exception as e:
        print(f"   [Error] Normalization failed: {e}")

def save_list_to_csv(df_list, output_path):
    if not df_list: return
    try:
        combined = pd.concat(df_list, ignore_index=True)
        combined.to_csv(output_path, index=False)
    except Exception as e:
        print(f"   [Error] Saving CSV failed: {e}")

def save_list_to_npy(df_list, output_path):
    """
    Converts a list of DataFrames directly to a .npy file.
    Preserves 3D shape (N, 4, 1536) for SRS data.
    """
    if not df_list: return
    try:
        npy_path = Path(output_path).with_suffix('.npy')
        
        if 'srs_ch' in df_list[0].columns:
            combined_list = []
            for df in df_list:
                combined_list.extend(df['srs_ch'].values)
            
            # Explicitly create arrays. Since process_srs_to_df now guarantees (4, 1536),
            # this stack will produce (N, 4, 1536).
            arrays_3d = [np.array(x, dtype=np.float32) for x in combined_list]
            final_array = np.stack(arrays_3d)
            
        else:
            combined_df = pd.concat(df_list, ignore_index=True)
            final_array = combined_df.to_numpy(dtype=np.float32)

        np.save(npy_path, final_array)
        print(f"   [NPY] Saved {os.path.basename(npy_path)} Shape: {final_array.shape}")
        
    except Exception as e:
        print(f"   [Error] NPY creation failed: {e}")

def main(srs_files_map, e2_files_map, time_frames_map, output_folder):
    
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Memory-Optimized Pipeline ---")
    print(f"Found {len(time_frames_map)} timeframes to process.")

    srs_data_frames = []
    e2_data_frames = []

    for serial_id, (start_time, end_time) in sorted(time_frames_map.items()):
        
        if serial_id not in srs_files_map or serial_id not in e2_files_map:
            print(f"Skipping ID {serial_id}: Missing E2 or SRS file.")
            continue

        print(f"\nProcessing ID {serial_id}...")
        
        raw_srs = srs_files_map[serial_id]
        raw_e2 = e2_files_map[serial_id]
        
        srs_df = process_srs_to_df(raw_srs, start_time, end_time)
        e2_df = process_e2_to_df(raw_e2, start_time, end_time)

        if srs_df is not None and e2_df is not None:
            srs_count = len(srs_df)
            e2_count = len(e2_df)
            
            print(f"   [Processed] E2 Rows: {e2_count} | SRS Rows: {srs_count}")

            expected_srs = e2_count * 5
            if srs_count == expected_srs:
                print(f"   [MATCH] E2({e2_count}) * 5 == SRS({srs_count})")
            else:
                print(f"   [MISMATCH] E2({e2_count}) * 5 != SRS({srs_count})")

            srs_data_frames.append(srs_df)
            e2_data_frames.append(e2_df)
        else:
            print("   [Error] One of the dataframes was empty, skipping ID.")

    total_files = len(srs_data_frames)
    
    if total_files > 1:
        split_index = total_files - 1
    else:
        split_index = total_files

    print(f"\n--- Stacking & Saving Datasets ---")
    print(f"Total IDs: {total_files} | Train: {split_index} | Test: {total_files - split_index}")

    train_e2 = output_dir / "E2_train.csv"
    test_e2 = output_dir / "E2_test.csv"
    train_srs = output_dir / "SRS_train.csv"
    test_srs = output_dir / "SRS_test.csv"

    save_list_to_csv(e2_data_frames[:split_index], train_e2)
    save_list_to_csv(e2_data_frames[split_index:], test_e2)
    save_list_to_csv(srs_data_frames[:split_index], train_srs)
    save_list_to_csv(srs_data_frames[split_index:], test_srs)

    print(f"\n--- Normalizing E2 Datasets ---")
    normalize_file(train_e2)
    normalize_file(test_e2)
    
    print(f"\n--- Converting to NPY ---")
    
    def e2_csv_to_npy(csv_path):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            npy_path = csv_path.with_suffix('.npy')
            arr = df.to_numpy(dtype=np.float32)
            np.save(npy_path, arr)
            print(f"   [NPY] Saved {os.path.basename(npy_path)} Shape: {arr.shape}")

    e2_csv_to_npy(train_e2)
    e2_csv_to_npy(test_e2)
    
    save_list_to_npy(srs_data_frames[:split_index], train_srs)
    save_list_to_npy(srs_data_frames[split_index:], test_srs)

    print(f"\n" + "="*40)
    print(f"FINAL REPORT: Row Counts")
    print(f"="*40)
    for label, path in [("Train E2", train_e2), ("Train SRS", train_srs), 
                        ("Test E2", test_e2), ("Test SRS", test_srs)]:
        if path.exists():
            count = sum(1 for _ in open(path, 'r', encoding='utf-8')) - 1
            print(f"{label:<12}: {max(0, count)} rows")
        else:
            print(f"{label:<12}: File Not Found")
    print(f"="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SRS and E2 Datasets")
    
    parser.add_argument(
        "--base_folder", 
        type=str, 
        default="./Datasets", 
        help="Base folder containing SRS files, E2 files, and Timeframes.txt"
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./preprocessed_datasets",
        help="Path for output files."
    )

    args = parser.parse_args()

    base_folder = Path(args.base_folder)
    timeframes_file = base_folder / "Timeframes.txt"

    print(f"Base Folder:      {base_folder}")
    print(f"Output Folder:    {args.output_folder}\n")

    if not base_folder.exists():
        print(f"Error: The folder '{base_folder}' does not exist.")
    elif not timeframes_file.exists():
        print(f"Error: 'Timeframes.txt' not found in '{base_folder}'.")
    else:
        try:
            srs_files = get_files_as_dict(base_folder, "SRS")
            e2_files = get_files_as_dict(base_folder, "E2")
            time_frames = get_time_frames(timeframes_file)
            
            main(srs_files, e2_files, time_frames, args.output_folder)
            
        except Exception as e:
            print(f"Error during execution: {e}")