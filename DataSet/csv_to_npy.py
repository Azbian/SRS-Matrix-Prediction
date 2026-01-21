import numpy as np
import pandas as pd
import ast
import os
import argparse

def convert_csv_to_numpy(convert_e2=False, convert_srs=False):
    """Parses raw CSVs into efficient NumPy binaries."""
    # Predefined paths
    e2_path = "P:/SP Challenge/DataSet/Preprocessed Dataset/E2_test.csv"
    srs_path = "P:/SP Challenge/DataSet/Preprocessed Dataset/srs_test.csv"
    output_prefix = "P:/SP Challenge/Model/"

    if convert_e2:
        print(f"Converting {e2_path} to binary format...")
        # Process E2
        df_e2 = pd.read_csv(e2_path, header=None).apply(pd.to_numeric, errors='coerce').fillna(0)
        e2_data = df_e2.iloc[:, :19].values.astype(np.float32)
        # Save to disk
        np.save(f"{output_prefix}E2_test.npy", e2_data)
        print(f"E2 data saved to {output_prefix}E2_test.npy")

    if convert_srs:
        print(f"Converting {srs_path} to binary format...")
        # Process SRS (The slow part)
        df_srs = pd.read_csv(srs_path, header=None)
        parsed_srs = []
        for val in df_srs[0]:
            # Parse string once and store as float32
            raw_arr = np.array(ast.literal_eval(val), dtype=np.float32)
            parsed_srs.append(raw_arr.T.reshape(4, 1536))
        srs_data = np.stack(parsed_srs, axis=0)
        # Save to disk
        np.save(f"{output_prefix}SRS_test.npy", srs_data)
        print(f"SRS data saved to {output_prefix}SRS_test.npy")

    print("Pre-processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert predefined CSV files to NumPy binaries.")
    parser.add_argument("--convert", choices=["e2", "srs", "both"], required=True, help="Specify which files to convert: 'e2', 'srs', or 'both'.")

    args = parser.parse_args()

    if args.convert == "e2":
        convert_csv_to_numpy(convert_e2=True)
    elif args.convert == "srs":
        convert_csv_to_numpy(convert_srs=True)
    elif args.convert == "both":
        convert_csv_to_numpy(convert_e2=True, convert_srs=True)