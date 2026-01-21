import numpy as np
import pandas as pd
import ast
import os
def convert_csv_to_numpy(e2_path, srs_path, output_prefix):
    """Parses raw CSVs into efficient NumPy binaries."""
    print(f"Converting {e2_path} and {srs_path} to binary format...")
    
    # Process E2
    df_e2 = pd.read_csv(e2_path, header=None).apply(pd.to_numeric, errors='coerce').fillna(0)
    e2_data = df_e2.iloc[:, :19].values.astype(np.float32)
    
    # Process SRS (The slow part)
    df_srs = pd.read_csv(srs_path, header=None)
    parsed_srs = []
    for val in df_srs[0]:
        # Parse string once and store as float32
        raw_arr = np.array(ast.literal_eval(val), dtype=np.float32)
        parsed_srs.append(raw_arr.T.reshape(4, 1536))
    srs_data = np.stack(parsed_srs, axis=0)
    
    # Save to disk
    np.save(f"{output_prefix}E2_test.npy", e2_data)
    np.save(f"{output_prefix}SRS_test.npy", srs_data)
    print("Pre-processing complete.")

e2_path = "P:/SP Challenge/DataSet/Preprocessed Dataset/E2_test.csv"
srs_path = "P:/SP Challenge/DataSet/Preprocessed Dataset/srs_test.csv"
convert_csv_to_numpy(e2_path, srs_path, "P:/SP Challenge/Model/")