import pandas as pd
import glob
import os
import argparse

def combine_csv_rowwise(pattern, output_file, has_header=True):
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    print(f"\nCombining {len(files)} files row-wise:")
    for f in files:
        print(" ", os.path.basename(f))

    dfs = []
    for file in files:
        if has_header:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file, header=None)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Save without header if original had no header
    combined_df.to_csv(output_file, index=False, header=has_header)

    print(f"Saved → {output_file}")
    print(f"Total rows: {combined_df.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple CSV files row-wise.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the CSV files.")

    args = parser.parse_args()

    base_dir = args.base_dir

    # ---- E2 (HAS HEADER) ----
    combine_csv_rowwise(
        pattern=os.path.join(base_dir, "E2_*.csv"),
        output_file=os.path.join(base_dir, "combined_E2.csv"),
        has_header=True
    )

    # ---- SRS (NO HEADER) ----
    combine_csv_rowwise(
        pattern=os.path.join(base_dir, "pp_srs_*.csv"),
        output_file=os.path.join(base_dir, "combined_pp_srs.csv"),
        has_header=False
    )
