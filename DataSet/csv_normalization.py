import pandas as pd
import argparse

def normalize_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Handle missing values by filling them with the column mean
    df = df.fillna(df.mean())

    # Normalize each column (subtract mean and divide by standard deviation)
    normalized_df = df.copy()
    for column in df.columns:
        if df[column].std() != 0:  # Check if standard deviation is not zero
            normalized_df[column] = (df[column] - df[column].mean()) / df[column].std()

    # Save the normalized data to a new CSV file
    normalized_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize a CSV file by columns.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output normalized CSV file.")

    args = parser.parse_args()

    normalize_csv(args.input_file, args.output_file)