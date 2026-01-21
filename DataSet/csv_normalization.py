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
    input_file = "DataSet/Preprocessed Dataset/E2_train_raw.csv"
    output_file = "DataSet/Preprocessed Dataset/E2_train.csv"
    normalize_csv(input_file, output_file)