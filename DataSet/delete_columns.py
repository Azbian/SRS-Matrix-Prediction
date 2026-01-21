import pandas as pd

def delete_columns(input_file, columns_to_delete):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Print the input shape
    print(f"Input shape: {df.shape}")

    # Drop the specified columns
    df = df.drop(columns=columns_to_delete, errors='ignore')

    # Print the output shape
    print(f"Output shape: {df.shape}")

    # Save the updated data back to the original file
    df.to_csv(input_file, index=False)

if __name__ == "__main__":
    # Specify the input file and columns to delete
    input_file = "DataSet/Preprocessed Dataset/E2_train.csv"
    columns_to_delete = ["dlQm", "ulQm", "ri", 'dlMcs', 'dlBler']

    delete_columns(input_file, columns_to_delete)