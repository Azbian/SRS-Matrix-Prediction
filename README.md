# SP Challenge

## Project Overview
The SP Challenge project is designed to process and analyze datasets for channel prediction using deep learning models. The project includes data preprocessing, model training, and evaluation.

## Project Structure

```
DataSet/
    add_csv_files.py
    count_csv_timestamps.py
    csv_normalization.py
    csv_to_npy.py
    Data_preprocessing.py
    delete_columns.py
    Timeframes.txt
    Preprocessed Dataset/
        E2_test_raw.csv
        E2_train_raw.csv
        E2_test.csv
        E2_train.csv
        srs_test.csv
        srs_train.csv
Model/
    Channel_prediction_model.ipynb
    E2_test.npy
    E2_train.npy
    SRS_test.npy
    SRS_train.npy
    logs/
        performance_log.txt
README.md
requirements.txt
```

### Key Directories and Files

- **DataSet/**: Contains scripts for data preprocessing and the preprocessed datasets.
  - `Data_preprocessing.py`: Main script for data preprocessing.
  - `add_csv_files.py`: Script to combine multiple CSV files row-wise.
  - `csv_to_npy.py`: Converts CSV files to NPY format.
  - `count_csv_timestamps.py`: Script to count timestamps in CSV files.
  - `csv_normalization.py`: Script for normalizing CSV data.
  - `delete_columns.py`: Script to delete specific columns manually from datasets.
  - `Timeframes.txt`: Contains information about timeframes for data processing.
  - **Preprocessed Dataset/**: Contains preprocessed datasets such as `E2_test.csv`, `E2_train.csv`, etc.

- **Model/**: Contains the deep learning model and related files.
  - `Channel_prediction_model.ipynb`: Jupyter notebook for model training and evaluation.
  - `E2_test.npy`, `E2_train.npy`, `SRS_test.npy`, `SRS_train.npy`: Preprocessed NPY files for training and testing.
  - **logs/**: Contains performance for different model parameters.

- **README.md**: Project documentation.

- **requirements.txt**: List of required Python libraries for the project.

## Requirements

The project requires the following Python libraries:

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

Install the dependencies using the following command:

```
pip install -r requirements.txt
```

## How to Run

### Preprocessing the Data

1. **Run the main preprocessing script**:
   ```
   python DataSet/Data_preprocessing.py --srs_input_json <path_to_srs_input_json> --srs_output_csv <path_to_srs_output_csv> --e2_input_csv <path_to_e2_input_csv>
   ```
   - Replace `<path_to_srs_input_json>` with the path to the input SRS JSON file.
   - Replace `<path_to_srs_output_csv>` with the path to save the preprocessed SRS CSV file.
   - Replace `<path_to_e2_input_csv>` with the path to the input E2 CSV file.

2. **Combine multiple CSV files**:
   ```
   python DataSet/add_csv_files.py --base_dir <path_to_base_directory>
   ```
   - Replace `<path_to_base_directory>` with the directory containing the CSV files to combine.

3. **Normalize the E2 CSV data**:
   ```
   python DataSet/csv_normalization.py --input_file <path_to_input_csv> --output_file <path_to_output_csv>
   ```
   - Replace `<path_to_input_csv>` with the path to the combined E2 CSV file.
   - Replace `<path_to_output_csv>` with the path to save the normalized E2 CSV file.

4. **Convert CSV files to NPY format**:
   ```
   python DataSet/csv_to_npy.py --e2_path <path_to_e2_csv> --srs_path <path_to_srs_csv> --output_prefix <path_to_output_directory> --convert <e2|srs|both>
   ```
   - Replace `<path_to_e2_csv>` with the path to the normalized E2 CSV file.
   - Replace `<path_to_srs_csv>` with the path to the combined SRS CSV file.
   - Replace `<path_to_output_directory>` with the directory to save the NPY files.
   - Use `--convert` to specify whether to convert `e2`, `srs`, or `both`.


### Full Preprocessing Workflow

1. Use the `Data_preprocessing.py` script to preprocess the raw data:
   - Convert `E2` and `SRS.json` raw data into preprocessed CSV files (`E2_train.csv`, `pp_srs.csv`).

2. Use the `add_csv_files.py` script to combine the preprocessed E2 and SRS CSV files into single combined files (`combined_E2.csv` and `combined_pp_srs.csv`).

3. Use the `csv_normalization.py` script to normalize the combined E2 CSV file only. This step ensures that the E2 data is scaled appropriately for model training.

4. Use the `csv_to_npy.py` script to convert the normalized E2 CSV file and the combined SRS CSV file into NPY format. These NPY files will be used as input for the deep learning model.

### Train the Model

1. Open the `Channel_prediction_model.ipynb` notebook in the `Model/` directory.
2. Follow the steps in the notebook to train and evaluate the model.

## Model Overview

The model is a deep learning architecture that combines convolutional layers and LSTM layers to predict channel data. It uses two input branches:

1. **Radio Input Branch**: Processes E2 data using convolutional layers.
2. **SRS Input Branch**: Processes SRS data using convolutional layers and residual connections.

The outputs of both branches are concatenated and passed through LSTM and dense layers to produce the final predictions.

## Results

The model is trained for 100 epochs, and the training loss is visualized using Matplotlib. The evaluation is performed using the test dataset, and metrics such as Mean Squared Error (MSE) are reported.

## Contact

For any questions or feedback, please contact the project maintainer.