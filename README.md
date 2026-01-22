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
    Preprocessed Dataset/
        E2_test_raw.csv
        E2_test.csv
        E2_train_raw.csv
        E2_train.csv
        srs_test.csv
        srs_train.csv
        Timeframes.txt
Model/
    Channel_prediction_model.ipynb
    E2_test.npy
    E2_train.npy
    SRS_test.npy
    SRS_train.npy
```

### Key Directories and Files

- **DataSet/**: Contains scripts for data preprocessing and the preprocessed datasets.
  - `add_csv_files.py`: Script to add CSV files.
  - `count_csv_timestamps.py`: Script to count timestamps in CSV files.
  - `csv_normalization.py`: Script for normalizing CSV data.
  - `csv_to_npy.py`: Converts CSV files to NPY format.
  - `Data_preprocessing.py`: Main script for data preprocessing.
  - `delete_columns.py`: Script to delete specific columns from datasets.
  - **Preprocessed Dataset/**: Contains preprocessed datasets such as `E2_test.csv`, `E2_train.csv`, etc.

- **Model/**: Contains the deep learning model and related files.
  - `Channel_prediction_model.ipynb`: Jupyter notebook for model training and evaluation.
  - `E2_test.npy`, `E2_train.npy`, `SRS_test.npy`, `SRS_train.npy`: Preprocessed NPY files for training and testing.


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

1. Clone the repository:
   ```
   git clone <repository-url>
   cd SP Challenge
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Preprocess the data:
   - Use the scripts in the `DataSet/` directory to preprocess the raw data.
   - Run Data_preprocessing.py to process the whole data and then convert to .npy with csv_tpy.py
   - Manual scripts are provided to customize the dataset

4. Train the model:
   - Open the `Channel_prediction_model.ipynb` notebook in the `Model/` directory.
   - Follow the steps in the notebook to train and evaluate the model.

## Model Overview

The model is a deep learning architecture that combines convolutional layers and LSTM layers to predict channel data. It uses two input branches:

1. **Radio Input Branch**: Processes E2 data using convolutional layers.
2. **SRS Input Branch**: Processes SRS data using convolutional layers and residual connections.

The outputs of both branches are concatenated and passed through LSTM and dense layers to produce the final predictions.

## Results

The model is trained for 100 epochs, and the training loss is visualized using Matplotlib. The evaluation is performed using the test dataset, and metrics such as Mean Squared Error (MSE) are reported.

## Contact

For any questions or feedback, please contact the project maintainer.