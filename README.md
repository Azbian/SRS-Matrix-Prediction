
# SRS Matrix Prediction Project

## Project Overview
This project processes and analyzes wireless channel datasets for channel prediction using deep learning. It takes SRS matrices and radio state (E2) data as input and forecasts the future SRS matrix 50 ms ahead.

## Project Structure

```
DataSet/
    Data_preprocessing.py
    E2_1.csv
    E2_2.csv
    ...
    E2_8.csv
    SRS_1.json
    SRS_2.json
    ...
    SRS_8.json
    Timeframes.txt
    preprocessed_datasets/
        E2_test.csv
        E2_train.csv
        SRS_test.csv
        SRS_train.csv
Model/
    Channel_prediction_model.ipynb
    E2_test.npy
    E2_train.npy
    SRS_test.npy
    SRS_train.npy
    logs/
        performance_log.txt
    Saved/
        model_weights.h5
README.md
requirements.txt
```

### Key Directories and Files

- **DataSet/**: Raw and preprocessed data, and preprocessing scripts.
  - `Data_preprocessing.py`: Main script for preprocessing raw E2 and SRS data.
  - `E2_1.csv` ... `E2_8.csv`: Raw E2 data for each experiment.
  - `SRS_1.json` ... `SRS_8.json`: Raw SRS data for each experiment.
  - `Timeframes.txt`: Timeframe information for data processing.
  - **preprocessed_datasets/**: Contains processed CSVs for training/testing.
    - `E2_train.csv`, `E2_test.csv`, `SRS_train.csv`, `SRS_test.csv`: Preprocessed datasets.

- **Model/**: Model code, training notebooks, and results.
  - `Channel_prediction_model.ipynb`: Jupyter notebook for model training/evaluation.
  - `E2_train.npy`, `E2_test.npy`, `SRS_train.npy`, `SRS_test.npy`: Numpy arrays for model input.
  - **logs/**: Model training logs (e.g., `performance_log.txt`).
  - **Saved/**: Saved model weights (e.g., `model_weights.h5`).

- **Papers/**: Research papers and references.

- **README.md**: Project documentation (this file).

- **requirements.txt**: Python dependencies for the project.

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

Main libraries used:

- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn
- glob
- os
- csv
- argparse
- tensorflow-addons
- json
- datetime
- io

Other standard libraries: glob, os, csv, argparse, json, datetime, io

## How to Run

### Data Preprocessing Script

All data preprocessing is handled by a single script: `Data_preprocessing.py`.

1. **Preprocess all raw data:**
  ```
  python DataSet/Data_preprocessing.py --base_folder <path_to_data_folder> --output_folder <path_to_output_folder>
  ```
  - `<path_to_data_folder>`: Path to the folder containing raw E2 CSVs, SRS JSONs, and `Timeframes.txt` (e.g., `DataSet/`)
  - `<path_to_output_folder>`: Path to save the preprocessed CSV and NPY files (e.g., `DataSet/preprocessed_datasets/`)

2. **Preprocessed output:**
  - Preprocessed CSVs (`E2_train.csv`, `E2_test.csv`, `SRS_train.csv`, `SRS_test.csv`) and NPY files are saved in the specified output folder.
  - These files are used directly for model training and evaluation.

### Model Training

1. Open `Model/Channel_prediction_model.ipynb` in Jupyter or VS Code.
2. Follow the notebook steps to train and evaluate the model using the NPY files.
3. Model weights and logs are saved in `Model/Saved/` and `Model/logs/`.

## Model Overview

The model is a deep learning architecture combining convolutional and LSTM layers for channel prediction. It uses two input branches:

- **Radio Input Branch:** Processes E2 data (radio state) with convolutional layers.
- **SRS Input Branch:** Processes SRS data (channel state) with convolutional and residual layers.

The outputs are concatenated and passed through LSTM and dense layers to predict the future SRS matrix.

### Input/Output

- **Input:**
  - E2: Tensor of shape `(5, 4, 14)`
  - SRS: Tensor of shape `(20, 20, 1536)`
- **Output:**
  - Tensor of shape `(4, 1536)` (predicted SRS matrices for two antennas)

## Methodology

### Data Processing

- Experiments 1-7: Used for training
- Experiment 8: Used for testing
- Steps:
  1. Extract raw files (`E2_*.csv`, `SRS_*.json`) into `DataSet/`
  2. Place the Timeframes.txt into the same folder
  3. Run `Data_preprocessing.py` to generate preprocessed CSVs
  4. Move/train/test NPY files to `Model/` as needed
  5. Train the model in the notebook

### Feature Engineering

- **E2 Data:**
  - 3D tensor `(5, 4, 14)` from 20 time steps, 14 features each
  - Normalized the data during preprocessing
- **SRS Data:**
  - 3D tensor `(20, 20, 1536)` from 100 rows, 1536 features each
  - Normalized during training, denormalized at output

## Results

- Output shape: `(4, 1536)` (two SRS matrices for antennas 0 and 1)
- Trained for 100 epochs
- Training loss visualized with Matplotlib
- Evaluation: Mean Squared Error (MSE) on test set

## Contact

For questions or feedback, please contact the project maintainer.