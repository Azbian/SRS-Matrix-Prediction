import pandas as pd
import numpy as np

# ===== CONFIG =====
INPUT_CSV = "DataSet/Preprocessed Dataset/E2_test_raw.csv"          # your raw file
OUTPUT_CSV = "DataSet/Preprocessed Dataset/E2_test.csv"    # output file
EPS = 1e-8                       # avoid divide-by-zero
# ==================

# Load data
df = pd.read_csv(INPUT_CSV)

print(f"Original shape: {df.shape}")

# Drop duplicate rows
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Select numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Create a copy for normalized data
df_norm = df.copy()

# Z-score normalization
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()

    if std < EPS:
        # Constant column → set to 0
        df_norm[col] = 0.0
        print(f"[INFO] Column '{col}' is constant → set to 0")
    else:
        df_norm[col] = (df[col] - mean) / std

# Save normalized data
df_norm.to_csv(OUTPUT_CSV, index=False)

print(f"Normalized data saved to: {OUTPUT_CSV}")
