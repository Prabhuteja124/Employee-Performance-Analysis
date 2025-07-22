import os
import pandas as pd
import numpy as np

def basic_checks(df:pd.DataFrame)-> str:
    print(df.head().to_string)
    print(f"\nDataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe().to_string)
    print("\nCategorical Columns:")
    print(df.columns[df.dtypes == 'object'].tolist())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")