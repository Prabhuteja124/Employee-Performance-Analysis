import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger_config import GetLogger

logger=GetLogger(__file__).get_logger()

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


def train_test_split_(df:pd.DataFrame,target_column:str=None)->pd.DataFrame:
    logger.info(f'Train-Test Split Initialized ...')
    if not target_column or not isinstance(target_column,str):
        logger.error('Target column is not valid.Provide a valid target column.')
        raise ValueError('Invalid target_column provided.')
    df=df.copy()

    X=df.drop(columns=target_column,axis=1)
    y=df[target_column]
    mapping={2:0,3:1,4:2}
    y=y.map(mapping)

    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)
    logger.info(f'Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}')
    logger.info(f'Target distribution (train):\n{y_train.value_counts().to_dict()}')
    logger.info(f'Target distribution (test):\n{y_test.value_counts().to_dict()}')

    return X_train,X_test,y_train,y_test

