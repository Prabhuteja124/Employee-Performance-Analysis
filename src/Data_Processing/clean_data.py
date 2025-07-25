import numpy as np
import pandas as pd
import os
from src.utils.logger_config import GetLogger

logger=GetLogger(__file__).get_logger()
class DataCleanner():
    def __init__(self,df:pd.DataFrame,drop_columns:list=None):
        self.df=df.copy()
        self.drop_columns=drop_columns if drop_columns else []

        logger.info('Data Cleanner initialized...')

        self.drop_irrelevent_columns()
        self.drop_duplicates()
        self.mapping_missing_values()

    def drop_irrelevent_columns(self):
        if not isinstance(self.drop_columns,list):
            raise ValueError('Please provide columns in a list.')
        initial_columns=self.df.columns.tolist()
        self.df.drop(columns=self.drop_columns,inplace=True,errors='ignore')
        dropped_columns=set(initial_columns) - set(self.df.columns)
        logger.info(f'Dropped irrelevent columns : {dropped_columns if dropped_columns else 'None'}')

    def drop_duplicates(self):
        before_drop=self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        after_drop=self.df.shape[0]
        logger.info(f'Removed duplicates : {before_drop - after_drop} rows dropped.')
        
    def mapping_missing_values(self):
        self.df.replace(to_replace=[None],value=np.nan,inplace=True)
        total_missing_values=self.df.isnull().sum()
        logger.info(f'Missing values mapped to Nan.Total misiing entries : {total_missing_values}')

    def get_cleaned_data(self):
        logger.info('Returning Cleaned DataFrame...')
        return self.df
    
