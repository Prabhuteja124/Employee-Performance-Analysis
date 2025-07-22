import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.model_selection import train_test_split
import os
from src.utils.logger_config import GetLogger

logger=GetLogger(__file__).get_logger()

class DataLoader():
    def __init__(self,folder_path='data/raw'):
        self.folder_path=os.path.join(os.getcwd(),folder_path)

    def load_dataset(self,filename:str)-> pd.DataFrame:
        file_path=os.path.join(self.folder_path,filename)
        try:
            logger.info(f'Loading Data from file...')
            if filename.endswith('.xls'):
                df=pd.read_excel(file_path)
                logger.info('Excel Dataset Loaded Successfully!')
            elif filename.endswith('.csv'):
                df=pd.read_csv(file_path)
                logger.info('CSV Dataset Loaded Successfully!')
            else:
                logger.debug('Error Occured while Loading Data..!')
                raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")

            logger.info(f" Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
            
            return df

        except FileNotFoundError as e:
            logger.error(f"{e}")
        except Exception as e:
            logger.error(f" Unexpected error occurred: {str(e)}")

        return None