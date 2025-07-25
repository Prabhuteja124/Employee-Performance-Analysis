import numpy as np
import pandas as pd
import joblib
import os
from src.utils.logger_config import GetLogger

logger=GetLogger(__file__).get_logger()

class InferencePipeline():
    def __init__(self,model_path:str,processor_path:str):
        self.model_path=os.path.join(os.path.dirname(os.getcwd()),model_path)
        self.processor_path=os.path.join(os.path.dirname(os.getcwd()),processor_path)
        self.model=None
        self.preprocessor=None
        if not os.path.exists(self.model_path) or not os.path.exists(self.processor_path):
            logger.error(f'Model and preprocessor file paths are not valid')

    def load_artifacts(self):
        logger.info(f'Loading Model and Preprocessor ...')
        try:
            self.model=joblib.load(self.model_path)
            self.preprocessor=joblib.load(self.processor_path)
            logger.info(f'Model and Preprocessor are loaded successfully.')
        except Exception as e:
            logger.error(f"Failed to load model/preprocessor: {e}")
            raise
    
    def predict(self,input_data:pd.DataFrame | dict)->pd.Series:
        logger.info(f'Started inference...')
        try:
            if isinstance(input_data,dict):
                input_data=pd.DataFrame([input_data])
            input_data=input_data.drop(columns=['PerformanceRating'],errors='ignore')
            preprocessed_data =self.preprocessor.transform(input_data)
            predictions=self.model.predict(preprocessed_data)
            reverse_mapping = {0: 2, 1: 3, 2: 4}
            mapped_preds=pd.Series(predictions).map(reverse_mapping)
            logger.info(f'Inference completed succesfully.')
            return mapped_preds
        except Exception as e:
            logger.error(f'Prediction failed: {str(e)}')
            raise 
