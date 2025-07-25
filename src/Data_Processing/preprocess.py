import numpy as np
import pandas as pd
import os
import joblib
from src.Data_Processing.clean_data import DataCleanner
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger_config import GetLogger
from src.utils.utils import train_test_split_

logger=GetLogger(__file__).get_logger()

def ordinal_encoder(x):
    logger.info('Applying ordinal encoding.')
    mappings ={
        'Gender':['Female','Male'],
        'MaritalStatus':['Single','Married','Divorced'],
        'BusinessTravelFrequency':['Non-Travel','Travel_Frequently','Travel_Rarely'],
        'OverTime':['No','Yes'],
        'Attrition':['No','Yes']
    }
    df=x.copy()
    for col in mappings:
        encoder=OrdinalEncoder(categories=[mappings[col]])
        df[col]=encoder.fit_transform(df[[col]])
        logger.info(f"Encoded column: {col}")
    return df

def manual_impute(x):
    logger.info('Applying Manual Imputation..')
    df=x.copy()
    for col in df.columns:
        if df[col].nunique()<10:
            df[col]=df[col].fillna(df[col].mode()[0])
            logger.info(f"Filled missing values in '{col}' using mode.")
        else:
            df[col]=df[col].fillna(df[col].median())
            logger.info(f"Filled missing values in '{col}' using median.")
    return df

def log_transform(x):
    logger.info('Applying log transformation')
    return np.log1p(np.clip(x,a_min=0,a_max=None))

class Preprocessor():
    def __init__(self,df:pd.DataFrame):
        self.df=df.copy()
        logger.info('Initialized Preprocessor with input Dataframe.')

        self.Ordinal_encoding_columns=['Gender','MaritalStatus','BusinessTravelFrequency','OverTime','Attrition']
        self.One_hot_encoding_columns=['EducationBackground','EmpDepartment','EmpJobRole']
        self.transform_columns=['Age','DistanceFromHome','EmpHourlyRate','EmpLastSalaryHikePercent',
                                'ExperienceYearsInCurrentRole','TotalWorkExperienceInYears','YearsSinceLastPromotion',
                                'YearsWithCurrManager','EmpEducationLevel','EmpEnvironmentSatisfaction','EmpJobInvolvement',
                                'EmpJobSatisfaction','NumCompaniesWorked','EmpRelationshipSatisfaction','TrainingTimesLastYear',
                                'EmpWorkLifeBalance']
        # self.imputation_columns=['YearsWithCurrManager','ExperienceYearsInCurrentRole']
        self.drop_columns_after_coreleationanalysis=['EmpJobLevel','ExperienceYearsAtThisCompany']
    
    def preprocess_run(self):
        logger.info('Starting Preprocessing Pipeline...')
        df=self.df.copy()
        df.drop(columns=self.drop_columns_after_coreleationanalysis,inplace=True,errors='ignore')
        logger.info(f'Dropped Columns : {self.drop_columns_after_coreleationanalysis}')

        X_train,X_test,y_train,y_test=train_test_split_(df,'PerformanceRating')

        self.pipeline =ColumnTransformer(
            transformers=[
                ('OneHotEncoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),self.One_hot_encoding_columns),
                ('OrdinalEncoding',FunctionTransformer(ordinal_encoder,validate=False),self.Ordinal_encoding_columns),
                ('Log_scaling',Pipeline(
                    steps=[
                        ('log',FunctionTransformer(log_transform,validate=False)),
                        ('scaling',StandardScaler())
                    ]
                ),self.transform_columns)
                # ('impute',FunctionTransformer(manual_impute,validate=False),self.imputation_columns)
            ],
            remainder='passthrough',
            verbose_feature_names_out=True
        )
        logger.info('ColumnTransformer pipeline defined...')
        X_train_processed=self.pipeline.fit_transform(X_train)

        X_test_processed=self.pipeline.transform(X_test)

        logger.info('Fitted pipeline on X_train and transformed both train and test data.')

        one_hot_cols=self.pipeline.named_transformers_['OneHotEncoder'].get_feature_names_out(self.One_hot_encoding_columns)
        ordinal_cols=self.Ordinal_encoding_columns
        log_scaled_cols=self.transform_columns

        passthrough_cols=[col for col in df.columns.drop('PerformanceRating') 
                            if col not in (self.One_hot_encoding_columns + 
                                        self.Ordinal_encoding_columns + 
                                        self.transform_columns + 
                                        self.drop_columns_after_coreleationanalysis)]

        all_columns=list(one_hot_cols)+ordinal_cols+log_scaled_cols+passthrough_cols

        logger.info(f'Getting feature names from ColumnTransformer pipeline...')

        X_train_processed = pd.DataFrame(X_train_processed,columns=all_columns)
        X_test_processed = pd.DataFrame(X_test_processed,columns=all_columns)

        model_dir = os.path.join(os.path.dirname(os.getcwd()),'models')
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.pipeline,os.path.join(model_dir,'Preprocessor.pkl'))
        logger.info(f"Saved the preprocessing pipeline to 'Preprocessor.pkl':{model_dir}.")

        return X_train_processed,X_test_processed,y_train,y_test

    