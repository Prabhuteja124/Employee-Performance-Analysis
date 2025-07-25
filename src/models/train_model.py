import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import (
    accuracy_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger_config import GetLogger

logger=GetLogger(__file__).get_logger()

class ModelTrainer():
    def __init__(self,models_path='models'):
        self.models_dict={
            "LogisticRegression":LogisticRegression(max_iter=300,multi_class='ovr',n_jobs=-1,class_weight='balanced'),
            "RidgeClassifier":RidgeClassifier(class_weight='balanced'), 
            "SVM":SVC(probability=True,class_weight='balanced'),
            "DecisionTree":DecisionTreeClassifier(class_weight='balanced'),
            "RandomForest":RandomForestClassifier(n_jobs=-1,class_weight='balanced'),
            "GradientBoosting":GradientBoostingClassifier(),
            "naive_bayes":GaussianNB(),
            "AdaBoost":AdaBoostClassifier(),
            "XGBoost":XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_jobs=-1),
            "LightGBM":LGBMClassifier(n_jobs=-1,class_weight='balanced')
            }
        
        self.project_root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
        self.model_dir=os.path.join(self.project_root,models_path)

        try:
            os.makedirs(self.model_dir,exist_ok=True)
            logger.info(f"Model directory initialized at {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to create models directory at {self.model_dir}: {str(e)}")


    def train_models(self,X_train,y_train):
        logger.info(f'Training started for all models...')
        self.trained_models={}
        for name,model in self.models_dict.items():
            try:
                logger.info(f'Training {name}')
                model.fit(X_train,y_train)
                self.trained_models[name]=model
            except Exception as e:
                logger.error(f'Model Training Failed {name} : {str(e)}')
        return self.trained_models
    


    def evaluate_models(self,X_train,y_train,X_test,y_test):
        logger.info('Starting Model Evaluation....')
        results=[]
        for name,model in self.trained_models.items():
            try:
                logger.info(f'Evaluating Model : {name}')
                y_pred_train=model.predict(X_train)
                train_acc=accuracy_score(y_train, y_pred_train)
                train_f1=f1_score(y_train, y_pred_train, average='weighted')
                y_pred_test=model.predict(X_test)
                test_acc=accuracy_score(y_test, y_pred_test)
                test_f1=f1_score(y_test, y_pred_test, average='weighted')
                
                logger.info(f"{name} - Train Accuracy: {train_acc:.3f} | Train F1: {train_f1:.3f}")
                logger.info(f"{name} - Test Accuracy: {test_acc:.3f} | Test F1: {test_f1:.3f}")
                
                if hasattr(model,"predict_proba"):
                    try :
                        y_proba=model.predict_proba(X_test)
                        if y_proba.shape[1] > 2:
                            roc_auc=roc_auc_score(y_test,y_proba,multi_class='ovr',average='weighted')
                        else:roc_auc=roc_auc_score(y_test,y_proba[:,1])
                        logger.info(f"{name} - ROC AUC: {roc_auc:.3f}")
                    except Exception as e:
                        logger.warning(f"ROC AUC could not be calculated for {name}: {str(e)}")
                        roc_auc=np.nan
                else:
                    roc_auc=None
                    logger.info(f"{name} does not support predict_proba; ROC AUC skipped.")
                logger.info(f"Confusion Matrix for {name}:\n{confusion_matrix(y_test, y_pred_test)}")
                results.append({
                    'Model':name,
                    'train_accuracy':train_acc,
                    'train_f1':train_f1,
                    'test_accuracy':test_acc,
                    'test_f1':test_f1,
                    'ROC':roc_auc
                })
            except Exception as e:
                logger.error(f"Error during evaluation of {name}: {str(e)}")

        logger.info('Evaluation completed for all models.')
        return results
