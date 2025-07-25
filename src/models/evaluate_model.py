import joblib
from src.models.train_model import ModelTrainer
import os
from src.utils.logger_config import GetLogger

logger=GetLogger(__file__,file_path='save_best_model.log').get_logger()

trainer=ModelTrainer()


def get_best_model(result:list[dict]) -> dict:
    logger.info('Getting the best model based on F1 Score.')
    best_model = max(result,key=lambda x : x['test_f1'])
    logger.info(f'Best Model :{best_model['Model']} with F1-Score : {best_model['test_f1']:.3f}')
    return best_model

def save_best_model(result:list[dict],models:dict):
    logger.info('Saving the best Model...')
    best_model_info=get_best_model(result=result)
    best_model_name=best_model_info['Model']
    best_model=models[best_model_name]
    logger.info(f"Best Model is {best_model} | F1-Score = {best_model_info['test_f1']:.3f}")
    save_path=os.path.join(trainer.model_dir,f'{best_model_name}_best_model.pkl')
    
    try:
        joblib.dump(best_model,save_path)
        logger.info(f'Model Saved Succesfully at : {save_path}')
    except Exception as e:
        logger.info(f'Failed to save model at : {str(e)}')
    return best_model,result
