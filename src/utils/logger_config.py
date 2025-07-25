import logging
import os
from datetime import datetime



class GetLogger():
    def __init__(self,logger_name:str,LOG_DIR:str='logs',file_path:str='Project_Pipeline'):
        self.logger_name=os.path.splitext(os.path.basename(logger_name))[0]
        current_file_path=os.path.abspath(__file__)
        src_folder_path=os.path.dirname(os.path.dirname(current_file_path))
        project_root=os.path.abspath(os.path.join(src_folder_path,".."))  
        self.LOG_DIR=os.path.join(project_root,LOG_DIR)
        self.file_path=file_path
        self.timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path=os.path.join(self.LOG_DIR,f'{self.file_path}_{self.timestamp}.log')
        self.create_log_directory()

    def create_log_directory(self):
        try:
            os.makedirs(self.LOG_DIR,exist_ok=True)
        except Exception as Error:
            raise OSError(f"Failed to create log directory '{self.LOG_DIR}':{Error}")

    def get_logger(self):
        logger=logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            file_handler=logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)
            console_handler=logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger