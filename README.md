
INX Future Inc. - Employee Performance Analysis
--------
This project is a comprehensive machine learning pipeline developed to analyze and predict employee performance at INX Future Inc. The solution covers everything from data preprocessing and exploratory analysis to model training, evaluation, and deployment-ready structure.

Objective
---------
To build predictive models that can accurately classify employees into performance categories based on their historical attributes. The goal is to assist management in identifying top performers and employees who might need support.

Project Structure
-----------------
``` bash
INX_Employee_Performance/
├── data/                        
├── logs/                        
├── models/                      
├── notebooks/                  
│   ├── data_analysis.ipynb     
│   └── modeling.ipynb          
├── Project Summary/
│   └── Summary_project.ipynb   
├── src/                        
│   ├── Data_Processing/
│   │   ├── clean_data.py      
│   │   └── preprocess.py       
│   ├── models/
│   │   ├── train_model.py      
│   │   ├── evaluate_model.py   
│   │   └── predict_model.py    
│   ├── utils/
│   │   ├── load_data.py        
│   │   ├── logger_config.py   
│   │   └── utils.py           
└── visualization/             
```

Key Features
------------
- Clean modular architecture
- Handles missing values, outliers, and scaling
- Supports multiple ML algorithms: SVM, RandomForest, XGBoost, LightGBM, etc.
- Evaluation includes confusion matrix, F1-score, precision, recall
- Logging included for reproducibility

Results Summary
---------------
- LightGBM and Gradient Boosting achieved 94% accuracy and 0.94 F1-score
- Ensemble models like XGBoost and RandomForest performed consistently well (92%+)
- Logistic Regression and Ridge Classifier showed decent baseline performance (~75%)
- SVM provided good balance (~80% accuracy)
- Naive Bayes underperformed due to strong assumptions

Installation
------------
Clone the repo and install dependencies:

    git clone https://github.com/your-username/INX_Employee_Performance.git
    cd INX_Employee_Performance
    pip install -r requirements.txt

Usage
-----
Run model training:

    python src/models/train_model.py

Predict on new data:

    python src/models/predict_model.py

Requirements
------------
See requirements.txt

License
-------
This project is for educational purposes only. Feel free to reuse or modify with attribution.
