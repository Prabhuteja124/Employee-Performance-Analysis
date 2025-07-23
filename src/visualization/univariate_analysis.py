import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

def numeric_univariate_analysis(df: pd.DataFrame,threshold:int=15)-> None:
  
    sns.set_theme(style="whitegrid")
    numerical_columns=df.select_dtypes(include='number').columns.tolist()

    hist_columns=[col for col in numerical_columns if df[col].nunique()>= threshold]
    discrete_columns=[col for col in numerical_columns if col not in hist_columns]

    n_cols = 3
    n_rows = math.ceil(len(numerical_columns) / n_cols)

    plt.figure(figsize=(6*n_cols,5*n_rows))
    plt.suptitle('Numerical Features Distribution',fontsize=24,fontweight='bold',color='Black',y=1.02, style='italic')

    for idx,col in enumerate(numerical_columns,start=1):
        ax = plt.subplot(n_rows,n_cols,idx)
        
        if col in hist_columns:
            sns.histplot(data=df,x=col,bins=50,kde=True,color='skyblue',ax=ax)
            ax.set_ylabel('Count',fontsize=12)
        else:
            sns.countplot(data=df,x=col,palette='Set2',stat='percent',ax=ax)
            ax.set_ylabel('Percentage',fontsize=12)

        ax.set_title(f'{col} Distribution',fontsize=14,fontweight='bold')
        ax.set_xlabel(col,fontsize=12)
        ax.tick_params(axis='x',rotation=45)

    plt.tight_layout(pad=2.0,h_pad=3.0,w_pad=3.0)
    plt.show()


def categorical_univariate_analysis(df: pd.DataFrame)-> None:

    sns.set_theme(style="whitegrid")
    discrete_columns=df.select_dtypes(include='O').columns.tolist()
    n_cols = 2
    n_rows = math.ceil(len(discrete_columns) / n_cols)

    plt.figure(figsize=(6*n_cols,5*n_rows))
    plt.suptitle('Numerical Features Distribution',fontsize=24,fontweight='bold',color='Black',y=1.02, style='italic')

    for idx,col in enumerate(discrete_columns,start=1):
        ax = plt.subplot(n_rows,n_cols,idx)
        
        if col in discrete_columns:
            sns.countplot(data=df,x=col,palette='Set2',stat='percent',ax=ax)
            ax.set_ylabel('Percentage',fontsize=12)

        ax.set_title(f'{col} Distribution',fontsize=14,fontweight='bold')
        ax.set_xlabel(col,fontsize=12)
        ax.tick_params(axis='x',rotation=45)

    plt.tight_layout(pad=2.0,h_pad=3.0,w_pad=3.0)
    plt.show()