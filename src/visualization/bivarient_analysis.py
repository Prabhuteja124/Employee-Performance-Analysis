import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

def bivariate_numerical_plots(df:pd.DataFrame)-> None:
    target='PerformanceRating'
    num_cols = [col for col in df.select_dtypes(exclude='O').columns if col != target]
    continuous_cols = [col for col in num_cols if df[col].nunique() >= 15]
    discrete_cols = [col for col in num_cols if 2 < df[col].nunique() < 15]
    total_plots=len(continuous_cols)+len(discrete_cols)
    n_cols=3
    n_rows=math.ceil(len(num_cols)/n_cols)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6*n_cols,5*n_rows))
    plt.suptitle('Bivariate Analysis of Numerical Featuers VS Target',fontsize=24,fontweight='bold',color='Black',y=1.01,style='italic')
    plot_idx=1
    for col in continuous_cols:
        ax = plt.subplot(n_rows,n_cols,plot_idx)
        sns.barplot(data=df,x=target,y=col,ci='sd',estimator=np.mean,alpha=0.8,palette='Set3',ax=ax)
        means = df.groupby(target)[col].mean()
        for i,val in enumerate(means):
            ax.text(i,val + 0.05 * df[col].max(),f'{val:.2f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
        ax.set_title(f'{col} vs {target}',fontsize=12,fontweight='bold')
        ax.set_xlabel(target)
        ax.set_ylabel(f'Mean {col}')
        plot_idx += 1
    for col in discrete_cols:
        ax = plt.subplot(n_rows,n_cols,plot_idx)
        sns.countplot(data=df,x=col,palette='Set1',hue=target,ax=ax)
        ax.set_title(f'{col} vs {target}',fontsize=12,fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.legend(title=target)
        plot_idx += 1
    plt.tight_layout(pad=2.0,h_pad=3.0,w_pad=3.0)
    plt.show()

def bivariate_categorical_plots(df:pd.DataFrame)-> None:
    target='PerformanceRating'
    cat_columns=[col for col in df.select_dtypes(include='O').columns if col != target]
    n_cols=2
    n_rows=math.ceil(len(cat_columns)/n_cols)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6*n_cols,5*n_rows))
    plt.suptitle('Bivariate Analysis of Categorical Featuers vs Target ',fontsize=24,fontweight='bold',color='Black',y=1.01,style='italic')
    plot_idx=1
    for column in cat_columns:
        ax=plt.subplot(n_rows,n_cols,plot_idx)
        sns.countplot(data=df,x=column,hue=target,palette='Set2',ax=ax)
        ax.set_title(f'{column} vs {target} ',fontsize=12,fontweight='bold')
        ax.set_xlabel(column,fontsize=12)
        ax.set_ylabel('Count')
        ax.legend(title=target)
        plt.xticks(rotation=30,ha='right')
        plot_idx+=1
    plt.tight_layout(pad=2.0,h_pad=3.0,w_pad=3.0)
    plt.show()