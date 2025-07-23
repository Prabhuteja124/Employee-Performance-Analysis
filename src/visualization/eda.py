import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

def department_wise_performance(df):
    return df.groupby("EmpDepartment")["PerformanceRating"].mean().sort_values(ascending=False)

def get_top_correlated_features(df, target='PerformanceRating',top_k=5):
    corr = df.corr(numeric_only=True)
    target_corr = corr[target].drop(target)
    return target_corr.abs().sort_values(ascending=False).head(top_k)

def plot_correlation_heatmap(df):
    num_cols = df.select_dtypes(exclude='O').columns.tolist()
    corr = df[num_cols].corr()
    num_featuers=len(num_cols)
    width_per_feature = 1.0
    height_per_feature = 0.7
    fig_width = max(8,num_featuers * width_per_feature)
    fig_height = max(6,num_featuers * height_per_feature)

    plt.figure(figsize=(fig_width,fig_height))
    sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,linewidths=0.5,square=True,cbar_kws={"shrink": 0.75})
    plt.title("Correlation Heatmap of Numerical Features",fontsize=16,fontweight='bold')
    plt.xticks(rotation=45,ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df:pd.DataFrame)-> pd.DataFrame:

    numerical_columns=df.select_dtypes(include='number').columns.tolist()
    continuous_columns=[col for col in numerical_columns if df[col].nunique()>=15]
    outlier_summary = []

    for col in continuous_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = outliers.shape[0]
        outlier_percentage = (outlier_count /len(df)) * 100

        outlier_summary.append({
            'Column': col,
            'Outlier Count': outlier_count,
            'Outlier Percentage': round(outlier_percentage,2)
        })

    outlier_df = pd.DataFrame(outlier_summary)
    print("Outlier Summary (IQR Method):\n")
    return outlier_df

def plot_boxplots(df:pd.DataFrame)-> None:
    """
    Plots boxplots for the given numerical columns to check for outliers.
    """
    numerical_columns=df.select_dtypes(include='number').columns.tolist()
    continuous_columns=[col for col in numerical_columns if df[col].nunique()>=15]
    n_cols=3
    n_rows =math.ceil(len(continuous_columns)/n_cols)
    figsize=(n_cols * 6,n_rows * 4)
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize, facecolor='white')
    plt.suptitle(" Boxplots for Outlier Detection", fontsize=24, fontweight='bold', color='black', y=1.02)

    if not continuous_columns:
        print("No continuous numerical columns found for boxplot.")
        return None

    for idx, column in enumerate(continuous_columns,1):
        ax = plt.subplot(n_rows,n_cols,idx)
        sns.boxplot(y=df[column],palette='Set3',ax=ax,width=0.3,linewidth=1.5)
        ax.set_title(f'{column}',fontsize=14,fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('')
        ax.grid(True)

    plt.tight_layout(pad=2.0)
    plt.show()