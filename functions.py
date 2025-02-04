import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns


def load_json(file_path):
    """Load JSON file and return the parsed content."""
    with open(file_path, 'r') as file:
        return json.load(file)


def load_csv(file_path, fill_na=None):
    """Load CSV file into a DataFrame, optionally filling NaN values."""
    df = pd.read_csv(file_path)
    if fill_na is not None:
        df = df.fillna(fill_na)
    return df


# Function to calculate, plot, and save the correlation matrix and values
def calculate_and_save_correlation(correlation_data, save_dir):
    """
    Calculate, plot, and save the correlation matrix and values.
    
    Parameters:
    correlation_data (DataFrame): DataFrame containing the data for correlation.
    save_dir (str): Directory to save the correlation matrix and scatter plots.
    
    Returns:
    correlation_matrix (DataFrame): DataFrame containing the correlation matrix.
    """
    # Calculate the correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Save the correlation matrix to a CSV file
    correlation_matrix.to_csv(os.path.join(save_dir, 'correlation_matrix.csv'))
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Create scatter plots for columns with correlation > 0.5 and < 1
    high_corr_pairs = [(col1, col2) for col1, col2 in itertools.combinations(correlation_matrix.columns, 2) 
                       if 0.5 < abs(correlation_matrix.loc[col1, col2]) < 1]

    for col1, col2 in high_corr_pairs:
        plt.figure()
        plt.scatter(correlation_data[col1], correlation_data[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter plot between {col1} and {col2}')
        plt.savefig(os.path.join(save_dir, f'scatter_{col1}_{col2}.png'))
        plt.close()
    
    # Return the correlation matrix
    return correlation_matrix


# Function to create scatter plots for columns with correlation > threshold and < 1
def corr_scatter_plots(correlation_data, correlation_matrix, threshold, save_dir):
    """
    Create scatter plots for columns with correlation > threshold and < 1.
    
    Parameters:
    correlation_data (DataFrame): DataFrame containing the data for correlation.
    correlation_matrix (DataFrame): DataFrame containing the correlation matrix.
    threshold (float): The correlation threshold to consider.
    save_dir (str): Directory to save the scatter plots.
    """
    high_corr_pairs = [(col1, col2) for col1, col2 in itertools.combinations(correlation_matrix.columns, 2) 
                       if threshold < abs(correlation_matrix.loc[col1, col2]) < 1]

    for col1, col2 in high_corr_pairs:
        plt.figure()
        plt.scatter(correlation_data[col1], correlation_data[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Scatter plot between {col1} and {col2}')
        plt.savefig(os.path.join(save_dir, f'scatter_{col1}_{col2}.png'))
        plt.close()


# Function to standardize data
def standardize_data(data):
    """
    Standardizes the given data using StandardScaler.

    Parameters:
    data (pd.DataFrame): The data to be standardized.

    Returns:
    pd.DataFrame: The standardized data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)


def apply_and_save_pca(data, n_components, result_path, loadings_path):
    """
    Applies PCA to the given data and saves the results and loadings to CSV files.

    Parameters:
    data (pd.DataFrame): The data to apply PCA on.
    n_components (int): The number of principal components to compute.
    result_path (str): The path to save the PCA results CSV file.
    loadings_path (str): The path to save the PCA loadings CSV file.

    Returns:
    tuple: A tuple containing the PCA result as a DataFrame and the PCA object.
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Save PCA results
    pca_df.to_csv(result_path, index=False)
    
    # Save loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=data.columns)
    loadings.to_csv(loadings_path)
    
    return pca_df, pca


# Function to encode a categorical column
def encode_column(data, column):
    """
    Encodes a categorical column to numeric values.

    Parameters:
    data (pd.DataFrame): The data containing the column to encode.
    column (str): The name of the column to encode.

    Returns:
    pd.DataFrame: The data with the encoded column.
    """
    data[f'{column}_encoded'] = data[column].astype('category').cat.codes
    return data


# Function to plot PCA scatter
def plot_pca_scatter(pca_df, clinical_data, color_by, title, colormap='viridis', save_path=None):
    """
    Plots a scatter plot for PCA results.

    Parameters:
    pca_df (pd.DataFrame): The DataFrame containing the PCA results.
    clinical_data (pd.DataFrame): The clinical data containing the column to color by.
    color_by (str): The column name to color the points by.
    title (str): The title of the plot.
    colormap (str): The colormap to use for the plot.
    save_path (str, optional): The path to save the plot. If None, the plot is not saved.
    """
    plt.figure()
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=clinical_data[color_by], cmap=colormap)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA - {title}')
    plt.colorbar(scatter, label=color_by)
    if save_path:
        plt.savefig(save_path)
    plt.close()
