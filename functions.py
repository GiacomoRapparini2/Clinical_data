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
    Calculates the correlation matrix, plots it, and saves the correlation values to a CSV file.

    Parameters:
    correlation_data (pd.DataFrame): The data to calculate the correlation matrix on.
    save_dir (str): The directory to save the correlation matrix plot and CSV file.

    Returns:
    correlation_matrix: The correlation matrix.
    """
    # Calculate the correlation matrix
    correlation_matrix = correlation_data.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    plt.matshow(correlation_matrix)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.colorbar()
    plt.title('Correlation Matrix', pad=20)
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))

    # Create a DataFrame with the correlation values
    correlation_values_list = []
    for col1, col2 in itertools.combinations(correlation_matrix.columns, 2):
        correlation_values_list.append({'feature1': col1, 'feature2': col2, 'correlation': correlation_matrix.loc[col1, col2]})
    correlation_values = pd.DataFrame(correlation_values_list)

    # Sort the DataFrame by the correlation values
    correlation_values = correlation_values.sort_values(by='correlation', ascending=False)

    # Save the correlation values to a csv file 
    correlation_values.to_csv(os.path.join(save_dir, 'correlation_clinical.csv'), index=False)

    # Return the correlation matrix
    return correlation_matrix

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

# Function to apply PCA
def apply_pca(data, n_components=3):
    """
    Applies PCA to the given data.

    Parameters:
    data (pd.DataFrame): The data to apply PCA on.
    n_components (int): The number of principal components to compute.

    Returns:
    tuple: A tuple containing the PCA result as a DataFrame and the PCA object.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df, pca

# Function to save PCA results and loadings
def save_pca_results(pca_df, pca, data_columns, result_path, loadings_path):
    """
    Saves the PCA results and loadings to CSV files.

    Parameters:
    pca_df (pd.DataFrame): The DataFrame containing the PCA results.
    pca (PCA): The PCA object.
    data_columns (Index): The columns of the original data.
    result_path (str): The path to save the PCA results CSV file.
    loadings_path (str): The path to save the PCA loadings CSV file.
    """
    pca_df.to_csv(result_path, index=False)
    loadings = pd.DataFrame(pca.components_.T, columns=pca_df.columns, index=data_columns)
    loadings.to_csv(loadings_path)

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