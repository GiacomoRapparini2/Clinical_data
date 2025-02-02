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


def load_csv_from_json(json_path, key):
    """Load a CSV from a path specified in a JSON file."""
    with open(json_path, 'r') as file:
        paths = json.load(file)
    return pd.read_csv(paths[key]['path'])

# Function to standardize data
def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)

# Function to apply PCA
def apply_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df, pca

# Function to save PCA results and loadings
def save_pca_results(pca_df, pca, data_columns, result_path, loadings_path):
    pca_df.to_csv(result_path, index=False)
    loadings = pd.DataFrame(pca.components_.T, columns=pca_df.columns, index=data_columns)
    loadings.to_csv(loadings_path)

# Function to encode a categorical column
def encode_column(data, column):
    data[f'{column}_encoded'] = data[column].astype('category').cat.codes
    return data

# Function to plot a scatter plot for the PCA results
def plot_pca_scatter(pca_df, clinical_data, color_by, title, colormap='viridis', save_path=None):
    plt.figure()
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=clinical_data[color_by], cmap=colormap)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA - {title}')
    plt.colorbar(scatter, label=color_by)
    if save_path:
        plt.savefig(save_path)
    plt.close()