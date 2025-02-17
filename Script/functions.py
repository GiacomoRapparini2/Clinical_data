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



def preprocess_clinical_data(paths):
    """
    Preprocess the clinical data based on the provided paths.

    Parameters:
    paths (dict): A dictionary containing paths to the clinical data and results folder.

    Returns:
    tuple: A tuple containing the preprocessed clinical data, ROI volumes, medians, and results directory.
    """
    # Path to the csv file containing clinical data
    file_path = paths['clinical_data']['path']

    # Load the clinical data
    clinical_data = load_csv(file_path, fill_na=1)

    # Path to the directory containing the results of the previous analysis
    res_dir = paths['results_folder']['path']

    # Load the ROI volumes (scaled by the brain volume)
    roi_volumes = load_csv(os.path.join(res_dir, 'roi_volumes_scaled.csv'), fill_na=0)

    # Preprocess the 'patient' column to extract only the number
    roi_volumes['patient'] = roi_volumes['patient'].astype(str).str.extract(r'(\d+)').astype(int)

    # Sort the DataFrame by the 'patient' column
    roi_volumes = roi_volumes.sort_values(by='patient')

    # Drop the column 'tot_volume'
    roi_volumes = roi_volumes.drop(columns=['tot_volume'])

    # Load the medians
    medians = load_csv(os.path.join(res_dir, 'median_results.csv'), fill_na=1)

    # Drop rows where 'region' column is 'contr'
    medians = medians[medians['region'] != 'contr']

    # Preprocess the 'patient' column to extract only the number
    medians['patient'] = medians['patient'].astype(str).str.extract(r'(\d+)').astype(int)

    # Sort the DataFrame by the 'patient' column
    medians = medians.sort_values(by='patient')

    return clinical_data, roi_volumes, medians, res_dir


def process_medians_and_merge(clinical_data, roi_volumes, medians):
    """
    Process medians and merge with clinical data and ROI volumes.

    Parameters:
    clinical_data (pd.DataFrame): The clinical data.
    roi_volumes (pd.DataFrame): The ROI volumes data.
    medians (pd.DataFrame): The medians data.

    Returns:
    pd.DataFrame: The merged clinical data.
    """
    # Get unique values for the specified columns
    parameters = medians['feature'].unique()
    patients = medians['patient'].unique()

    # Create a DataFrame with the first column as 'patient' and other 4 as the 
    # absolute median difference between the brain and the roi for the 4 features
    median_diff = pd.DataFrame(columns=['patient'])
    median_diff['patient'] = patients

    for patient in patients:
        for parameter in parameters:
            brain_median = medians[(medians['feature'] == parameter) & (medians['region'] == 'brain') & (medians['patient'] == patient)]['median'].values
            roi_median = medians[(medians['feature'] == parameter) & (medians['region'] == 'roi') & (medians['patient'] == patient)]['median'].values
            if len(brain_median) > 0 and len(roi_median) > 0:
                median_diff.loc[median_diff['patient'] == patient, f'{parameter}_diff'] = abs(brain_median - roi_median)
            else: 
                median_diff.loc[median_diff['patient'] == patient, f'{parameter}_diff'] = None

    # Merge the clinical data with the median_diff DataFrame and the roi_volumes DataFrame
    clinical_data = clinical_data.merge(roi_volumes, on='patient')
    clinical_data = clinical_data.merge(median_diff, on='patient')

    return clinical_data


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


def perform_dbscan_clustering(pca_df, clin_res_dir, n_neighbors=5, min_samples=7, percentile=95):
    """
    Perform DBSCAN clustering on PCA data and save the results.

    Parameters:
    pca_df (pd.DataFrame): The PCA data.
    clin_res_dir (str): The directory to save the results.
    n_neighbors (int): The number of neighbors to use for k-NN.
    min_samples (int): The minimum number of samples for DBSCAN.
    percentile (int): The percentile to use for determining the optimal eps value.

    Returns:
    pd.DataFrame: The PCA data with clustering labels.
    """
    # Find the optimal eps using k-NN
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(pca_df[['PC1', 'PC2', 'PC3']])
    distances, indices = neighbors_fit.kneighbors(pca_df[['PC1', 'PC2', 'PC3']])

    # Sort the distances and plot the k-distance graph
    distances = np.sort(distances[:, -1])  # Use last column dynamically
    plt.figure()
    plt.plot(distances)
    plt.title('k-NN Distance Graph')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-NN distance')
    plt.savefig(os.path.join(clin_res_dir, 'knn_distance_graph.png'))
    plt.show()

    # Choose the optimal eps value from the k-distance graph
    optimal_eps = np.percentile(distances, percentile)

    # Perform DBSCAN clustering with the optimal eps
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    clustering_labels = dbscan.fit_predict(pca_df[['PC1', 'PC2', 'PC3']])

    # Add clustering labels to the PCA DataFrame
    pca_df['cluster'] = clustering_labels

    # Plot a 3D scatter plot for PC1 vs PC2 vs PC3 with the points colored by the cluster
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['cluster'], cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('PCA - Clinical Data')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(os.path.join(clin_res_dir, 'pca_clinical_clusters.png'))
    plt.show()
    plt.close()

    return pca_df


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
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=clinical_data[color_by], cmap=colormap)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA - {title}')
    plt.colorbar(label=color_by)
    if save_path:
        plt.savefig(save_path)
    plt.close()
