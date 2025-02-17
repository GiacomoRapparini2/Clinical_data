import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import functions as fn


# Load the paths from the json file
paths = fn.load_json('paths.json')

# Preprocess the clinical data
clinical_data, roi_volumes, medians, res_dir = fn.preprocess_clinical_data(paths)

# Process medians and merge with clinical data and ROI volumes
clinical_data = fn.process_medians_and_merge(clinical_data, roi_volumes, medians)

# Create a folder to save the results
clin_res_dir = os.path.join(res_dir, 'clinical_results')
if not os.path.exists(clin_res_dir):
    os.makedirs(clin_res_dir, exist_ok=True)

# Correlation analysis ########################################################################################

# Drop the columns 'patient', 'date', 'tpa', 'tici_end' and 'sex' for correlation calculation
correlation_data = clinical_data.drop(columns=['patient', 'date', 'tpa', 'tici_end', 'sex'])

# Calculate, plot, and save the correlation matrix and values
correlation_matrix = fn.calculate_and_save_correlation(correlation_data, clin_res_dir)


# PCA analysis with DBSCAN clustering ######################################################################

# Only keep the columns: 'age', 'roi_volume', 'ltsw_to_ct', 'intensity_cbf_diff', 'intensity_tm_diff'
pca_data = correlation_data[['age', 'roi_volume', 'ltsw_to_ct', 'intensity_cbf_diff', 'intensity_tm_diff']]

# Drop columns with NaN values
pca_data = pca_data.dropna(axis=1)

# Standardize the data
data_scaled_df = fn.standardize_data(pca_data)

# Apply PCA to the data frame and save the results
pca_df, pca = fn.apply_and_save_pca(data_scaled_df, n_components=3, 
                                    result_path=os.path.join(clin_res_dir, 'pca_clinical.csv'), 
                                    loadings_path=os.path.join(clin_res_dir, 'loadings_pca3d.csv'))
pca_df['patient'] = clinical_data['patient']

# Perform DBSCAN clustering and save the results
pca_df = fn.perform_dbscan_clustering(pca_df, clin_res_dir)


#########################################################################################################

# Do the PCA only for the perfusion parameters medians
perf_data = clinical_data[['intensity_cbv_diff', 'intensity_cbf_diff', 'intensity_mtt_diff', 'intensity_tm_diff']]

# Drop columns with NaN values
perf_data = perf_data.dropna(axis=1)

# Standardize the data
perf_data_scaled_df = fn.standardize_data(perf_data)

# Apply PCA to the data frame and save the results
pca_perf_df, pca_perf = fn.apply_and_save_pca(perf_data_scaled_df, n_components=2, 
                                              result_path=os.path.join(clin_res_dir, 'pca_perf_clinical.csv'), 
                                              loadings_path=os.path.join(clin_res_dir, 'loadings_pca_perf.csv'))
pca_perf_df['patient'] = clinical_data['patient']

# Encode the 'sex' column to numeric values
clinical_data = fn.encode_column(clinical_data, 'sex')

# Plot a scatter plot for PC1 vs PC2 with the points colored by the 'sex', 'age', 'roi_volume' and 'ltsw_to_ct' columns of the clinical data
fn.plot_pca_scatter(pca_perf_df, clinical_data, 'sex_encoded', 'Sex', save_path=os.path.join(clin_res_dir, 'pca_sex.png'))
fn.plot_pca_scatter(pca_perf_df, clinical_data, 'age', 'Age', save_path=os.path.join(clin_res_dir, 'pca_age.png'))
fn.plot_pca_scatter(pca_perf_df, clinical_data, 'roi_volume', 'ROI Volume', save_path=os.path.join(clin_res_dir, 'pca_roi_volume.png'))
fn.plot_pca_scatter(pca_perf_df, clinical_data, 'ltsw_to_ct', 'LTSW to CT', save_path=os.path.join(clin_res_dir, 'pca_ltsw_to_ct.png'))
