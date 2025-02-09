import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import functions as fn


# Preprocess the clinical data ################################################################################

# Load the paths from the json file
paths = fn.load_json('paths.json')

# Path to the csv file containing clinical data
file_path = paths['clinical_data']['path']

# Load the clinical data
clinical_data = fn.load_csv(file_path, fill_na=1)

# Path to the directory containing the results of the previous analysis
res_dir = paths['results_folder']['path']

# Load the ROI volumes (scaled by the brain volume)
roi_volumes = fn.load_csv(os.path.join(res_dir, 'roi_volumes_scaled.csv'), fill_na=0)

# Preprocess the 'patient' column to extract only the number
roi_volumes['patient'] = roi_volumes['patient'].astype(str).str.extract(r'(\d+)').astype(int)

# Sort the DataFrame by the 'patient' column
roi_volumes = roi_volumes.sort_values(by='patient')

# Drop the column 'tot_volume'
roi_volumes = roi_volumes.drop(columns=['tot_volume'])

# Load the medians
medians = fn.load_csv(os.path.join(res_dir, 'median_results.csv'), fill_na=1)

# Drop rows where 'region' column is 'contr'
medians = medians[medians['region'] != 'contr']

# Preprocess the 'patient' column to extract only the number
medians['patient'] = medians['patient'].astype(str).str.extract(r'(\d+)').astype(int)

# Sort the DataFrame by the 'patient' column
medians = medians.sort_values(by='patient')
print(medians.head())

# Get unique values for the specified columns
parameters = medians['feature'].unique()
clinical_features = clinical_data.columns
print(clinical_features)
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

print(median_diff.head())

# Merge the clinical data with the median_diff DataFrame and the roi_volumes DataFrame
clinical_data = clinical_data.merge(roi_volumes, on='patient')
clinical_data = clinical_data.merge(median_diff, on='patient')

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

# Find the optimal eps using k-NN
neighbors = NearestNeighbors(n_neighbors=5)
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
optimal_eps = np.percentile(distances, 95)

# Perform DBSCAN clustering with the optimal eps
dbscan = DBSCAN(eps=optimal_eps, min_samples=7)
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
