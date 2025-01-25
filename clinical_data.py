import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the path to the csv file from paths.json
with open('paths.json', 'r') as file:
    paths = json.load(file)

# Path to the csv file containing clinical data
file_path = paths['clinical_data']['path']

# Load the clinical data
clinical_data = pd.read_csv(file_path)

# If there are missing values in any of the columns, fill them with NaN
clinical_data = clinical_data.fillna('NaN')

# Path to the directory containing the results of the analysis
res_dir = paths['results_folder']['path']

# Load the ROI volumes
roi_volumes = pd.read_csv(os.path.join(res_dir, 'roi_volumes.csv'))

# Preprocess the 'patient' column to extract only the number
roi_volumes['patient'] = roi_volumes['patient'].astype(str).str.extract(r'(\d+)').astype(int)

# Load the medians
medians = pd.read_csv(os.path.join(res_dir, 'median_results.csv'))

# Drop rows where the 'median' column has NaN values
medians = medians.dropna(subset=['median'])

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
clin_res = os.path.join(res_dir, 'clinical_results')
if not os.path.exists(clin_res):
    os.makedirs(clin_res)

# Correlation analysis

# Drop the columns 'patient', 'date', 'tpa', 'tici_end' and 'sex' for correlation calculation
correlation_data = clinical_data.drop(columns=['patient', 'date', 'tpa', 'tici_end', 'sex'])

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Print the correlation matrix
print(correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.colorbar()
plt.title('Correlation Matrix', pad=20)
plt.savefig(os.path.join(clin_res, 'correlation_matrix.png'))

# Create a DataFrame with the correlation values
correlation_values_list = []
for col1, col2 in itertools.combinations(correlation_matrix.columns, 2):
    correlation_values_list.append({'feature1': col1, 'feature2': col2, 'correlation': correlation_matrix.loc[col1, col2]})
correlation_values = pd.DataFrame(correlation_values_list)

# Sort the DataFrame by the correlation values
correlation_values = correlation_values.sort_values(by='correlation', ascending=False)

# Save the correlation values to a csv file 
correlation_values.to_csv(os.path.join(clin_res, 'correlation_clinical.csv'), index=False)

# Create scatter plots for columns with correlation > 0.5 and < 1
high_corr_pairs = [(col1, col2) for col1, col2 in itertools.combinations(correlation_matrix.columns, 2) 
                   if 0.5 < abs(correlation_matrix.loc[col1, col2]) < 1]

for col1, col2 in high_corr_pairs:
    plt.figure()
    plt.scatter(correlation_data[col1], correlation_data[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Scatter plot between {col1} and {col2}')
    #plt.show()
    plt.savefig(os.path.join(clin_res, f'scatter_{col1}_{col2}.png'))
    plt.close()

# PCA analysis

# Drop the columns 'aspects', 'mrs_basal', 'mrs_90d', 'tici_end_code'  for PCA calculation
pca_data = correlation_data.drop(columns=['aspects', 'mrs_basal', 'mrs_90d', 'tici_end_code'])

# Drop columns with NaN values
pca_data = pca_data.dropna(axis=1)

# Initialize the scaler
scaler = StandardScaler()

# Standardize the data
data_scaled = scaler.fit_transform(pca_data)

# Convert back to a DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=pca_data.columns)

# Check the standardized data
print(data_scaled_df.head())

# Apply PCA to the data frame
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_scaled_df)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
pca_df['patient'] = clinical_data['patient']

# Save the PCA results to a csv file
pca_df.to_csv(os.path.join(clin_res, 'pca_clinical.csv'), index=False)
