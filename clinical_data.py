import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# Load the path to the csv file from paths.json
with open('paths.json', 'r') as file:
    paths = json.load(file)

# Path to the csv file containing clinical data
file_path = paths['clinical_data']['path']

# Load the clinical data
clinical_data = pd.read_csv(file_path)

# Path to the directory containing the results of the analysis
res_dir = paths['results_folder']['path']

# Load the ROI volumes
roi_volumes = pd.read_csv(os.path.join(res_dir, 'roi_volumes.csv'))

# Load the medians
medians = pd.read_csv(os.path.join(res_dir, 'median_results.csv'))

# Drop rows where the 'median' column has NaN values
medians = medians.dropna(subset=['median'])

# Drop rows where 'region' column is 'contr' (just remove this line to consider 'contr' region)
medians = medians[medians['region'] != 'contr']

# Preprocess the 'patient' column to extract only the number
medians['patient'] = medians['patient'].astype(str).str.extract(r'(\d+)').astype(int)

# Sort the DataFrame by the 'patient' column
medians = medians.sort_values(by='patient')
print(medians.head())

# Get unique values for the specified column
parameters = medians['feature'].unique()
clinical_features = clinical_data.columns
print(clinical_features)
patients = medians['patient'].unique()
