import os
import unittest
import pandas as pd
import numpy as np
from functions import preprocess_clinical_data, calculate_and_save_correlation, apply_and_save_pca, encode_column, standardize_data

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for saving files
        self.test_dir = 'test_output'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a sample DataFrame for testing (clinical data)
        self.data = pd.DataFrame({
            'age': [35, 45, 55, 65, 75],
            'roi_volume': [100, 150, 200, 250, 300],
            'ltsw_to_ct': [50, 100, 150, 200, 250],
            'intensity_cbf_diff': [40, 80, 120, 160, 200],
            'intensity_tm_diff': [60, 100, 140, 180, 220],
            'sex': ['M', 'F', 'M', 'F', 'M']
        })

        # Create a sample DataFrame for testing (ROI volumes)
        self.roi_volumes = pd.DataFrame({
            'patient': [1, 2, 3, 4, 5],
            'roi_volume': [100, 150, 200, 250, 300],
            'tot_volume': [1000, 1500, 2000, 2500, 3000],
            'roi_vol_scaled': [120, 180, 230, 260, 300]
        })

        # Create a sample DataFrame for testing (median results)
        self.medians = pd.DataFrame({
            'patient': [1, 2, 3, 4, 5],
            'region': ['brain', 'roi', 'penum', 'roi', 'brain'],
            'feature': ['intensity_mtt', 'intensity_tm', 'intensity_cbv', 'intensity_cbf', 'intensity_cbv'],
            'median': [35, 40, 150, 160, 200]
        })


    def tearDown(self):
        # Remove the temporary directory after tests
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_preprocess_clinical_data(self):
        """
        Test the preprocess_clinical_data function.

        This test verifies that the preprocess_clinical_data function correctly processes
        clinical data, ROI volumes, and median results, and returns the expected outputs.

        Steps:
        1. Create mock data files for clinical data, ROI volumes, and median results.
        2. Call the preprocess_clinical_data function with the paths to the mock data files.
        3. Assert that the returned clinical_data, roi_volumes, and medians are instances of pd.DataFrame.
        4. Assert that the returned results directory is the same as the test directory.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        paths = {
            'clinical_data': {'path': os.path.join(self.test_dir, 'clinical_data.csv')},
            'results_folder': {'path': self.test_dir}
        }
        # Create mock data files
        self.data.to_csv(paths['clinical_data']['path'], index=False)
        self.roi_volumes.to_csv(os.path.join(self.test_dir, 'roi_volumes_scaled.csv'), index=False)
        self.medians.to_csv(os.path.join(self.test_dir, 'median_results.csv'), index=False)

        clinical_data, roi_volumes, medians, res_dir = preprocess_clinical_data(paths)
        self.assertIsInstance(clinical_data, pd.DataFrame)
        self.assertIsInstance(roi_volumes, pd.DataFrame)
        self.assertIsInstance(medians, pd.DataFrame)
        self.assertEqual(res_dir, self.test_dir)

    def test_calculate_and_save_correlation(self):
        """
        Test the calculate_and_save_correlation function.

        This test verifies that the calculate_and_save_correlation function correctly
        calculates the correlation matrix from the provided data and saves it to the
        specified directory.

        Assertions:
            - The returned correlation matrix is an instance of pandas DataFrame.
            - A file named 'correlation_matrix.csv' exists in the specified test directory.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        correlation_matrix = calculate_and_save_correlation(self.data.drop(columns=['sex']), self.test_dir)
        self.assertIsInstance(correlation_matrix, pd.DataFrame)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'correlation_matrix.csv')))

    def test_apply_and_save_pca(self):
        """
        Test the apply_and_save_pca function.

        This test function standardizes the specified columns of the dataset, applies PCA 
        with 3 components, and saves the resulting PCA dataframe and loadings to CSV files. 
        It then checks if the resulting PCA dataframe is a pandas DataFrame and verifies 
        the existence of the saved CSV files.

        Steps:
        1. Standardize the data for the specified columns.
        2. Apply PCA with 3 components and save the results to CSV files.
        3. Assert that the resulting PCA dataframe is an instance of pandas DataFrame.
        4. Assert that the PCA results CSV file exists.
        5. Assert that the PCA loadings CSV file exists.

        Raises:
            AssertionError: If the PCA dataframe is not a pandas DataFrame or if the 
                            expected CSV files do not exist.
        """
        standardized_data = standardize_data(self.data[['age', 'roi_volume', 'ltsw_to_ct', 'intensity_cbf_diff', 'intensity_tm_diff']])
        pca_df, pca = apply_and_save_pca(standardized_data, n_components=3, 
                                         result_path=os.path.join(self.test_dir, 'pca_clinical.csv'), 
                                         loadings_path=os.path.join(self.test_dir, 'loadings_pca3d.csv'))
        self.assertIsInstance(pca_df, pd.DataFrame)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'pca_clinical.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'loadings_pca3d.csv')))

    def test_encode_column(self):
        """
        Test the encode_column function to ensure it correctly encodes the 'sex' column.

        This test checks that:
        1. The encoded column 'sex_encoded' is added to the DataFrame.
        2. The 'sex_encoded' column has an integer data type.

        Assertions:
            - The 'sex_encoded' column is present in the DataFrame.
            - The 'sex_encoded' column is of integer type.
            - The encoded data matches the expected values.
        """
        encoded_data = encode_column(self.data, 'sex')
        self.assertIn('sex_encoded', encoded_data.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(encoded_data['sex_encoded']))
        expected_encoded_values = pd.Series([1, 0, 1, 0, 1], dtype=encoded_data['sex_encoded'].dtype)
        self.assertTrue(encoded_data['sex_encoded'].equals(expected_encoded_values))



if __name__ == '__main__':
    unittest.main()