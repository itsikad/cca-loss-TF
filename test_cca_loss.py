import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import wbdata
import os.path
import scipy.linalg
from src.cca_loss import cca_loss
from sklearn.cross_decomposition import CCA


def load_word_bank_dataset():
    """
    This function loads the World Bank Data and return it as NxD numpy arrays
    """
    fert_dataset_path = './demo/WorldBankData/fertility_rate.csv'
    life_exp_dataset_path = './demo/WorldBankData/life_expectancy.csv'
    years_str_list = [str(year) for year in range(1960, 2017)]
    if os.path.exists(fert_dataset_path) & os.path.exists(life_exp_dataset_path):
        # If files exists, load from files
        # Load and drop rows with missing values
        fert_rate = pd.read_csv(fert_dataset_path).dropna()
        life_exp = pd.read_csv(life_exp_dataset_path).dropna()
        country_field_name = 'Country Code'
    else:
        # If files don't exist, download data with wbdata instead
        # Get life expectancy and fertility rate data
        life_exp = wbdata.get_dataframe(indicators={"SP.DYN.LE00.IN": 'value'}).unstack(
            level=0).transpose().reset_index()
        fert_rate = wbdata.get_dataframe(indicators={"SP.DYN.TFRT.IN": 'value'}).unstack(
            level=0).transpose().reset_index()

        # Keep only country name and years columns, filter row with N/A's
        life_exp = life_exp[['country'] + years_str_list].dropna()
        fert_rate = fert_rate[['country'] + years_str_list].dropna()
        country_field_name = 'country'

    # Keep only countries which appear on both dataframes
    valid_countries = list(set(life_exp[country_field_name]) & set(fert_rate[country_field_name]))
    life_exp = life_exp[life_exp[country_field_name].isin(valid_countries)]
    fert_rate = fert_rate[fert_rate[country_field_name].isin(valid_countries)]

    # Convert to numpy
    life_exp = life_exp[years_str_list].to_numpy()
    fert_rate = fert_rate[years_str_list].to_numpy()

    # Apply CCA
    cca_transformer = CCA(n_components=2)
    life_exp_cca, fert_rate_cca = cca_transformer.fit_transform(fert_rate, life_exp)
    return life_exp_cca, fert_rate_cca


# def gen_multivariate_random_normal(num_samples, num_features):
#     mean_vec = np.zeros(2)
#     cov_mat = np.array([[2, 0.4], [0.4, 0.25]])
#     dataset = np.random.multivariate_normal(mean_vec, cov_mat, size=(num_samples))

class TestCCALoss(unittest.TestCase):
    """
    This class defines the tests for CCA loss.
    """
    def test_covariance(self):
        """
        This test checks that the calculated loss is correct,
        i.e., that trace(sqrt(R^T*R)) is correct
        """
        # Set parameters
        REG_1 = 0.0
        REG_2 = 0.0
        EPS = 1e-6  # Step for gradient approximation
        MAX_ERR = 1e-6

        # Load data
        life_expectancy, fertility_rate = load_word_bank_dataset()

        # Compute loss using the tested unit
        loss = cca_loss(life_expectancy, fertility_rate, r1=REG_1, r2=REG_2)

        # Compute corr coeffs
        corrcoeffs = np.corrcoef(life_expectancy, fertility_rate, rowvar=False)

        # Compute relative error
        relative_arror = np.abs(corrcoeffs[0, 1] + loss) / corrcoeffs[0, 1]  # True only for 1 componenets

        self.assertTrue(relative_arror < MAX_ERR)


    def test_numerical_gradients(self):
        """"
        This test checks that the calculated gradients are correct
        by comparing them to a numerically approximated gradients.
        """
        # Set parameters
        REG_1 = 0.0 # 1e-3
        REG_2 = 0.0 # 1e-3
        EPS = 1e-3  # Step for gradient approximation
        MAX_ERR = 1e-6

        # Load datasets
        life_expectancy, fertility_rate = load_word_bank_dataset()

        # Initialize arrays
        num_countries, num_years = life_expectancy.shape
        delta = np.zeros_like(life_expectancy)
        approx_grad_x1 = np.zeros_like(life_expectancy)
        approx_grad_x2 = np.zeros_like(life_expectancy)

        # Compute analytically
        life_expectancy_tensor = tf.convert_to_tensor(life_expectancy, dtype=tf.float32)
        fertility_rate_tensor = tf.convert_to_tensor(fertility_rate, dtype=tf.float32)

        # Need to work in the context of GradientTape to compute gradients
        # Each tape can only compute one gradient at a time
        with tf.GradientTape() as tape:
            tape.watch(life_expectancy_tensor)
            loss = cca_loss(life_expectancy_tensor, fertility_rate_tensor, r1=REG_1, r2=REG_2)
        analytic_grad_x1 = tape.gradient(loss, life_expectancy_tensor).numpy()
        del tape

        with tf.GradientTape() as tape:
            tape.watch(fertility_rate_tensor)
            loss = cca_loss(life_expectancy_tensor, fertility_rate_tensor, r1=REG_1, r2=REG_2)
        analytic_grad_x2 = tape.gradient(loss, fertility_rate_tensor).numpy()
        del tape

        # Numerical gradients approximation
        for country in range(num_countries):
            for year in range(num_years):
                delta[country, year] = EPS
                approx_grad_x1[country, year] = (cca_loss(life_expectancy + delta,
                                                          fertility_rate, r1=REG_1, r2=REG_2) -
                                                 cca_loss(life_expectancy - delta,
                                                          fertility_rate, r1=REG_1, r2=REG_2)) / (2*EPS)
                approx_grad_x2[country, year] = (cca_loss(life_expectancy,
                                                          fertility_rate + delta, r1=REG_1, r2=REG_2) -
                                                 cca_loss(life_expectancy,
                                                          fertility_rate - delta, r1=REG_1, r2=REG_2)) / (2*EPS)
                delta[country, year] = 0

        # Compute diff
        grad_rel_err_1 = np.sum(np.abs(analytic_grad_x1 - approx_grad_x1) / np.abs(approx_grad_x1))
        grad_rel_err_2 = np.sum(np.abs(analytic_grad_x2 - approx_grad_x2) / np.abs(approx_grad_x2))

        # Test results
        self.assertTrue(grad_rel_err_1 < MAX_ERR, msg='Gradient error w.r.t input x1 too high')
        self.assertTrue(grad_rel_err_2 < MAX_ERR, msg='Gradient error w.r.t input x2 too high')


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    unittest.main()
