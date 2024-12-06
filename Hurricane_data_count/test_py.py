import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.api import families
from statsmodels.genmod.generalized_linear_model import GLM

data = pd.read_csv('hurricane_data.csv')

def log_likelihood(params, data):
    alpha_0, alpha_long, alpha_lat, b_enso, b_nao, b_african, b_sst, gamma_h, gamma_v, gamma_c = params
    
    alpha_ij = alpha_0 + alpha_long[data['grid_i']] + alpha_lat[data['grid_j']]
    climate_effect = b_enso * data['enso_index'] + b_nao * data['nao_index'] + b_african * data['african_rainfall'] + b_sst * data['sst']
    
    coupling_effect = gamma_h * (data['hurricane_count_left'] + data['hurricane_count_right']) + \
                      gamma_v * (data['hurricane_count_up'] + data['hurricane_count_down']) + \
                      gamma_c * data['hurricane_count_lag']
    
    log_lambda = alpha_ij + climate_effect + coupling_effect
    log_likelihood = np.sum(data['hurricane_count'] * log_lambda - np.exp(log_lambda) - np.log(np.math.factorial(data['hurricane_count'])))
    
    return -log_likelihood

initial_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result = minimize(log_likelihood, initial_params, args=(data,), method='BFGS')
optimal_params = result.x
print("Optimal Parameters:", optimal_params)
