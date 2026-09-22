# Reads data_bram.csv and fits with the simplest model 
# Poly0 : y(x)=y_0
# Poly1 : y(x)=y_0 + a*x + b*x**2
# Poly2 : y(x)=y_0 + b*x**2
# computes dimensionless variables directly from lattice data,
# calculates effective errors and p-values,
# and saves the results for each model to fit_results_poly.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import json
import os

# =============================================================================
# 1. Constants & Scale Conversions
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_MeV = r0_fm / hbar_c

A_types = ['Ar0', 'Api12']
A_index = 1

data_file = f'Data/data_bram_{A_types[A_index]}.csv'
filename = f'Bram_Poly/fit_results_poly_{A_types[A_index]}.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(filename), exist_ok=True)

# =============================================================================
# 2. Data Processing Function & Dimensionless Models
# =============================================================================
def calculate_dimensionless(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    
    y = a2sigma * (r0_a ** 2)
    y_err = y * np.sqrt((a2sigma_err / a2sigma)**2 + (2 * r0_a_err / r0_a)**2)
    
    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)
    
    return x, x_err, y, y_err


def poly0(x, y_0):
    return np.full_like(x, y_0, dtype=float)

def poly1(x, y_0, a, b):
    return y_0 + a * x + b * x**2

def poly2(x, y_0, b):
    return y_0 + b * x**2

# =============================================================================
# 3. Load Data & Define Model Registry
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')
results = []
data_types = ['bare', 'smeared']

models = {
    'Poly0': poly0,
    'Poly1': poly1,
    'Poly2': poly2
}

# =============================================================================
# 4. Fitting Routine
# =============================================================================
for prefix in data_types:
    x_list, x_err_list, y_list, y_err_list = [], [], [], []
    for _, row in df.iterrows():
        x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
        x_list.append(x_val)
        x_err_list.append(x_e)
        y_list.append(y_val)
        y_err_list.append(y_e)
        
    x_arr, x_err_arr = np.array(x_list, dtype=float), np.array(x_err_list, dtype=float)
    y_arr, y_err_arr = np.array(y_list, dtype=float), np.array(y_err_list, dtype=float)

    for model_name, model_func in models.items():
        res_dict = {
            'Model': model_name,
            'Data_Type': prefix,
            'y_0': np.nan, 'y_0_err': np.nan,
            'a': np.nan, 'a_err': np.nan,
            'b': np.nan, 'b_err': np.nan,
            'cov_matrix': ''
        }

        try:
            # Perform fit using standard y_err weighting
            popt, pcov = curve_fit(model_func, x_arr, y_arr, sigma=y_err_arr, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))

            # Store parameters correctly based on which model is being fit
            if model_name == 'Poly0':
                res_dict['y_0'] = popt[0]
                res_dict['y_0_err'] = perr[0]
            elif model_name == 'Poly1':
                res_dict['y_0'], res_dict['a'], res_dict['b'] = popt
                res_dict['y_0_err'], res_dict['a_err'], res_dict['b_err'] = perr
            elif model_name == 'Poly2':
                res_dict['y_0'], res_dict['b'] = popt
                res_dict['y_0_err'], res_dict['b_err'] = perr

            # Convert covariance matrix array to JSON string for safe CSV storage
            res_dict['cov_matrix'] = json.dumps(pcov.tolist())

            # Evaluate model and calculate chi squared
            y_fit = model_func(x_arr, *popt)
            chi_sq = np.sum(((y_arr - y_fit) / y_err_arr)**2)
            dof = len(x_arr) - len(popt)

            # Càlculs estadístics
            res_dict['chi_sq'] = chi_sq
            res_dict['dof'] = dof
            res_dict['red_chi_sq'] = chi_sq / dof if dof > 0 else np.nan
            res_dict['p_value'] = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

            results.append(res_dict)

        except RuntimeError as e:
            print(f"Fit failed for {model_name} on {prefix} data: {e}")

# =============================================================================
# 5. Output / Safe DataFrame printing
# =============================================================================
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)

    print("\n--- POLYNOMIAL FIT COMPLETE ---")
    print("Results successfully saved to", filename)
    print("\nSummary of Extracted Parameters:")

    # Select relevant columns for terminal summary
    cols_to_print = ['Model', 'Data_Type', 'y_0', 'a', 'b', 'red_chi_sq', 'p_value']
    
    print(results_df[cols_to_print].to_string(index=False))
else:
    print("\nNo fits converged successfully. Please check your model parameters and bounds.")