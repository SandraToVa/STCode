# Reads data_bram.csv and fits with the global dimensionless ST logarithmic model,
# computes dimensionless variables directly from lattice data,
# performs the global fit to find a universal y_0, c_l (and c2_l if LQCD=True),
# calculates effective errors and p-values,
# and saves the results to fit_results_ST_global_dim.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import json

# =============================================================================
# 1. Constants & Scale Conversions
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_MeV = r0_fm / hbar_c

A_types = ['Ar0', 'Api12']
A_index = 0
LQCD = True
# Define lower and upper limits matching parameter order: [y_0, c_l, c2_l, mu]
lower_bounds1 = [-np.inf, -np.inf, 1e-15]
upper_bounds1 = [ np.inf,  np.inf, np.inf]
bounds1 = (lower_bounds1, upper_bounds1)
# c2_l can be negative but curve_fit only accepts one-sided bounds
# First we work with positive numbers and look at the chi^2/dof and p-value to see if the fit is reasonable.
# Then we do the same for negative numbers and compare the results.
lower_bounds2 = [-np.inf, -np.inf, 200.0, 1e-15]
upper_bounds2 = [ np.inf,  np.inf,  600.0, np.inf]
bounds2 = (lower_bounds2, upper_bounds2)


data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
# Selecció dinàmica del nom de fitxer segons la variable LQCD
lqcd_str = "_full" if LQCD else ""
filename = f'fit_results{lqcd_str}_ST_global_{A_types[A_index]}.csv'

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

def model_func_dim(x, y_0, c_l, gamma):
    term1 = (c_l**2 / (2 * np.pi)) * x**2 * np.log(x**2)
    term2 = gamma * x**2
    
    return y_0 + term1 * term2

def model_func2_dim(x, y_0, c_l, c2_l, gamma):
    z = x + c_l/c2_l
    term1 = (c_l**2 / (2 * np.pi)) * z**2 * np.log(z**2)
    term2 = gamma * z**2
    
    return y_0 + term1 * term2

# =============================================================================
# 3. Load Data
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')
results = []
data_types = ['bare', 'smeared']



# =============================================================================
# 4. Fitting Routine (Global)
# =============================================================================
for prefix in data_types:
    x_list, x_err_list, y_list, y_err_list = [], [], [], []
    for _, row in df.iterrows():
        x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
        x_list.append(x_val)
        x_err_list.append(x_e)
        y_list.append(y_val)
        y_err_list.append(y_e)
        
    x_arr, x_err_arr = np.array(x_list), np.array(x_err_list)
    y_arr, y_err_arr = np.array(y_list), np.array(y_err_list)

    res_dict = {'Data_Type': prefix}

    if not LQCD:
        # -----------------------------------------------------------------
        # GLOBAL FIT: LQCD = False (Model Original)
        # -----------------------------------------------------------------
        try:
            popt_init, _ = curve_fit(
                model_func_dim, x_arr, y_arr, 
                sigma=y_err_arr, absolute_sigma=True, maxfev=10000,
                p0=[np.mean(y_arr), 1.0, 1.0], bounds=bounds1
            )
            c_l_guess = popt_init[1]
            gamma_guess = popt_init[2]
        except:
            c_l_guess = 1.0
            gamma_guess = 1.0

        def effective_y_err(c_l_est, gamma_est):
            log_arg = x_arr**2
            log_arg = np.maximum(log_arg, 1e-15) 
            df_dx = 2 * gamma_est * x_arr + (c_l_est**2 / np.pi) * x_arr * (1 + np.log(log_arg))
            return np.sqrt(y_err_arr**2 + (df_dx * x_err_arr)**2)
        

        try:
            popt, pcov = curve_fit(
                model_func_dim, x_arr, y_arr, 
                sigma=effective_y_err(c_l_guess, gamma_guess), absolute_sigma=True, maxfev=10000, bounds=bounds1
            )

            res_dict['y_0'] = popt[0]
            res_dict['y_0_err'] = np.sqrt(np.diag(pcov))[0]
            res_dict['c_l'] = popt[1]
            res_dict['c_l_err'] = np.sqrt(np.diag(pcov))[1]
            res_dict['gamma'] = popt[2]
            res_dict['gamma_err'] = np.sqrt(np.diag(pcov))[2]
            res_dict['pcov'] = json.dumps(pcov.tolist())
            res_dict['x_min'] = float(np.min(x_arr))
            res_dict['x_max'] = float(np.max(x_arr))

            residuals = y_arr - model_func_dim(x_arr, popt[0], popt[1], popt[2])
            chi_sq = np.sum((residuals / effective_y_err(popt[1], popt[2]))**2)
            dof = len(x_arr) - 3

        except RuntimeError:
            print(f"Fit failed for Data Type: {prefix} (Model 1)")
            continue

    else:
        # -----------------------------------------------------------------
        # GLOBAL FIT: LQCD = True (Nou Model amb c2_l)
        # -----------------------------------------------------------------
        try:
            popt_init, _ = curve_fit(
                model_func2_dim, x_arr, y_arr, 
                sigma=y_err_arr, absolute_sigma=True, maxfev=10000,
                p0=[np.mean(y_arr), 1.0, 200.0, 1.0],
                bounds=bounds2
            )
            c_l_guess, c2_l_guess, gamma_guess = popt_init[1], popt_init[2], popt_init[3]
        except:
            c_l_guess, c2_l_guess, gamma_guess = 1.0, 200.0, 1.0
            
        def effective_y_err2(c_l_est, c2_l_est, gamma_est):
            z = x_arr + c2_l_est/c_l_est
            log_arg = (z**2) 
            log_arg = np.maximum(log_arg, 1e-15) 
                
            df_dx = z * (2 * gamma_est + (c_l_est**2 / np.pi) * (1 + np.log(log_arg)))
            return np.sqrt(y_err_arr**2 + (df_dx * x_err_arr)**2)


        try:
            popt, pcov = curve_fit(
                model_func2_dim, x_arr, y_arr, 
                sigma=effective_y_err2(c_l_guess, c2_l_guess, gamma_guess), absolute_sigma=True, maxfev=10000, bounds=bounds2
            )
            res_dict['y_0'] = popt[0]
            res_dict['y_0_err'] = np.sqrt(np.diag(pcov))[0]
            res_dict['c_l'] = popt[1]
            res_dict['c_l_err'] = np.sqrt(np.diag(pcov))[1]
            res_dict['c2_l'] = popt[2]
            res_dict['c2_l_err'] = np.sqrt(np.diag(pcov))[2]
            res_dict['gamma'] = popt[3]
            res_dict['gamma_err'] = np.sqrt(np.diag(pcov))[3]
            res_dict['pcov'] = json.dumps(pcov.tolist())
            res_dict['x_min'] = float(np.min(x_arr))
            res_dict['x_max'] = float(np.max(x_arr))

            residuals = y_arr - model_func2_dim(x_arr, popt[0], popt[1], popt[2], popt[3])
            chi_sq = np.sum((residuals / effective_y_err2(popt[1], popt[2], popt[3]))**2)
            dof = len(x_arr) - 4

        except RuntimeError:
            print(f"Fit failed for Data Type: {prefix} (Model 2)")
            continue

    # Càlculs estadístics comuns
    res_dict['chi_sq'] = chi_sq
    res_dict['dof'] = dof
    res_dict['red_chi_sq'] = chi_sq / dof if dof > 0 else np.nan
    res_dict['p_value'] = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

    results.append(res_dict)

results_df = pd.DataFrame(results)
results_df.to_csv(filename, index=False)

print("\n--- GLOBAL DIMENSIONLESS ST FIT COMPLETE ---")
print("Model used:", "LQCD (c2_l active)" if LQCD else "Original (c2_l = 0)")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters (y_0 = r0^2 * sigma_0):")

cols_to_print = ['Data_Type', 'y_0', 'c_l']
if LQCD:
    cols_to_print.append('c2_l')
cols_to_print.extend(['gamma', 'red_chi_sq', 'p_value'])

print(results_df[cols_to_print].to_string(index=False))