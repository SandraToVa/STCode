# Reads data_bram.csv and fits with the function form paper Soto-Tarrús,,
# converts lattice units to physical units (propagating errors),
# performs the global fit to find sigma_0 and c_l for the whole dataset,
# calculates systematic errors and p-values,
# and saves the results to fit_results_ST_global.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# =============================================================================
# 1. Constants & Unit Conversions (Converting GeV to fm^-1)
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
mu = np.sqrt(210)  # MeV sqrt(sigma) from paper Soto-Tarrús
gamma_E = np.euler_gamma

A_types = ['Ar0', 'Api12']
A_index = 0

data_file = f'data_bram_{A_types[A_index]}.csv'
# Updated output file name to reflect log model and global fit
filename = f'fit_results_ST_global_{A_types[A_index]}.csv'

# =============================================================================
# 2. Data Processing Function & Logarithmic Model
# =============================================================================

def calculate_physics(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    
    a = r0_MeV / r0_a
    a_err = a * (r0_a_err / r0_a + r0_MeV_err / r0_MeV)  
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    sigma = a2sigma / (a**2)
    sigma_err = sigma * np.sqrt((a2sigma_err/a2sigma)**2 + (2*a_err/a)**2)
    return m_l, m_l_err, sigma, sigma_err

def model_func(m_l, sigma_0, c_l):
    log_arg = (c_l**2 * m_l**2) / (4 * np.pi * mu**2)
    log_arg = np.maximum(log_arg, 1e-15) # Protect against log(<=0)
    
    term1 = (c_l**2 / (2 * np.pi)) * m_l**2
    term2 = 1 + gamma_E - np.log(log_arg)
    
    return sigma_0 - term1 * term2

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
    m_l_list, m_l_err_list, sigma_list, sigma_err_list = [], [], [], []
    for _, row in df.iterrows():
        ml, ml_e, sig, sig_e = calculate_physics(row, prefix)
        m_l_list.append(ml)
        m_l_err_list.append(ml_e)
        sigma_list.append(sig)
        sigma_err_list.append(sig_e)
        
    m_l_arr, m_l_err_arr = np.array(m_l_list), np.array(m_l_err_list)
    sigma_arr, sigma_err_arr = np.array(sigma_list), np.array(sigma_err_list)

    # Initial fit using only y-errors
    try:
        popt_init, _ = curve_fit(
            model_func, m_l_arr, sigma_arr, 
            sigma=sigma_err_arr, absolute_sigma=True, maxfev=10000,
            p0=[np.mean(sigma_arr), 1.0]
        )
        c_l_guess = popt_init[1]
    except:
        c_l_guess = 1.0
        
    # Function to calculate effective y-errors propagating x-errors
    def effective_sigma_err(c_l_est):
        log_arg = (c_l_est**2 * m_l_arr**2) / (4 * np.pi * mu**2)
        log_arg = np.maximum(log_arg, 1e-15)
        
        # Analytical derivative: df/dx
        df_dx = - (c_l_est**2 / np.pi) * m_l_arr * (gamma_E - np.log(log_arg))
        return np.sqrt(sigma_err_arr**2 + (df_dx * m_l_err_arr)**2)

    # Final fit using effective errors
    try:
        popt, pcov = curve_fit(
            model_func, m_l_arr, sigma_arr, 
            sigma=effective_sigma_err(c_l_guess), absolute_sigma=True, maxfev=10000
        )
        sigma_0_opt, c_l_opt = popt[0], popt[1]
        sigma_0_err, c_l_err = np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]

        # Goodness of Fit & P-value
        residuals = sigma_arr - model_func(m_l_arr, sigma_0_opt, c_l_opt)
        chi_sq = np.sum((residuals / effective_sigma_err(c_l_opt))**2)
        dof = len(m_l_arr) - 2  # Global data points - 2 parameters
        red_chi_sq = chi_sq / dof if dof > 0 else np.nan
        p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

        results.append({
            'Data_Type': prefix,
            'sigma_0': sigma_0_opt,
            'sigma_0_err': sigma_0_err,
            'c_l': c_l_opt,
            'c_l_err': c_l_err,
            'chi_sq': chi_sq,
            'dof': dof,
            'red_chi_sq': red_chi_sq,
            'p_value': p_value
        })
    except RuntimeError:
        print(f"Fit failed for Data Type: {prefix}")

results_df = pd.DataFrame(results)
results_df.to_csv(filename, index=False)

print("\n--- FITTING COMPLETE ---")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters:")
print(results_df[['Data_Type', 'sigma_0', 'c_l', 'red_chi_sq', 'p_value']].to_string(index=False))