# In this fit I used the real value of \sigma and m_l.q to obtain the y0 of the ST equation as a function of cl and gamma
# by doing this I have reduced the fitted parameters and hopefully increased the accuracy of the fit.

# Reads data_bram.csv and fits with the global dimensionless ST model,
# computes dimensionless variables directly from lattice data,
# performs the global fit to find a universal c_l and gamma (and c2_l if LQCD=True),
# calculates effective errors and p-values,
# and saves the results to fit_results_theo_ST.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import json

# =============================================================================
# 1. Constants & Scale Conversions
# =============================================================================
# Fix seed for reproducibility in Monte Carlo error propagation
np.random.seed(42)

hbar_c = 0.1973269804 # GeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_GeV_central = r0_fm / hbar_c
r0_GeV_err = r0_fm_err / hbar_c

# Physical point constraints
m2_lq_phys = 0.051        # GeV^2
sigma_phys = 0.21         # GeV^2

A_types = ['Ar0', 'Api12']
A_index = 0
LQCD = True

# Define lower and upper limits (y_0 is no longer a fit parameter)
# Bounds for Model 1: [c_l, gamma]
lower_bounds1 = [-np.inf, -np.inf]
upper_bounds1 = [ np.inf,  np.inf]
bounds1 = (lower_bounds1, upper_bounds1)

# Bounds for Model 2: [c_l, c2_l, gamma]
lower_bounds2 = [-np.inf, 200.0, -np.inf]
upper_bounds2 = [ np.inf,  600.0, np.inf]
bounds2 = (lower_bounds2, upper_bounds2)


data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
# Dynamic filename selection
lqcd_str = "_full" if LQCD else ""
filename = f'fit_results_theo{lqcd_str}_ST_{A_types[A_index]}.csv'

# =============================================================================
# 2. Constraint Functions to compute y_0
# =============================================================================
def compute_y0(c_l, gamma, r0_GeV):
    """
    Computes y0 dynamically by enforcing y(x_phys) = y_phys for Model 1
    m_lq^2 = 0.051 GeV^2 and sigma = 0.21 GeV^2.
    """
    x2_phys = (m2_lq_phys / c_l**2) * (r0_GeV**2)
    y_phys = sigma_phys * (r0_GeV**2)
    
    log_arg = np.maximum(x2_phys, 1e-15)  # Prevent log(0)
    term1 = (c_l**2 / (2.0 * np.pi)) * x2_phys * np.log(log_arg)
    term2 = gamma * x2_phys
    
    return y_phys - term1 - term2

def compute_y02(c_l, c2_l, gamma, r0_GeV):
    """
    Computes y0 dynamically by enforcing y(x_phys) = y_phys for Model 2 (LQCD)
    m_lq^2 = 0.051 GeV^2 and sigma = 0.21 GeV^2.
    """
    m_lq = np.sqrt(m2_lq_phys)
    x_phys = (m_lq / c_l) * r0_GeV
    z_phys = x_phys + (c2_l / c_l)
    z2_phys = z_phys**2
    y_phys = sigma_phys * (r0_GeV**2)
    
    log_arg = np.maximum(z2_phys, 1e-15)
    term1 = (c_l**2 / (2.0 * np.pi)) * z2_phys * np.log(log_arg)
    term2 = gamma * z2_phys
    
    return y_phys - term1 - term2

# =============================================================================
# 3. Data Processing & Dimensionless Models
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

def model_func_dim(x, c_l, gamma):
    y_0 = compute_y0(c_l, gamma, r0_GeV_central)
    term1 = (c_l**2 / (2.0 * np.pi)) * x**2 * np.log(np.maximum(x**2, 1e-15))
    term2 = gamma * x**2
    return y_0 + term1 + term2

def model_func2_dim(x, c_l, c2_l, gamma):
    y_0 = compute_y02(c_l, c2_l, gamma, r0_GeV_central)
    z = x + c2_l/c_l
    log_arg = np.maximum(z**2, 1e-15)
    term1 = (c_l**2 / (2.0 * np.pi)) * z**2 * np.log(log_arg)
    term2 = gamma * z**2
    return y_0 + term1 + term2

# =============================================================================
# 4. Load Data & Initialize
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')
results = []
data_types = ['bare', 'smeared']
mc_size = 1000  # Number of Monte Carlo samples for error propagation

# =============================================================================
# 5. Fitting Routine (Global)
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
        # GLOBAL FIT: LQCD = False (Original Model)
        # -----------------------------------------------------------------
        try:
            popt_init, _ = curve_fit(
                model_func_dim, x_arr, y_arr, 
                sigma=y_err_arr, absolute_sigma=True, maxfev=10000,
                p0=[1.0, 1.0], bounds=bounds1
            )
            c_l_guess, gamma_guess = popt_init[0], popt_init[1]
        except:
            c_l_guess, gamma_guess = 1.0, 1.0

        def effective_y_err(c_l_est, gamma_est):
            log_arg = np.maximum(x_arr**2, 1e-15)
            df_dx = 2 * gamma_est * x_arr + (c_l_est**2 / np.pi) * x_arr * (1 + np.log(log_arg))
            return np.sqrt(y_err_arr**2 + (df_dx * x_err_arr)**2)
        
        try:
            popt, pcov = curve_fit(
                model_func_dim, x_arr, y_arr, 
                sigma=effective_y_err(c_l_guess, gamma_guess), 
                absolute_sigma=True, maxfev=10000, bounds=bounds1
            )

            # Extract Parameters
            res_dict['c_l'] = popt[0]
            res_dict['c_l_err'] = np.sqrt(np.diag(pcov))[0]
            res_dict['gamma'] = popt[1]
            res_dict['gamma_err'] = np.sqrt(np.diag(pcov))[1]
            
            # --- Monte Carlo Error Propagation for y_0 ---
            r0_samples = np.random.normal(r0_GeV_central, r0_GeV_err, mc_size)
            param_samples = np.random.multivariate_normal(popt, pcov, mc_size, check_valid='warn')
            y0_samples = np.array([compute_y0(p[0], p[1], r) for p, r in zip(param_samples, r0_samples)])
            
            res_dict['y_0'] = compute_y0(popt[0], popt[1], r0_GeV_central)
            res_dict['y_0_err'] = np.std(y0_samples)
            # ---------------------------------------------

            res_dict['pcov'] = json.dumps(pcov.tolist())
            res_dict['x_min'] = float(np.min(x_arr))
            res_dict['x_max'] = float(np.max(x_arr))

            residuals = y_arr - model_func_dim(x_arr, *popt)
            chi_sq = np.sum((residuals / effective_y_err(popt[0], popt[1]))**2)
            dof = len(x_arr) - len(popt)

        except RuntimeError:
            print(f"Fit failed for Data Type: {prefix} (Model 1)")
            continue

    else:
        # -----------------------------------------------------------------
        # GLOBAL FIT: LQCD = True (New Model with c2_l)
        # -----------------------------------------------------------------
        try:
            popt_init, _ = curve_fit(
                model_func2_dim, x_arr, y_arr, 
                sigma=y_err_arr, absolute_sigma=True, maxfev=10000,
                p0=[1.0, 400.0, 1.0], bounds=bounds2
            )
            c_l_guess, c2_l_guess, gamma_guess = popt_init[0], popt_init[1], popt_init[2]
        except:
            c_l_guess, c2_l_guess, gamma_guess = 1.0, 400.0, 1.0
            
        def effective_y_err2(c_l_est, c2_l_est, gamma_est):
            z = x_arr + c2_l_est/c_l_est
            log_arg = np.maximum(z**2, 1e-15)
                
            df_dx = z * (2 * gamma_est + (c_l_est**2 / np.pi) * (1 + np.log(log_arg)))
            return np.sqrt(y_err_arr**2 + (df_dx * x_err_arr)**2)

        try:
            popt, pcov = curve_fit(
                model_func2_dim, x_arr, y_arr, 
                sigma=effective_y_err2(c_l_guess, c2_l_guess, gamma_guess), 
                absolute_sigma=True, maxfev=10000, bounds=bounds2
            )
            
            # Extract Parameters
            res_dict['c_l'] = popt[0]
            res_dict['c_l_err'] = np.sqrt(np.diag(pcov))[0]
            res_dict['c2_l'] = popt[1]
            res_dict['c2_l_err'] = np.sqrt(np.diag(pcov))[1]
            res_dict['gamma'] = popt[2]
            res_dict['gamma_err'] = np.sqrt(np.diag(pcov))[2]
            
            # --- Monte Carlo Error Propagation for y_0 ---
            r0_samples = np.random.normal(r0_GeV_central, r0_GeV_err, mc_size)
            param_samples = np.random.multivariate_normal(popt, pcov, mc_size, check_valid='warn')
            y0_samples = np.array([compute_y02(p[0], p[1], p[2], r) for p, r in zip(param_samples, r0_samples)])
            
            res_dict['y_0'] = compute_y02(popt[0], popt[1], popt[2], r0_GeV_central)
            res_dict['y_0_err'] = np.std(y0_samples)
            # ---------------------------------------------

            res_dict['pcov'] = json.dumps(pcov.tolist())
            res_dict['x_min'] = float(np.min(x_arr))
            res_dict['x_max'] = float(np.max(x_arr))

            residuals = y_arr - model_func2_dim(x_arr, *popt)
            chi_sq = np.sum((residuals / effective_y_err2(popt[0], popt[1], popt[2]))**2)
            dof = len(x_arr) - len(popt)

        except RuntimeError:
            print(f"Fit failed for Data Type: {prefix} (Model 2)")
            continue

    # Shared Statistical Metrics
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
print("\nSummary of Extracted Parameters (y_0 theoretically fixed):")

cols_to_print = ['Data_Type', 'y_0', 'c_l']
if LQCD:
    cols_to_print.append('c2_l')
cols_to_print.extend(['gamma', 'red_chi_sq', 'p_value'])

print(results_df[cols_to_print].to_string(index=False))