# Reads data_bram.csv and fits with the dimensionless ST logarithmic model,
# computes dimensionless variables directly from lattice data,
# performs the fit to find y_0 and c_l for each ensemble,
# calculates effective errors and p-values,
# and saves the results to fit_results_ST_dim.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2


# =============================================================================
# 1. Constants & Scale Conversions
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_MeV = r0_fm / hbar_c

mu = np.sqrt(210)  # MeV sqrt(sigma) from paper Soto-Tarrús
mu_dim = mu * r0_MeV # Dimensionless scale factor
gamma_E = np.euler_gamma

A_types = ['Ar0', 'Api12']
A_index = 1
LQCD = True  # Posa-ho a False per usar el model original, o True pel nou model (amb c2_l).

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
# Selecció dinàmica del nom de fitxer segons la variable LQCD
lqcd_str = "_full" if LQCD else ""
filename = f'fit_results_ST{lqcd_str}_{A_types[A_index]}.csv'

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

def model_func_dim(x, y_0, c_l):
    log_arg = (c_l**2 * x**2) / (4 * np.pi * mu_dim**2)
    log_arg = np.maximum(log_arg, 1e-15) 
    
    term1 = (c_l**2 / (2 * np.pi)) * x**2
    term2 = 1 + gamma_E - np.log(log_arg)
    
    return y_0 - term1 * term2

def model_func2_dim(x, y_0, c_l, c2_l):
    z = c_l * x + c2_l
    log_arg = (z**2) / (4 * np.pi * mu_dim**2)
    log_arg = np.maximum(log_arg, 1e-15) 
    
    term1 = (1 / (2 * np.pi)) * z**2
    term2 = 1 + gamma_E - np.log(log_arg)
    
    return y_0 - term1 * term2

# =============================================================================
# 3. Load & Group Data
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: "_".join(s.split('_')[-2:])) 
ensembles = df['mass_group'].unique()

results = []
data_types = ['bare', 'smeared']

# =============================================================================
# 4. Fitting Routine 
# =============================================================================
for group in ensembles:
    subset = df[df['mass_group'] == group]
    
    for prefix in data_types:
        x_list, x_err_list, y_list, y_err_list = [], [], [], []
        for _, row in subset.iterrows():
            x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
            x_list.append(x_val)
            x_err_list.append(x_e)
            y_list.append(y_val)
            y_err_list.append(y_e)
            
        x_arr, x_err_arr = np.array(x_list), np.array(x_err_list)
        y_arr, y_err_arr = np.array(y_list), np.array(y_err_list)
        
        # Estructura base per guardar els resultats
        res_dict = {
            'Ensemble': group,
            'Data_Type': prefix
        }

        if not LQCD:
            # -----------------------------------------------------------------
            # FIT: LQCD = False (Model Original)
            # -----------------------------------------------------------------
            try:
                popt_init, _ = curve_fit(
                    model_func_dim, x_arr, y_arr, 
                    sigma=y_err_arr, absolute_sigma=True, maxfev=10000,
                    p0=[np.mean(y_arr), 1.0]
                )
                c_l_guess = popt_init[1]
            except:
                c_l_guess = 1.0 
                
            def effective_y_err(c_l_est):
                log_arg = (c_l_est**2 * x_arr**2) / (4 * np.pi * mu_dim**2)
                log_arg = np.maximum(log_arg, 1e-15) 
                df_dx = - (c_l_est**2 / np.pi) * x_arr * (gamma_E - np.log(log_arg))
                return np.sqrt(y_err_arr**2 + (df_dx * x_err_arr)**2)

            try:
                popt, pcov = curve_fit(
                    model_func_dim, x_arr, y_arr, 
                    sigma=effective_y_err(c_l_guess), absolute_sigma=True,
                    maxfev=10000
                )
                
                res_dict['y_0'] = popt[0]
                res_dict['y_0_err'] = np.sqrt(np.diag(pcov))[0]
                res_dict['c_l'] = popt[1]
                res_dict['c_l_err'] = np.sqrt(np.diag(pcov))[1]

                residuals = y_arr - model_func_dim(x_arr, popt[0], popt[1])
                chi_sq = np.sum((residuals / effective_y_err(popt[1]))**2)
                dof = len(x_arr) - 2  
                
            except RuntimeError:
                print(f"Fit failed for {group} - {prefix} (Model 1)")
                continue

        else:
            # -----------------------------------------------------------------
            # FIT: LQCD = True (Nou Model amb c2_l)
            # -----------------------------------------------------------------
            try:
                popt_init, _ = curve_fit(
                    model_func2_dim, x_arr, y_arr, 
                    sigma=y_err_arr, absolute_sigma=True, maxfev=10000,
                    p0=[np.mean(y_arr), 1.0, 0.0]
                )
                c_l_guess, c2_l_guess = popt_init[1], popt_init[2]
            except:
                c_l_guess, c2_l_guess = 1.0, 0.0
                
            def effective_y_err2(c_l_est, c2_l_est):
                z = c_l_est * x_arr + c2_l_est
                log_arg = (z**2) / (4 * np.pi * mu_dim**2)
                log_arg = np.maximum(log_arg, 1e-15) 
                
                df_dx = - (z * c_l_est / np.pi) * (gamma_E - np.log(log_arg))
                return np.sqrt(y_err_arr**2 + (df_dx * x_err_arr)**2)

            try:
                popt, pcov = curve_fit(
                    model_func2_dim, x_arr, y_arr, 
                    sigma=effective_y_err2(c_l_guess, c2_l_guess), absolute_sigma=True,
                    maxfev=10000
                )
                
                res_dict['y_0'] = popt[0]
                res_dict['y_0_err'] = np.sqrt(np.diag(pcov))[0]
                res_dict['c_l'] = popt[1]
                res_dict['c_l_err'] = np.sqrt(np.diag(pcov))[1]
                res_dict['c2_l'] = popt[2]
                res_dict['c2_l_err'] = np.sqrt(np.diag(pcov))[2]

                residuals = y_arr - model_func2_dim(x_arr, popt[0], popt[1], popt[2])
                chi_sq = np.sum((residuals / effective_y_err2(popt[1], popt[2]))**2)
                dof = len(x_arr) - 3  
                
            except RuntimeError:
                print(f"Fit failed for {group} - {prefix} (Model 2)")
                continue

        # Càlculs estadístics comuns
        res_dict['chi_sq'] = chi_sq
        res_dict['dof'] = dof
        res_dict['red_chi_sq'] = chi_sq / dof if dof > 0 else np.nan
        res_dict['p_value'] = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
        
        results.append(res_dict)

results_df = pd.DataFrame(results).sort_values(by=['Ensemble', 'Data_Type'])
results_df.to_csv(filename, index=False)

print("\n--- ENSEMBLE DIMENSIONLESS ST FIT COMPLETE ---")
print("Model used:", "LQCD (c2_l active)" if LQCD else "Original (c2_l = 0)")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters:")
# Dinàmic: només mostra la columna c2_l si existeix
cols_to_print = ['Ensemble', 'Data_Type', 'y_0', 'c_l']
if LQCD:
    cols_to_print.append('c2_l')
cols_to_print.extend(['red_chi_sq', 'p_value'])

print(results_df[cols_to_print].to_string(index=False))