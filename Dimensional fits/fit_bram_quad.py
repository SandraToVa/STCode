# Reads data_bram.csv,
# converts lattice units to physical units (propagating errors),
# performs the fit to find σ_0 and epsilon for each ensemble,
# calculates systematic errors,
# and saves the results to fit_results_log.csv.

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
nf = 3  # Number of flavors

A_types = ['Ar0', 'Api12']
A_index = 0 # Change index to select r0 or pi12 data

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_quad_{A_types[A_index]}.csv'

# =============================================================================
# 2. Data Processing Function & Quadratic Model
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

def model_func(m_l, sigma_0, epsilon):
    return sigma_0 - (nf / (2 * np.pi * epsilon)) * m_l**2

# =============================================================================
# 3. Load & Group Data
# =============================================================================

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda x: "_".join(x.split('_')[-2:])) 
ensembles = df['mass_group'].unique()

results = []
data_types = ['bare', 'smeared']

# =============================================================================
# 4. Fitting Routine 
# =============================================================================

for group in ensembles:
    subset = df[df['mass_group'] == group]
    
    for prefix in data_types:
        m_l_list, m_l_err_list, sigma_list, sigma_err_list = [], [], [], []
        for _, row in subset.iterrows():
            ml, ml_e, sig, sig_e = calculate_physics(row, prefix)
            m_l_list.append(ml)
            m_l_err_list.append(ml_e)
            sigma_list.append(sig)
            sigma_err_list.append(sig_e)
            
        m_l_arr, m_l_err_arr = np.array(m_l_list), np.array(m_l_err_list)
        sigma_arr, sigma_err_arr = np.array(sigma_list), np.array(sigma_err_list)

        # Step 1: Initial fit using only sigma_err
        try:
            popt_init, _ = curve_fit(model_func, m_l_arr, sigma_arr, sigma=sigma_err_arr, absolute_sigma=True, maxfev=10000)
            epsilon_guess = popt_init[1]
        except:
            epsilon_guess = 1.0 # Fallback guess
            
        # Step 2: Final fit using effective errors propagating x-errors
        def effective_sigma_err(eps):
            # Error propagation for x^2: d(f)/dx = - (nf / (pi * eps)) * x
            df_dx = (nf / (np.pi * eps)) * m_l_arr
            return np.sqrt(sigma_err_arr**2 + (df_dx * m_l_err_arr)**2)

        try:
            popt, pcov = curve_fit(
                model_func, m_l_arr, sigma_arr, 
                sigma=effective_sigma_err(epsilon_guess), absolute_sigma=True,
                maxfev=10000
            )
            sigma_0_opt, epsilon_opt = popt[0], popt[1]
            sigma_0_err, epsilon_err = np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]

            # Goodness of Fit
            residuals = sigma_arr - model_func(m_l_arr, sigma_0_opt, epsilon_opt)
            chi_sq = np.sum((residuals / effective_sigma_err(epsilon_opt))**2)
            dof = len(m_l_arr) - 2  # N points - 2 fit parameters
            red_chi_sq = chi_sq / dof if dof > 0 else np.nan
            p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

            results.append({
                'Ensemble': group,
                'Data_Type': prefix,
                'sigma_0': sigma_0_opt,
                'sigma_0_err': sigma_0_err,
                'epsilon': epsilon_opt,
                'epsilon_err': epsilon_err,
                'chi_sq': chi_sq,
                'dof': dof,
                'red_chi_sq': red_chi_sq,
                'p_value': p_value
            })
        except RuntimeError:
            print(f"Fit failed for {group} - {prefix}")

results_df = pd.DataFrame(results).sort_values(by=['Ensemble', 'Data_Type'])
results_df.to_csv(filename, index=False)

print("\n--- FITTING COMPLETE ---")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters:")
print(results_df[['Ensemble', 'Data_Type', 'sigma_0', 'epsilon', 'red_chi_sq']].to_string(index=False))
