# Does the same as fit_bram_lineal.py but instead of separate in ensambles and do the fit for each ensamble it does a global fit

# Performs a strictly global linear fit across all ensembles 
# to find a single, universal σ_0.
# Calculates systematic errors from λ' and saves to CSV.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# --- Constants ---
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700

A_types = ['Ar0', 'Api12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_lineal_global_{A_types[A_index]}.csv'


# We define both branches
branches = {
    'positive': {'frac': 0.124, 'frac_err': 0.005},
    'negative': {'frac': -0.124, 'frac_err': 0.005}
}

def calculate_physics(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    
    a = r0_MeV / r0_a
    a_err = a * (r0_a_err / r0_a + r0_MeV_err / r0_MeV)  # Combine relative errors of r0_a and r0_MeV
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    sigma = a2sigma / (a**2)
    sigma_err = sigma * np.sqrt((a2sigma_err/a2sigma)**2 + (2*a_err/a)**2)
    return m_l, m_l_err, sigma, sigma_err

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda x: 'M_iii' if 'M_iii' in x else ('M_ii' if 'M_ii' in x else 'M_i'))

results = []
data_types = ['bare', 'smeared']

for prefix in data_types:
    # Gather ALL data from all ensembles
    m_l_list, m_l_err_list, sigma_list, sigma_err_list = [], [], [], []
    for _, row in df.iterrows():
        ml, ml_e, sig, sig_e = calculate_physics(row, prefix)
        m_l_list.append(ml)
        m_l_err_list.append(ml_e)
        sigma_list.append(sig)
        sigma_err_list.append(sig_e)
        
    m_l_arr, m_l_err_arr = np.array(m_l_list), np.array(m_l_err_list)
    sigma_arr, sigma_err_arr = np.array(sigma_list), np.array(sigma_err_list)

    def effective_sigma_err(lp_val):
        # The square handles negative lamb_prime naturally
        return np.sqrt(sigma_err_arr**2 + (4 * lp_val * m_l_err_arr)**2)

    def model_func(m_l, sigma_0, lp_val):
        return sigma_0 - 4 * lp_val * m_l

    # Loop over the branches
    for branch_name, params in branches.items():
        lamb_prime = params['frac'] * 2 * B0_MeV
        lamb_prime_up = (params['frac'] + params['frac_err']) * 2 * B0_MeV
        lamb_prime_down = (params['frac'] - params['frac_err']) * 2 * B0_MeV

        # Central Fit
        popt_central, pcov_central = curve_fit(
            lambda x, s0: model_func(x, s0, lamb_prime), 
            m_l_arr, sigma_arr, sigma=effective_sigma_err(lamb_prime), absolute_sigma=True
        )
        sigma_0_opt = popt_central[0]
        sigma_0_stat_err = np.sqrt(np.diag(pcov_central))[0]

        # GOODNESS OF FIT CALCULATION
        residuals = sigma_arr - model_func(m_l_arr, sigma_0_opt, lamb_prime)
        chi_sq = np.sum((residuals / effective_sigma_err(lamb_prime))**2)
        dof = len(m_l_arr) - 1  # 1 global fit parameter: sigma_0
        red_chi_sq = chi_sq / dof
        p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

        # Upper Fit 
        popt_up, _ = curve_fit(
            lambda x, s0: model_func(x, s0, lamb_prime_up), 
            m_l_arr, sigma_arr, sigma=effective_sigma_err(lamb_prime_up), absolute_sigma=True
        )
        
        # Lower Fit 
        popt_down, _ = curve_fit(
            lambda x, s0: model_func(x, s0, lamb_prime_down), 
            m_l_arr, sigma_arr, sigma=effective_sigma_err(lamb_prime_down), absolute_sigma=True
        )

        sigma_0_sys_err = max(abs(sigma_0_opt - popt_up[0]), abs(sigma_0_opt - popt_down[0]))
        sigma_0_total_err = np.sqrt(sigma_0_stat_err**2 + sigma_0_sys_err**2)

        results.append({
            'Data_Type': prefix,
            'Branch': branch_name,
            'sigma_0_central': sigma_0_opt,
            'sigma_0_stat_err': sigma_0_stat_err,
            'sigma_0_sys_err_lamb': sigma_0_sys_err,
            'sigma_0_total_err': sigma_0_total_err,
            'chi_sq': chi_sq,  
            'dof': dof,           
            'red_chi_sq': red_chi_sq,     
            'p_value': p_value,
            'sigma_0_up': popt_up[0],
            'sigma_0_down': popt_down[0],
            'lamb_prime_central': lamb_prime, 
            'lamb_prime_up': lamb_prime_up,
            'lamb_prime_down': lamb_prime_down
        })

results_df = pd.DataFrame(results).sort_values(by=['Data_Type', 'Branch'])
results_df.to_csv(filename, index=False)

print("\n--- GLOBAL LINEAR FITTING COMPLETE ---")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters:")
print(results_df[['Data_Type', 'Branch', 'sigma_0_central', 'sigma_0_total_err', 'chi_sq', 'dof', 'red_chi_sq', 'p_value']].to_string(index=False))