# Reads data_fran.csv,
# converts t0 to physical units (MeV^-1) and propagates errors,
# performs the fit to find σ_0 on the new data,
# and saves the results to fit_results_lineal_fran.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# --- Constants ---
hbar_c = 197.3269804 # MeV*fm
B0_MeV = 2700
sqrt_t0_fm = 0.1443

data_file = 'data_fran.csv'
filename = 'fit_results_lineal_fran.csv'


# We define both branches
branches = {
    'positive': {'frac': 0.124, 'frac_err': 0.005},
    'negative': {'frac': -0.124, 'frac_err': 0.005}
}

def model_func(m_l, sigma_0, lp_val):
    return sigma_0 - 4 * lp_val * m_l

# --- Load Data ---
df = pd.read_csv(data_file, sep=r'\s+')

# --- Calculate Physics ---
# Combine the statistical and systematic errors in quadrature: sqrt(0.0007^2 + 0.0013^2)
sqrt_t0_err_fm = np.sqrt(0.0007**2 + 0.0013**2) 

# Convert t0 to MeV^-1
sqrt_t0_MeV_inv = sqrt_t0_fm / hbar_c
sqrt_t0_err_MeV_inv = sqrt_t0_err_fm / hbar_c

# Calculate t0 in MeV^-2
t0_MeV_inv2 = sqrt_t0_MeV_inv**2

# mu_l is used directly as the x-axis mass (m_l)
m_l_arr = df['mu_l'].values

# Calculate sigma in physical units (MeV^2)
sigma_arr = df['sigma_t0'].values / t0_MeV_inv2

# Propagate errors for sigma
rel_err_sigma_t0 = df['statistical_error_of_sigma_t0'].values / df['sigma_t0'].values
rel_err_t0 = 2 * (sqrt_t0_err_fm / sqrt_t0_fm) # Power rule: relative error of x^2 is 2 * rel_err of x

sigma_err_arr = sigma_arr * np.sqrt(rel_err_sigma_t0**2 + rel_err_t0**2)

# --- Fitting ---
results = []

for branch_name, params in branches.items():
    lamb_prime = params['frac'] * 2 * B0_MeV
    lamb_prime_up = (params['frac'] + params['frac_err']) * 2 * B0_MeV
    lamb_prime_down = (params['frac'] - params['frac_err']) * 2 * B0_MeV

    # Central Fit
    popt_central, pcov_central = curve_fit(
        lambda x, s0: model_func(x, s0, lamb_prime), 
        m_l_arr, sigma_arr, sigma=sigma_err_arr, absolute_sigma=True
    )
    sigma_0_opt = popt_central[0]
    sigma_0_stat_err = np.sqrt(np.diag(pcov_central))[0]

    # Goodness of Fit Calculation
    residuals = sigma_arr - model_func(m_l_arr, sigma_0_opt, lamb_prime)
    chi_sq = np.sum((residuals / sigma_err_arr)**2)
    dof = len(m_l_arr) - 1  # 3 points - 1 fit parameter = 2 degrees of freedom
    red_chi_sq = chi_sq / dof
    p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

    # Upper Fit 
    popt_up, _ = curve_fit(
        lambda x, s0: model_func(x, s0, lamb_prime_up), 
        m_l_arr, sigma_arr, sigma=sigma_err_arr, absolute_sigma=True
    )
    
    # Lower Fit 
    popt_down, _ = curve_fit(
        lambda x, s0: model_func(x, s0, lamb_prime_down), 
        m_l_arr, sigma_arr, sigma=sigma_err_arr, absolute_sigma=True
    )

    # Calculate systematic and total error for sigma_0
    sigma_0_sys_err = max(abs(sigma_0_opt - popt_up[0]), abs(sigma_0_opt - popt_down[0]))
    sigma_0_total_err = np.sqrt(sigma_0_stat_err**2 + sigma_0_sys_err**2)

    results.append({
        'Branch': branch_name,
        'sigma_0_central': sigma_0_opt,
        'sigma_0_stat_err': sigma_0_stat_err,
        'sigma_0_sys_err_lamb': sigma_0_sys_err,
        'sigma_0_total_err': sigma_0_total_err,
        'chi_sq': chi_sq,             
        'red_chi_sq': red_chi_sq, 
        'p-value': p_value,    
        'sigma_0_up': popt_up[0],
        'sigma_0_down': popt_down[0],
        'lamb_prime_central': lamb_prime,
        'lamb_prime_up': lamb_prime_up,
        'lamb_prime_down': lamb_prime_down
    })

results_df = pd.DataFrame(results)
results_df.to_csv(filename, index=False)

print("\n--- FITTING COMPLETE ---")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters:")
print(results_df[['Branch', 'sigma_0_central', 'sigma_0_total_err', 'red_chi_sq', 'p-value']].to_string(index=False))