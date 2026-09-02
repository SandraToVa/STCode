# Reads data_fran.csv,
# converts t0 to physical units (MeV^-1) and properly isolates stat/sys errors,
# performs the logarithmic fit to find σ_0,
# calculates systematic errors from λ', η, frac, and the t0 scale,
# and saves the results to fit_results_log_fran.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# =============================================================================
# 1. Constants & Unit Conversions
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 2000 # 2 GeV in MeV from PDG

data_file = 'data_fran.csv'
filename = 'fit_results_log_fran.csv'

# t0 parameters
sqrt_t0_fm = 0.1443
sqrt_t0_err_fm = np.sqrt(0.0007**2 + 0.0013**2) 

# Convert central t0 to MeV^-2
sqrt_t0_MeV_inv = sqrt_t0_fm / hbar_c
t0_MeV_inv2 = sqrt_t0_MeV_inv**2

# =============================================================================
# 2. Parameter Sets (Covariance matrices for [lamb, eta, frac])
# =============================================================================
param_sets = {
    'Branch_1': {
        'means': [0.00978833, 0.0262025, -0.124474],
        'cov': [[ 0.0000352816, -0.0000458309,  0.0000182559],
                [-0.0000458309,  0.0000615173, -0.0000286899],
                [ 0.0000182559, -0.0000286899,  0.0000271675]]
    },
    'Branch_2': {
        'means': [-0.0458536, 0.0815424, -0.124023],
        'cov': [[ 0.0000190469,  -0.0000134491,  -8.07208e-6],
                [-0.0000134491,   0.0000131926,  -2.67543e-6],
                [-8.07208e-6,    -2.67543e-6,     0.0000276483]]
    },
    'Branch_3': {
        'means': [0.0458536, -0.0815424, 0.124023],
        'cov': [[ 0.0000190469,  -0.0000134491,  -8.07208e-6],
                [-0.0000134491,   0.0000131926,  -2.67543e-6],
                [-8.07208e-6,    -2.67543e-6,     0.0000276483]]
    },
    'Branch_4': {
        'means': [-0.00978833, -0.0262025, 0.124474],
        'cov': [[ 0.0000352816, -0.0000458309,  0.0000182559],
                [-0.0000458309,  0.0000615173, -0.0000286899],
                [ 0.0000182559, -0.0000286899,  0.0000271675]]
    }
}

# =============================================================================
# 3. Chiral Model
# =============================================================================
def chiral_model(ml, sigma_0, lamb, eta, frac):
    c_pipi = (lamb + eta/2) / 2
    lamb_prime = 2 * B0_MeV * frac
    
    term1 = 4 * lamb_prime * ml
    
    log_arg = (2 * B0_MeV * ml) / (mu_MeV**2)
    log_arg = np.where(log_arg > 0, log_arg, 1e-10) 
    
    term2_coeff = (3 * B0_MeV) / (4 * np.pi**2 * f_pi_MeV**2)
    term2 = term2_coeff * (4 * B0_MeV * c_pipi - lamb_prime) * ( np.log(log_arg) - 1 )
    term2 = term2 * (ml**2)

    return sigma_0 - term1 - term2

# =============================================================================
# 4. Load Data
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')

# m_l is exact
m_l_data = df['mu_l'].values

# Central sigma in physical units
sigma_data = df['sigma_t0'].values / t0_MeV_inv2

# Central errors (used for chi_sq calculation only)
rel_err_sigma_t0 = df['statistical_error_of_sigma_t0'].values / df['sigma_t0'].values
rel_err_t0 = 2 * (sqrt_t0_err_fm / sqrt_t0_fm) 
sigma_err = sigma_data * np.sqrt(rel_err_sigma_t0**2 + rel_err_t0**2)

# =============================================================================
# 5. Fitting Routine (Monte Carlo Loop)
# =============================================================================
fit_results = []
N_boot = 1000 

for branch_name, params in param_sets.items():
    means = params['means']
    cov = params['cov']
    
    # Central fit
    try:
        popt_central, _ = curve_fit(lambda x, s0: chiral_model(x, s0, *means), m_l_data, sigma_data)
        s0_central = popt_central[0]

        # GOODNESS OF FIT
        y_fit = chiral_model(m_l_data, s0_central, *means)
        residuals = sigma_data - y_fit
        chi_sq = np.sum((residuals / sigma_err)**2)
        dof = len(m_l_data) - 1
        red_chi_sq = chi_sq / dof
    except:
        continue
    
    # Derived constants central
    c_pipi_central = (means[0] + means[1]/2) / 2
    lamb_prime_central = 2 * B0_MeV * means[2]
    
    # Bootstrap distributions
    s0_stat_list, s0_sys_list, s0_tot_list = [], [], []
    c_pipi_list, lamb_prime_list = [], []
    
    for _ in range(N_boot):
        # 1. Smear Statistical Variables (Data points)
        sigma_t0_s = np.random.normal(df['sigma_t0'].values, df['statistical_error_of_sigma_t0'].values)
        
        # 2. Smear Systematic Variables (Global scale & Chiral params)
        sqrt_t0_s = np.random.normal(sqrt_t0_fm, sqrt_t0_err_fm)
        t0_MeV_inv2_s = (sqrt_t0_s / hbar_c)**2
        l_s, e_s, f_s = np.random.multivariate_normal(means, cov)

        # 3. Construct Data Configurations
        sig_stat_only = sigma_t0_s / t0_MeV_inv2      # Scale fixed to central, data smeared
        sig_sys_only  = df['sigma_t0'].values / t0_MeV_inv2_s # Data fixed to central, scale smeared
        sig_tot       = sigma_t0_s / t0_MeV_inv2_s    # Both smeared
        
        # Store derived constant distributions
        c_pipi_list.append((l_s + e_s/2) / 2)
        lamb_prime_list.append(2 * B0_MeV * f_s)
        
        try:
            # Stat Error only (fixed chiral params, varying data stat)
            popt_stat, _ = curve_fit(lambda x, s0: chiral_model(x, s0, *means), m_l_data, sig_stat_only)
            s0_stat_list.append(popt_stat[0])
            
            # Sys Error only (fixed data stat, varying scale and chiral params)
            popt_sys, _ = curve_fit(lambda x, s0: chiral_model(x, s0, l_s, e_s, f_s), m_l_data, sig_sys_only)
            s0_sys_list.append(popt_sys[0])
            
            # Total Error (varying both)
            popt_tot, _ = curve_fit(lambda x, s0: chiral_model(x, s0, l_s, e_s, f_s), m_l_data, sig_tot)
            s0_tot_list.append(popt_tot[0])
        except:
            continue

    if not s0_tot_list:
        continue
    
    # Calculate standard deviations & bounds
    s0_stat_err = np.std(s0_stat_list)
    s0_sys_err = np.std(s0_sys_list)
    s0_tot_err = np.std(s0_tot_list)
    
    c_pipi_err = np.std(c_pipi_list)
    lamb_prime_err = np.std(lamb_prime_list)
    
    fit_results.append({
        'Branch': branch_name,
        'sigma_0_central': s0_central,
        'chi_sq': chi_sq,             
        'red_chi_sq': red_chi_sq,
        'sigma_0_stat_err': s0_stat_err,
        'sigma_0_sys_err': s0_sys_err,
        'sigma_0_total_err': s0_tot_err,
        'sigma_0_up': s0_central + s0_tot_err,
        'sigma_0_down': s0_central - s0_tot_err,
        'lamb_prime_central': lamb_prime_central,
        'lamb_prime_up': lamb_prime_central + lamb_prime_err,
        'lamb_prime_down': lamb_prime_central - lamb_prime_err,
        'c_pipi_central': c_pipi_central,
        'c_pipi_up': c_pipi_central + c_pipi_err,
        'c_pipi_down': c_pipi_central - c_pipi_err
    })

# Save to CSV
results_df = pd.DataFrame(fit_results)
results_df.to_csv(filename, index=False)

print("\n--- FITTING COMPLETE ---")
print(f"Results successfully saved to '{filename}'.")
print("\nSummary of Extracted Parameters:")
print(results_df[['Branch', 'sigma_0_central', 'sigma_0_stat_err', 'sigma_0_sys_err', 'sigma_0_total_err', 'red_chi_sq']].to_string(index=False))