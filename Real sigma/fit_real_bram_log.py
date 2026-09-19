# In this fit I used the real value of \sigma and m_\pi to obtain the y0 of the log equation as a function of \lambda''
# by doing this I have reduced the fitted parameters and hopefully increased the accuracy of the fit.

# Performs a strictly global linear-log fit across all data points
# to find a single, universal dimensionless curvature L'' with y0 fixed by
# the physical point (sigma = 0.187 GeV^2 at m_pi = 140 MeV).
# Uses Monte Carlo bootstrap to propagate correlated parameter errors and r0 errors,
# and saves to CSV.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
from scipy.stats import chi2

# =============================================================================
# 1. Constants & Physical Reference Values
# =============================================================================
# Fix seed for reproducibility
np.random.seed(42)
fit_results = []
N_boot = 1000 

hbar_c = 197.3269804 
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV_central = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c

B0_MeV = 2700.0
f_pi_MeV = 92.0
mu_MeV = 200.0 

# Physical point constraint
m_pi_phys = 140.0        # MeV
sigma_phys = 0.187 * 1e6 # MeV^2 (0.187 GeV^2)

A_types = ['Ar0', 'Api12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_theo_log_{A_types[A_index]}.csv'

param_sets = {
    'Branch_1': {'means': [0.00978833, 0.0262025, -0.124474], 'cov': [[0.0000352816, -0.0000458309, 0.0000182559], [-0.0000458309, 0.0000615173, -0.0000286899], [0.0000182559, -0.0000286899, 0.0000271675]]},
    'Branch_2': {'means': [-0.0458536, 0.0815424, -0.124023], 'cov': [[0.0000190469, -0.0000134491, -8.07208e-6], [-0.0000134491, 0.0000131926, -2.67543e-6], [-8.07208e-6, -2.67543e-6, 0.0000276483]]},
    'Branch_3': {'means': [0.0458536, -0.0815424, 0.124023], 'cov': [[0.0000190469, -0.0000134491, -8.07208e-6], [-0.0000134491, 0.0000131926, -2.67543e-6], [-8.07208e-6, -2.67543e-6, 0.0000276483]]},
    'Branch_4': {'means': [-0.00978833, -0.0262025, 0.124474], 'cov': [[0.0000352816, -0.0000458309, 0.0000182559], [-0.0000458309, 0.0000615173, -0.0000286899], [0.0000182559, -0.0000286899, 0.0000271675]]}
}

# =============================================================================
# 2. Data Processing & Constrained Global Model
# =============================================================================
def calculate_dimensionless(df_row, prefix):
    """
    Computes dimensionless variables x and y and propagates their errors
    directly from the lattice data.
    """
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    
    # y = (a2sigma) * (r0/a)^2
    y = a2sigma * (r0_a ** 2)
    y_err = y * np.sqrt((a2sigma_err / a2sigma)**2 + (2 * r0_a_err / r0_a)**2)
    
    # x = (am_l) * (r0/a)
    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)
    
    return x, x_err, y, y_err

def compute_y0(L2, L_prime, c_pipi, B0_dim, f_pi_dim, mu_dim, r0_MeV):
    """
    Computes y0 dynamically by enforcing y(x_phys) = y_phys for m_pi = 140 MeV and sigma = 0.187 GeV^2.
    """
    B0_val = B0_dim / r0_MeV
    x_phys = (m_pi_phys**2 / (2.0 * B0_val)) * r0_MeV
    y_phys = sigma_phys * (r0_MeV**2)
    
    term1_phys = 4.0 * L_prime * x_phys
    log_arg_phys = (2.0 * B0_dim * x_phys) / (mu_dim**2)
    term2_coeff = (3.0 * B0_dim) / (4.0 * np.pi**2 * f_pi_dim**2)
    term2_phys = term2_coeff * (4.0 * B0_dim * c_pipi - L_prime) * (np.log(log_arg_phys) - 1.0) * (x_phys**2)
    term3_phys = 2.0 * L2 * (x_phys**2)
    
    return y_phys + term1_phys + term2_phys + term3_phys

def chiral_model_universal_dim(x, L2, L_prime, c_pipi, B0_dim, f_pi_dim, mu_dim, r0_MeV):
    """
    Evaluates y(x) with y0 dynamically constrained by L2 and physical constants.
    """
    y_0 = compute_y0(L2, L_prime, c_pipi, B0_dim, f_pi_dim, mu_dim, r0_MeV)
    
    term1 = 4.0 * L_prime * x
    log_arg = (2.0 * B0_dim * x) / (mu_dim**2)
    log_arg = np.where(log_arg > 0, log_arg, 1e-10)
    
    term2_coeff = (3.0 * B0_dim) / (4.0 * np.pi**2 * f_pi_dim**2)
    term2 = term2_coeff * (4.0 * B0_dim * c_pipi - L_prime) * (np.log(log_arg) - 1.0) * (x**2)
    
    term3 = 2.0 * L2 * (x**2)
    
    return y_0 - term1 - term2 - term3

# =============================================================================
# 3. Load Data & Fit
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')

data_types = ['bare', 'smeared']

for dtype in data_types:
    dim_data = df.apply(lambda row: calculate_dimensionless(row, dtype), axis=1)
    x_data = np.array([d[0] for d in dim_data])
    y_data = np.array([d[2] for d in dim_data])
    y_err = np.array([d[3] for d in dim_data])
    
    for branch_name, params in param_sets.items():
        means, cov = params['means'], params['cov']
        
        # Central Dimensionless Parameters
        c_pipi_cen = (means[0] + means[1]/2) / 2
        L_prime_cen = 2 * B0_MeV * means[2] * r0_MeV_central
        B0_dim_cen = B0_MeV * r0_MeV_central
        f_pi_dim_cen = f_pi_MeV * r0_MeV_central
        mu_dim_cen = mu_MeV * r0_MeV_central
        
        # Central fit (Only L2 is a free parameter, dof = N - 1)
        try:
            popt_central, _ = curve_fit(
                lambda x, l2: chiral_model_universal_dim(
                    x, l2, L_prime_cen, c_pipi_cen, 
                    B0_dim_cen, f_pi_dim_cen, mu_dim_cen, r0_MeV_central
                ), 
                x_data, y_data, p0=[0.001], sigma=y_err, absolute_sigma=True
            )
            L2_cen = popt_central[0]
            y0_cen = compute_y0(L2_cen, L_prime_cen, c_pipi_cen, B0_dim_cen, f_pi_dim_cen, mu_dim_cen, r0_MeV_central)

            # --- CHI-SQUARED CALCULATION ---
            y_fit = chiral_model_universal_dim(
                x_data, L2_cen, L_prime_cen, c_pipi_cen, 
                B0_dim_cen, f_pi_dim_cen, mu_dim_cen, r0_MeV_central
            )
            residuals = y_data - y_fit
            chi_sq = np.sum((residuals / y_err)**2)
            dof = len(x_data) - 1 
            red_chi_sq = chi_sq / dof if dof > 0 else np.nan
            p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

        except Exception as e:
            print(f"Failed central fit for {dtype} {branch_name}: {e}")
            continue
            
        # Lists to store Bootstrap traces
        y0_list, L2_list = [], []
        c_pipi_list, L_prime_list = [], []
        
        for _ in range(N_boot):
            x_s_list, y_s_list = [], []
            
            # 1. Smear raw lattice values directly to construct bootstrap datasets
            for _, row in df.iterrows():
                r0_a_s = np.random.normal(row[f'r0_a_{dtype}'], row[f'r0_a_{dtype}_err'])
                a2sigma_s = np.random.normal(row[f'a2sigma_{dtype}'], row[f'a2sigma_{dtype}_err'])
                
                y_s_list.append(a2sigma_s * (r0_a_s ** 2))
                x_s_list.append(row['am_l'] * r0_a_s)
            
            x_s = np.array(x_s_list)
            y_s = np.array(y_s_list)
            
            # 2. Smear constants and physical scale r0
            l_s, e_s, f_s = np.random.multivariate_normal(means, cov)
            r0_s = np.random.normal(r0_MeV_central, r0_MeV_err)
            
            c_pipi_s = (l_s + e_s/2) / 2
            L_prime_s = 2 * B0_MeV * f_s * r0_s
            B0_dim_s = B0_MeV * r0_s
            f_pi_dim_s = f_pi_MeV * r0_s
            mu_dim_s = mu_MeV * r0_s
            
            c_pipi_list.append(c_pipi_s)
            L_prime_list.append(L_prime_s)
            
            try:
                popt_tot, _ = curve_fit(
                    lambda x, l2: chiral_model_universal_dim(
                        x, l2, L_prime_s, c_pipi_s, 
                        B0_dim_s, f_pi_dim_s, mu_dim_s, r0_s
                    ), 
                    x_s, y_s, p0=[L2_cen]
                )
                L2_s = popt_tot[0]
                y0_s = compute_y0(L2_s, L_prime_s, c_pipi_s, B0_dim_s, f_pi_dim_s, mu_dim_s, r0_s)
                
                L2_list.append(L2_s)
                y0_list.append(y0_s)
            except:
                continue

        if not L2_list: 
            continue
            
        fit_results.append({
            'Data_Type': dtype, 'Branch': branch_name,
            'y0_central': y0_cen, 'L2_central': L2_cen, 
            'chi_sq': chi_sq, 'dof': dof, 'red_chi_sq': red_chi_sq,
            'p_value': p_value,
            'L_prime_central': L_prime_cen, 'c_pipi_central': c_pipi_cen,
            'y0_traces': json.dumps(y0_list), 'L2_traces': json.dumps(L2_list), 
            'L_prime_traces': json.dumps(L_prime_list), 'c_pipi_traces': json.dumps(c_pipi_list)
        })

# Save to CSV
results_df = pd.DataFrame(fit_results)
results_df.to_csv(filename, index=False)

print(f"\nTheoretical Global Dimensionless Log Fit complete. Saved to '{filename}'.")
print("\nSummary of Extracted Parameters (y0 constrained, L2 fitted):")
print(results_df[['Data_Type', 'Branch', 'y0_central', 'L2_central', 'chi_sq', 'dof', 'red_chi_sq', 'p_value']].to_string(index=False))
