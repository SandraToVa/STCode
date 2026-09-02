# Does the same as fit_bram_log_v2.py but instead of separate in ensambles and do the fit for each ensamble it does a global fit

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
from scipy.stats import chi2

# =============================================================================
# 1. Constants & Parameter Sets
# =============================================================================
hbar_c = 197.3269804 
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200 

A_types = ['Ar0', 'Api12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_global_v2_{A_types[A_index]}.csv'

param_sets = {
    'Branch_1': {'means': [0.00978833, 0.0262025, -0.124474], 'cov': [[0.0000352816, -0.0000458309, 0.0000182559], [-0.0000458309, 0.0000615173, -0.0000286899], [0.0000182559, -0.0000286899, 0.0000271675]]},
    'Branch_2': {'means': [-0.0458536, 0.0815424, -0.124023], 'cov': [[0.0000190469, -0.0000134491, -8.07208e-6], [-0.0000134491, 0.0000131926, -2.67543e-6], [-8.07208e-6, -2.67543e-6, 0.0000276483]]},
    'Branch_3': {'means': [0.0458536, -0.0815424, 0.124023], 'cov': [[0.0000190469, -0.0000134491, -8.07208e-6], [-0.0000134491, 0.0000131926, -2.67543e-6], [-8.07208e-6, -2.67543e-6, 0.0000276483]]},
    'Branch_4': {'means': [-0.00978833, -0.0262025, 0.124474], 'cov': [[0.0000352816, -0.0000458309, 0.0000182559], [-0.0000458309, 0.0000615173, -0.0000286899], [0.0000182559, -0.0000286899, 0.0000271675]]}
}

# =============================================================================
# 2. Data Processing & Global Model
# =============================================================================
def calculate_physics(df_row, prefix):
    r0_a, r0_a_err = df_row[f'r0_a_{prefix}'], df_row[f'r0_a_{prefix}_err']
    a2sigma, a2sigma_err = df_row[f'a2sigma_{prefix}'], df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    a = r0_MeV / r0_a
    a_err = a * (r0_a_err / r0_a + r0_MeV_err / r0_MeV)  # Combine relative errors of r0_a and r0_MeV
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    sigma = a2sigma / (a**2)
    sigma_err = sigma * np.sqrt((a2sigma_err/a2sigma)**2 + (2*a_err/a)**2)
    return m_l, m_l_err, sigma, sigma_err

def chiral_model_universal(ml, s0, l2, lamb, eta, frac):
    c_pipi = (lamb + eta/2) / 2
    lamb_prime = 2 * B0_MeV * frac
    
    term1 = 4 * lamb_prime * ml
    log_arg = np.where((2 * B0_MeV * ml) / (mu_MeV**2) > 0, (2 * B0_MeV * ml) / (mu_MeV**2), 1e-10)
    term2_coeff = (3 * B0_MeV) / (4 * np.pi**2 * f_pi_MeV**2)
    term2 = term2_coeff * (4 * B0_MeV * c_pipi - lamb_prime) * (np.log(log_arg) - 1) * (ml**2)
    term3 = 2 * l2 * (ml**2)
    
    return s0 - term1 - term2 - term3

# =============================================================================
# 3. Load Data & Fit
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')

data_types = ['bare', 'smeared']
fit_results = []
N_boot = 1000 

for dtype in data_types:
    # Use all data points simultaneously for the fit
    phys_data = df.apply(lambda row: calculate_physics(row, dtype), axis=1)
    m_l_data = np.array([x[0] for x in phys_data])
    sigma_data = np.array([x[2] for x in phys_data])
    sigma_err = np.array([x[3] for x in phys_data])
    
    for branch_name, params in param_sets.items():
        means, cov = params['means'], params['cov']
        
        # Central fit (1 universal sigma_0, 1 universal lamb_2prime)
        try:
            popt_central, _ = curve_fit(lambda x, s0, l2: chiral_model_universal(x, s0, l2, *means), 
                                        m_l_data, sigma_data, p0=[200000, 1000])
            s0_cen, l2_cen = popt_central

            # --- CHI-SQUARED CALCULATION ---
            # Evaluate the best fit line at the data points
            y_fit = chiral_model_universal(m_l_data, s0_cen, l2_cen, *means)
            residuals = sigma_data - y_fit
            chi_sq = np.sum((residuals / sigma_err)**2)
            dof = len(m_l_data) - 2 # 2 free parameters: s0 and l2
            red_chi_sq = chi_sq / dof
            p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
            # -------------------------------

        except Exception as e:
            print(f"Failed central fit for {dtype} {branch_name}: {e}")
            continue
            
        c_pipi_cen = (means[0] + means[1]/2) / 2
        lamb_prime_cen = 2 * B0_MeV * means[2]
        
        # Lists to store Bootstrap traces
        s0_list, l2_list = [], []
        c_pipi_list, lamb_prime_list = [], []
        
        for _ in range(N_boot):
            ml_s, sig_s = [], []
            # Smear all experimental points
            for _, row in df.iterrows():
                r0_a_s = np.random.normal(row[f'r0_a_{dtype}'], row[f'r0_a_{dtype}_err'])
                a2sigma_s = np.random.normal(row[f'a2sigma_{dtype}'], row[f'a2sigma_{dtype}_err'])
                a_s = r0_MeV / r0_a_s
                ml_s.append(row['am_l'] / a_s)
                sig_s.append(a2sigma_s / (a_s**2))
            
            ml_s = np.array(ml_s)
            sig_s = np.array(sig_s)
            
            # Smear constants
            l_s, e_s, f_s = np.random.multivariate_normal(means, cov)
            
            c_pipi_list.append((l_s + e_s/2) / 2)
            lamb_prime_list.append(2 * B0_MeV * f_s)
            
            try:
                popt_tot, _ = curve_fit(lambda x, s0, l2: chiral_model_universal(x, s0, l2, l_s, e_s, f_s), 
                                        ml_s, sig_s, p0=[s0_cen, l2_cen])
                s0_list.append(popt_tot[0])
                l2_list.append(popt_tot[1])
            except:
                continue

        if not l2_list: continue
            
        fit_results.append({
            'Data_Type': dtype, 'Branch': branch_name,
            'sigma_0_central': s0_cen, 'lamb_2prime_central': l2_cen, 
            'chi_sq': chi_sq, 'dof': dof, 'red_chi_sq': red_chi_sq,
            'p_value': p_value,
            'lamb_prime_central': lamb_prime_cen, 'c_pipi_central': c_pipi_cen,
            's0_traces': json.dumps(s0_list), 'l2_traces': json.dumps(l2_list), 
            'lamb_prime_traces': json.dumps(lamb_prime_list), 'c_pipi_traces': json.dumps(c_pipi_list)
        })

# Save to CSV
results_df = pd.DataFrame(fit_results)
results_df.to_csv(filename, index=False)

print(f"\nStrictly Global Fit complete. Saved to '{filename}'.")
print("\nSummary of Extracted Parameters:")
# Clean terminal printout focusing on the goodness of fit
print(results_df[['Data_Type', 'Branch', 'sigma_0_central', 'lamb_2prime_central', 'chi_sq', 'dof', 'red_chi_sq', 'p_value']].to_string(index=False))