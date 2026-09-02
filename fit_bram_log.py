# Reads data_bram.csv,
# computes dimensionless variables directly from lattice data,
# performs the fit to find y_0 and L'' for each ensemble,
# uses Monte Carlo bootstrap to propagate correlated parameter errors and r0 errors,
# and saves the results to fit_results_dimensionless_log.csv.

import pandas as pd
import numpy as np
import json
from scipy.optimize import curve_fit
from scipy.stats import chi2

# =============================================================================
# 1. Constants & Unit Conversions
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV_central = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200

A_types = ['Ar0', 'Api12']
A_index = 0

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_{A_types[A_index]}.csv'

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
# 3. Data Processing Function & Chiral Model
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

def chiral_model_dim(x, y_0, L2, L_prime, c_pipi, B0_dim, f_pi_dim, mu_dim):
    # Term 1: 4 * L' * x
    term1 = 4 * L_prime * x
    
    # Term 2: Logarithmic
    log_arg = (2 * B0_dim * x) / (mu_dim**2)
    log_arg = np.where(log_arg > 0, log_arg, 1e-10) 
    
    term2_coeff = (3 * B0_dim) / (4 * np.pi**2 * f_pi_dim**2)
    term2 = term2_coeff * (4 * B0_dim * c_pipi - L_prime) * (np.log(log_arg) - 1)
    term2 = term2 * (x**2)

    # Term 3: L'' term
    term3 = 2 * L2 * (x**2)

    return y_0 - term1 - term2 - term3

# =============================================================================
# 4. Load & Group Data
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')

def get_ensemble_group(ens_str):
    if 'M_iii' in ens_str: return 'M_iii'
    if 'M_ii' in ens_str:  return 'M_ii'
    if 'M_i' in ens_str:   return 'M_i'
    return 'Unknown'

df['mass_group'] = df['Ensemble'].apply(get_ensemble_group)

groups = ['M_i', 'M_ii', 'M_iii']
data_types = ['bare', 'smeared']

fit_results = []
N_boot = 1000

# =============================================================================
# 5. Fitting Routine (Monte Carlo Loop)
# =============================================================================
for group in groups:
    for dtype in data_types:
        subset = df[df['mass_group'] == group]

        if subset.empty: continue

        # Extract central dimensionless data
        dim_data = subset.apply(lambda row: calculate_dimensionless(row, dtype), axis=1)
        x_data = np.array([d[0] for d in dim_data])
        x_err = np.array([d[1] for d in dim_data])
        y_data = np.array([d[2] for d in dim_data])
        y_err = np.array([d[3] for d in dim_data])
        
        for branch_name, params in param_sets.items():
            means = params['means']
            cov = params['cov']
            
            # Central Derived Constants
            c_pipi_central = (means[0] + means[1]/2) / 2
            L_prime_central = 2 * B0_MeV * means[2] * r0_MeV_central
            B0_dim_central = B0_MeV * r0_MeV_central
            f_pi_dim_central = f_pi_MeV * r0_MeV_central
            mu_dim_central = mu_MeV * r0_MeV_central
            
            try:
                # Central fit
                popt_central, _ = curve_fit(
                    lambda x_val, y0, l2: chiral_model_dim(
                        x_val, y0, l2, L_prime_central, c_pipi_central, 
                        B0_dim_central, f_pi_dim_central, mu_dim_central
                    ), 
                    x_data, y_data, p0=[0.05, 0.001], sigma=y_err, absolute_sigma=True
                )
                y0_central = popt_central[0]
                L2_central = popt_central[1]

                # GOODNESS OF FIT
                y_fit = chiral_model_dim(x_data, y0_central, L2_central, L_prime_central, 
                                         c_pipi_central, B0_dim_central, f_pi_dim_central, mu_dim_central)
                chi_sq = np.sum(((y_data - y_fit) / y_err)**2)
                dof = len(x_data) - 2 
                red_chi_sq = chi_sq / dof if dof > 0 else np.nan
                p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
            except:
                continue
            
            y0_stat_list, y0_sys_list, y0_tot_list = [], [], []
            L2_stat_list, L2_sys_list, L2_tot_list = [], [], []
            
            for _ in range(N_boot):
                # 1. Smear Data
                x_s_list, y_s_list = [], []
                for _, row in subset.iterrows():
                    r0_a_s = np.random.normal(row[f'r0_a_{dtype}'], row[f'r0_a_{dtype}_err'])
                    a2sigma_s = np.random.normal(row[f'a2sigma_{dtype}'], row[f'a2sigma_{dtype}_err'])
                    
                    y_s_list.append(a2sigma_s * (r0_a_s ** 2))
                    x_s_list.append(row['am_l'] * r0_a_s)
                
                x_s = np.array(x_s_list)
                y_s = np.array(y_s_list)

                # 2. Smear Parameters and physical r0
                l_s, e_s, f_s = np.random.multivariate_normal(means, cov)
                r0_s = np.random.normal(r0_MeV_central, r0_MeV_err)
                
                c_pipi_s = (l_s + e_s/2) / 2
                L_prime_s = 2 * B0_MeV * f_s * r0_s
                B0_dim_s = B0_MeV * r0_s
                f_pi_dim_s = f_pi_MeV * r0_s
                mu_dim_s = mu_MeV * r0_s
                
                try:
                    # Stat Error only (fixed central params, varying data)
                    popt_stat, _ = curve_fit(
                        lambda x_val, y0, l2: chiral_model_dim(x_val, y0, l2, L_prime_central, c_pipi_central, B0_dim_central, f_pi_dim_central, mu_dim_central), 
                        x_s, y_s, p0=[y0_central, L2_central]
                    )
                    y0_stat_list.append(popt_stat[0])
                    L2_stat_list.append(popt_stat[1])
                    
                    # Sys Error only (fixed central data, varying params)
                    popt_sys, _ = curve_fit(
                        lambda x_val, y0, l2: chiral_model_dim(x_val, y0, l2, L_prime_s, c_pipi_s, B0_dim_s, f_pi_dim_s, mu_dim_s), 
                        x_data, y_data, p0=[y0_central, L2_central]
                    )
                    y0_sys_list.append(popt_sys[0])
                    L2_sys_list.append(popt_sys[1])
                    
                    # Total Error (varying both)
                    popt_tot, _ = curve_fit(
                        lambda x_val, y0, l2: chiral_model_dim(x_val, y0, l2, L_prime_s, c_pipi_s, B0_dim_s, f_pi_dim_s, mu_dim_s), 
                        x_s, y_s, p0=[y0_central, L2_central]
                    )
                    y0_tot_list.append(popt_tot[0])
                    L2_tot_list.append(popt_tot[1])
                except:
                    continue

            if not y0_tot_list:
                continue
            
            fit_results.append({
                'Ensemble': group,
                'Data_Type': dtype,
                'Branch': branch_name,
                'y0_central': y0_central,
                'L2_central': L2_central,
                'chi_sq': chi_sq,             
                'red_chi_sq': red_chi_sq,
                'p_value': p_value,
                'y0_stat_err': np.std(y0_stat_list),
                'y0_sys_err': np.std(y0_sys_list),
                'y0_total_err': np.std(y0_tot_list),
                'L2_stat_err': np.std(L2_stat_list),
                'L2_sys_err': np.std(L2_sys_list),
                'L2_total_err': np.std(L2_tot_list),
                'y0_tot_list': json.dumps(y0_tot_list),
                'L2_tot_list': json.dumps(L2_tot_list)
            })

results_df = pd.DataFrame(fit_results)
results_df.to_csv(filename, index=False)

print("\n--- DIMENSIONLESS LOG FITTING COMPLETE ---")
print(f"Results successfully saved to '{filename}'.")
print("\nSummary of Extracted Parameters (y0 = r0^2 * sigma_0, L2 = r0^2 * lamb_2prime):")
print(results_df[['Ensemble', 'Data_Type', 'Branch', 'y0_central', 'y0_total_err', 'L2_central', 'L2_total_err', 'red_chi_sq', 'p_value']].to_string(index=False))