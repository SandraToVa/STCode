# Reads data_bram.csv,
# converts lattice units to physical units (propagating errors),
# performs the fit to find σ_0 and λ'' for each ensemble,
# calculates systematic errors from λ', η, and frac,
# and saves the results to fit_results_log.csv.

import pandas as pd
import numpy as np
import json
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
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200 # order of magnitude of pion mass in MeV, used as scale for chiral logs

A_types = ['Ar0', 'Api12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_{A_types[A_index]}_v2.csv'


# =============================================================================
# 2. Parameter Sets (Covariance matrices for [lamb, eta, frac])
# =============================================================================
# Order of variables in vectors/matrices: [lamb, eta, frac]
# from Uncertainty Propagation document in Mathematica
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

def chiral_model(ml, sigma_0, lamb_2prime, lamb, eta, frac):
    # Derived parameters
    c_pipi = (lamb + eta/2) / 2
    lamb_prime = 2 * B0_MeV * frac
    
    # 4 * lamb_prime * ml
    term1 = 4 * lamb_prime * ml
    
    # Logarithmic argument: 2*B0*m_l / mu^2
    log_arg = (2 * B0_MeV * ml) / (mu_MeV**2)
    log_arg = np.where(log_arg > 0, log_arg, 1e-10) # protect against zeros
    
    # Pre-factor for the log term
    term2_coeff = (3 * B0_MeV) / (4 * np.pi**2 * f_pi_MeV**2)
    term2 = term2_coeff * (4 * B0_MeV * c_pipi - lamb_prime) * ( np.log(log_arg) - 1 )
    term2 = term2 * (ml**2)

    # lamb_2prime term
    term3 = 2 * lamb_2prime * (ml**2)

    return sigma_0 - term1 - term2 - term3

# =============================================================================
# 4. Load & Group Data
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')

# Identify ensemble (M_i, M_ii, M_iii)
def get_ensemble_group(ens_str):
    if 'M_iii' in ens_str: return 'M_iii'
    if 'M_ii' in ens_str:  return 'M_ii'
    if 'M_i' in ens_str:   return 'M_i'
    return 'Unknown'

df['mass_group'] = df['Ensemble'].apply(get_ensemble_group)

# We want 6 datasets: (M_i, M_ii, M_iii) x (bare, smeared)
groups = ['M_i', 'M_ii', 'M_iii']
data_types = ['bare', 'smeared']

# Prepare a list to save fit results
fit_results = []
N_boot = 1000 # Number of Monte Carlo iterations

# =============================================================================
# 5. Fitting Routine (Monte Carlo Loop)
# =============================================================================

for group in groups:
    for dtype in data_types:
        subset = df[df['mass_group'] == group]

        if subset.empty: continue

        # Extract physics values    
        phys_data = subset.apply(lambda row: calculate_physics(row, dtype), axis=1)
        m_l_data = np.array([x[0] for x in phys_data])
        m_l_err = np.array([x[1] for x in phys_data])
        sigma_data = np.array([x[2] for x in phys_data])
        sigma_err = np.array([x[3] for x in phys_data])
        
        for branch_name, params in param_sets.items():
            means = params['means']
            cov = params['cov']
            
            # Central fit
            try:
                popt_central, _ = curve_fit(lambda x, s0, l2: chiral_model(x, s0, l2, *means), 
                                            m_l_data, sigma_data, p0=[10000, 100])
                s0_central = popt_central[0]
                l2_central = popt_central[1]

                # GOODNESS OF FIT CALCULATION
                # Calculate the model prediction
                y_fit = chiral_model(m_l_data, s0_central, l2_central, *means)
                residuals = sigma_data - y_fit
                # Calculate Chi-Square
                chi_sq = np.sum((residuals / sigma_err)**2)
                dof = len(m_l_data) - 2 # 2 fit parameters: sigma_0 and lamb_2prime
                red_chi_sq = chi_sq / dof
                p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

            except:
                continue
            
            # Derived constants central
            c_pipi_central = (means[0] + means[1]/2) / 2
            lamb_prime_central = 2 * B0_MeV * means[2]
            
            # Bootstrap distributions
            s0_stat_list, s0_sys_list, s0_tot_list = [], [], []
            l2_stat_list, l2_sys_list, l2_tot_list = [], [], []
            c_pipi_list, lamb_prime_list = [], []
            
            for _ in range(N_boot):
                # Smear raw lattice values to preserve m_l & sigma correlations
                ml_s = []
                sig_s = []
                for _, row in subset.iterrows():
                    # 1. Smear the independent raw variables
                    r0_a_s = np.random.normal(row[f'r0_a_{dtype}'], row[f'r0_a_{dtype}_err'])
                    a2sigma_s = np.random.normal(row[f'a2sigma_{dtype}'], row[f'a2sigma_{dtype}_err'])
                    
                    # 2. Recalculate physical variables (preserves covariance perfectly)
                    a_s = r0_MeV / r0_a_s
                    m_l_s = row['am_l'] / a_s
                    sigma_s = a2sigma_s / (a_s**2)
                    
                    ml_s.append(m_l_s)
                    sig_s.append(sigma_s)
                
                ml_s = np.array(ml_s)
                sig_s = np.array(sig_s)
                # Smear constants (Sys)
                l_s, e_s, f_s = np.random.multivariate_normal(means, cov)
                
                # Store derived constant distributions
                c_pipi_list.append((l_s + e_s/2) / 2)
                lamb_prime_list.append(2 * B0_MeV * f_s)
                
                try:
                    # Stat Error only (fixed params, varying data)
                    popt_stat, _ = curve_fit(lambda x, s0, l2: chiral_model(x, s0, l2, *means), 
                                             ml_s, sig_s, p0=[s0_central, l2_central])
                    s0_stat_list.append(popt_stat[0])
                    l2_stat_list.append(popt_stat[1])
                    
                    # Sys Error only (fixed data, varying params)
                    popt_sys, _ = curve_fit(lambda x, s0, l2: chiral_model(x, s0, l2, l_s, e_s, f_s), 
                                            ml_s, sig_s, p0=[s0_central, l2_central])
                    s0_sys_list.append(popt_sys[0])
                    l2_sys_list.append(popt_sys[1])
                    
                    # Total Error (varying both)
                    popt_tot, _ = curve_fit(lambda x, s0, l2: chiral_model(x, s0, l2, l_s, e_s, f_s), 
                                            ml_s, sig_s, p0=[s0_central, l2_central])
                    s0_tot_list.append(popt_tot[0])
                    l2_tot_list.append(popt_tot[1])
                except:
                    continue

            if not s0_tot_list:
                continue
            
            # Calculate standard deviations & bounds
            s0_stat_err = np.std(s0_stat_list)
            s0_sys_err = np.std(s0_sys_list)
            s0_tot_err = np.std(s0_tot_list)
            l2_stat_err = np.std(l2_stat_list)
            l2_sys_err = np.std(l2_sys_list)
            l2_tot_err = np.std(l2_tot_list)
            
            c_pipi_err = np.std(c_pipi_list)
            lamb_prime_err = np.std(lamb_prime_list)
            
            fit_results.append({
                'Ensemble': group,
                'Data_Type': dtype,
                'Branch': branch_name,
                'sigma_0_central': s0_central,
                'lamb_2prime_central': l2_central,
                'chi_sq': chi_sq,             
                'red_chi_sq': red_chi_sq,
                'p_value': p_value,
                'sigma_0_stat_err': s0_stat_err,
                'sigma_0_sys_err': s0_sys_err,
                'sigma_0_total_err': s0_tot_err,
                'lamb_2prime_stat_err': l2_stat_err,
                'lamb_2prime_sys_err': l2_sys_err,
                'lamb_2prime_total_err': l2_tot_err,
                'sigma_0_up': s0_central + s0_tot_err,
                'sigma_0_down': s0_central - s0_tot_err,
                'lamb_2prime_up': l2_central + l2_tot_err,
                'lamb_2prime_down': l2_central - l2_tot_err,
                'lamb_prime_central': lamb_prime_central,
                'lamb_prime_up': lamb_prime_central + lamb_prime_err,
                'lamb_prime_down': lamb_prime_central - lamb_prime_err,
                'c_pipi_central': c_pipi_central,
                'c_pipi_up': c_pipi_central + c_pipi_err,
                'c_pipi_down': c_pipi_central - c_pipi_err,
                # ADD THESE 4 LINES TO SAVE THE TRACES:
                's0_tot_list': json.dumps(s0_tot_list),
                'l2_tot_list': json.dumps(l2_tot_list),
                'lamb_prime_list': json.dumps(lamb_prime_list),
                'c_pipi_list': json.dumps(c_pipi_list)
            })

# Save to CSV
results_df = pd.DataFrame(fit_results)
results_df.to_csv(filename, index=False)

print("\n--- FITTING COMPLETE ---")
print(f"Results successfully saved to '{filename}'.")
print("\nSummary of Extracted Parameters:")
# Hide some columns for a cleaner terminal view, but everything is in the CSV
print(results_df[['Ensemble', 'Data_Type', 'Branch', 'sigma_0_central', 'sigma_0_total_err', 'lamb_2prime_central', 'lamb_2prime_total_err', 'red_chi_sq', 'p_value']].to_string(index=False))