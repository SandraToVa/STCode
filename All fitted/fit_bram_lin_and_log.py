# Reads data_bram.csv,
# computes dimensionless variables directly from lattice data (propagating errors),
# performs the fit to find y_0 (dimensionless sigma_0) for each ensemble,
# using either a linear or a logaritmic model, depending on the user's choice,
# does not use any reasult for the paper to obtain the fitting parameters.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import json

# =============================================================================
# 1. Constants & Unit Conversions
# =============================================================================

# Fix seed for reproducibility
np.random.seed(42)

hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV_central = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200

A_types = ['Ar0', 'Api12']
fit_types = ['linear', 'logarithmic']
# Change index to select r0 or pi12 data
A_index = 0
fit_index = 1  # 0 for linear, 1 for logarithmic

# Number of Monte Carlo iterations for error propagation
N_MC = 1000

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_full_{fit_types[fit_index]}_{A_types[A_index]}.csv'

# =============================================================================
# 3. Data Processing Function & Chiral Model
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
    
    y = a2sigma * (r0_a ** 2)
    y_err = y * np.sqrt((a2sigma_err / a2sigma)**2 + (2 * r0_a_err / r0_a)**2)
    
    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)
    
    return x, x_err, y, y_err

def linear_model_dim(x, y_0, L_prime):
    # Term 1: 4 * L' * x
    term1 = 4 * L_prime * x
    return y_0 - term1 

def logarithmic_model_dim(x, y_0, L2, L_prime, c_pipi, B0_dim, f_pi_dim, mu_dim):
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

# =============================================================================
# 5. Fitting Routine (Monte Carlo Loop)
# =============================================================================
for group in groups:
    subset = df[df['mass_group'] == group]

    for dtype in data_types:
        if subset.empty: continue

        # Extract central dimensionless data
        dim_data = subset.apply(lambda row: calculate_dimensionless(row, dtype), axis=1)
        x_data = np.array([d[0] for d in dim_data])
        x_err = np.array([d[1] for d in dim_data])
        y_data = np.array([d[2] for d in dim_data])
        y_err = np.array([d[3] for d in dim_data])
        
        # Central Derived Constants
        B0_dim_central = B0_MeV * r0_MeV_central
        f_pi_dim_central = f_pi_MeV * r0_MeV_central
        mu_dim_central = mu_MeV * r0_MeV_central

        if fit_index == 0:
            # ======================== LINEAR FIT ========================
            try:
                popt_central, _ = curve_fit(
                    linear_model_dim, x_data, y_data, p0=[0.05, 0.001], sigma=y_err, absolute_sigma=True
                )
                y0_central, L_prime_central = popt_central

                # GOODNESS OF FIT
                y_fit = linear_model_dim(x_data, y0_central, L_prime_central)
                chi_sq = np.sum(((y_data - y_fit) / y_err)**2)
                dof = len(x_data) - 2 
                red_chi_sq = chi_sq / dof if dof > 0 else np.nan
                p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
            except Exception as e:
                print(f"Fit failed for {group} ({dtype}): {e}")
                continue

            # MC Loop for uncertainties
            y0_stat_list, y0_sys_list, y0_tot_list = [], [], []
            L_prime_stat_list, L_prime_sys_list, L_prime_tot_list = [], [], []

            for _ in range(N_MC):
                # Data variation (Stat & Tot)
                x_mc = np.random.normal(x_data, x_err)
                y_mc = np.random.normal(y_data, y_err)

                # Linear model doesn't depend on r0_fm explicitly, so sys error is effectively 0 here.
                # However, for structure parity, we track them.
                try:
                    # Stat (vary data, fixed model)
                    p_stat, _ = curve_fit(linear_model_dim, x_mc, y_mc, p0=popt_central, sigma=y_err, absolute_sigma=True)
                    y0_stat_list.append(p_stat[0])
                    L_prime_stat_list.append(p_stat[1])
                    
                    # Tot (same as stat for linear model)
                    y0_tot_list.append(p_stat[0])
                    L_prime_tot_list.append(p_stat[1])
                    
                    # Sys (fixed data, varying r0 would go here, but linear lacks r0 dependencies)
                    y0_sys_list.append(y0_central)
                    L_prime_sys_list.append(L_prime_central)
                except:
                    continue

            fit_results.append({
                'Ensemble': group,
                'Data_Type': dtype,
                'y0_central': y0_central,
                'L_prime_central': L_prime_central,
                'chi_sq': chi_sq,             
                'red_chi_sq': red_chi_sq,
                'p_value': p_value,
                'y0_stat_err': np.std(y0_stat_list) if y0_stat_list else np.nan,
                'y0_sys_err': np.std(y0_sys_list) if y0_sys_list else np.nan,
                'y0_total_err': np.std(y0_tot_list) if y0_tot_list else np.nan,
                'L_prime_stat_err': np.std(L_prime_stat_list) if L_prime_stat_list else np.nan,
                'L_prime_sys_err': np.std(L_prime_sys_list) if L_prime_sys_list else np.nan,
                'L_prime_total_err': np.std(L_prime_tot_list) if L_prime_tot_list else np.nan,
                'y0_tot_list': json.dumps(y0_tot_list),
                'L_prime_tot_list': json.dumps(L_prime_tot_list)
            })

        elif fit_index == 1:  
            # ======================== LOGARITHMIC FIT ========================
            try:
                # Need 4 initial guesses for 4 parameters!
                p0_log = [0.05, 0.001, 0.001, 0.001]
                popt_central, _ = curve_fit(
                    lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(
                        x_val, y0, l2, l1, cpi, B0_dim_central, f_pi_dim_central, mu_dim_central
                    ), 
                    x_data, y_data, p0=p0_log, sigma=y_err, absolute_sigma=True, maxfev=5000
                )
                y0_central, L2_central, L_prime_central, c_pipi_central = popt_central

                # GOODNESS OF FIT
                y_fit = logarithmic_model_dim(x_data, y0_central, L2_central, L_prime_central, 
                                              c_pipi_central, B0_dim_central, f_pi_dim_central, mu_dim_central)
                chi_sq = np.sum(((y_data - y_fit) / y_err)**2)
                dof = len(x_data) - 4  # 4 parameters for log fit
                red_chi_sq = chi_sq / dof if dof > 0 else np.nan
                p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
            except Exception as e:
                print(f"Fit failed for {group} ({dtype}): {e}")
                continue
            
            # MC Loop for uncertainties
            y0_stat_list, y0_sys_list, y0_tot_list = [], [], []
            L_prime_stat_list, L_prime_sys_list, L_prime_tot_list = [], [], []
            L2_stat_list, L2_sys_list, L2_tot_list = [], [], []
            c_pipi_stat_list, c_pipi_sys_list, c_pipi_tot_list = [], [], []

            for _ in range(N_MC):
                # Data variation
                x_mc = np.random.normal(x_data, x_err)
                y_mc = np.random.normal(y_data, y_err)

                # Physical constant variation (Sys error source)
                r0_MeV_mc = np.random.normal(r0_fm, r0_fm_err) / hbar_c
                B0_dim_mc = B0_MeV * r0_MeV_mc
                f_pi_dim_mc = f_pi_MeV * r0_MeV_mc
                mu_dim_mc = mu_MeV * r0_MeV_mc

                # 1. STAT: Vary data, central constants
                try:
                    p_stat, _ = curve_fit(
                        lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(x_val, y0, l2, l1, cpi, B0_dim_central, f_pi_dim_central, mu_dim_central), 
                        x_mc, y_mc, p0=popt_central, sigma=y_err, absolute_sigma=True, maxfev=5000)
                    y0_stat_list.append(p_stat[0])
                    L2_stat_list.append(p_stat[1])
                    L_prime_stat_list.append(p_stat[2])
                    c_pipi_stat_list.append(p_stat[3])
                except: pass

                # 2. SYS: Central data, vary constants
                try:
                    p_sys, _ = curve_fit(
                        lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(x_val, y0, l2, l1, cpi, B0_dim_mc, f_pi_dim_mc, mu_dim_mc), 
                        x_data, y_data, p0=popt_central, sigma=y_err, absolute_sigma=True, maxfev=5000)
                    y0_sys_list.append(p_sys[0])
                    L2_sys_list.append(p_sys[1])
                    L_prime_sys_list.append(p_sys[2])
                    c_pipi_sys_list.append(p_sys[3])
                except: pass

                # 3. TOT: Vary data AND vary constants
                try:
                    p_tot, _ = curve_fit(
                        lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(x_val, y0, l2, l1, cpi, B0_dim_mc, f_pi_dim_mc, mu_dim_mc), 
                        x_mc, y_mc, p0=popt_central, sigma=y_err, absolute_sigma=True, maxfev=5000)
                    y0_tot_list.append(p_tot[0])
                    L2_tot_list.append(p_tot[1])
                    L_prime_tot_list.append(p_tot[2])
                    c_pipi_tot_list.append(p_tot[3])
                except: pass

            fit_results.append({
                'Ensemble': group,
                'Data_Type': dtype,
                'y0_central': y0_central,
                'L_prime_central': L_prime_central,
                'L2_central': L2_central,
                'c_pipi_central': c_pipi_central,
                'chi_sq': chi_sq,             
                'red_chi_sq': red_chi_sq,
                'p_value': p_value,
                'y0_stat_err': np.std(y0_stat_list) if y0_stat_list else np.nan,
                'y0_sys_err': np.std(y0_sys_list) if y0_sys_list else np.nan,
                'y0_total_err': np.std(y0_tot_list) if y0_tot_list else np.nan,
                'L_prime_stat_err': np.std(L_prime_stat_list) if L_prime_stat_list else np.nan,
                'L_prime_sys_err': np.std(L_prime_sys_list) if L_prime_sys_list else np.nan,
                'L_prime_total_err': np.std(L_prime_tot_list) if L_prime_tot_list else np.nan,
                'L2_stat_err': np.std(L2_stat_list) if L2_stat_list else np.nan,
                'L2_sys_err': np.std(L2_sys_list) if L2_sys_list else np.nan,
                'L2_total_err': np.std(L2_tot_list) if L2_tot_list else np.nan,
                'c_pipi_stat_err': np.std(c_pipi_stat_list) if c_pipi_stat_list else np.nan,
                'c_pipi_sys_err': np.std(c_pipi_sys_list) if c_pipi_sys_list else np.nan,
                'c_pipi_total_err': np.std(c_pipi_tot_list) if c_pipi_tot_list else np.nan,
                'y0_tot_list': json.dumps(y0_tot_list),
                'L_prime_tot_list': json.dumps(L_prime_tot_list),
                'L2_tot_list': json.dumps(L2_tot_list),
                'c_pipi_tot_list': json.dumps(c_pipi_tot_list)
            })

results_df = pd.DataFrame(fit_results)
if not results_df.empty:
    results_df.to_csv(filename, index=False)
    print(f"\n--- DIMENSIONLESS {fit_types[fit_index].upper()} FITTING COMPLETE ---")
    print(f"Results successfully saved to '{filename}'.")
    print("\nSummary of Extracted Parameters:")
    
    if fit_index == 0:
        cols_to_print = ['Ensemble', 'Data_Type', 'y0_central', 'y0_total_err', 'L_prime_central', 'L_prime_total_err', 'red_chi_sq', 'p_value']
    else:
        cols_to_print = ['Ensemble', 'Data_Type', 'y0_central', 'y0_total_err', 'L_prime_central', 'L_prime_total_err', 'L2_central', 'L2_total_err', 'c_pipi_central', 'c_pipi_total_err', 'red_chi_sq', 'p_value']
        
    print(results_df[cols_to_print].to_string(index=False))
else:
    print("\nNo fits were successful or data was empty.")