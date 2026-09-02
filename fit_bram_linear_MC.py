# Reads data_bram.csv,
# computes dimensionless variables directly from lattice data (propagating errors),
# performs the fit to find y_0 (dimensionless sigma_0) with fixed L_prime for each ensemble,
# uses Monte Carlo resampling for Statistical, Systematic, and Total uncertainties,
# and saves the results to fit_results_linear.csv.

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import json

# --- Constants ---
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV_central = r0_fm / hbar_c
B0_MeV = 2700

A_types = ['Ar0', 'Api12']
# Change index to select r0 or pi12 data
A_index = 0

N_MC = 1000  # Number of Monte Carlo samples

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_linear_MC_{A_types[A_index]}.csv'

# Defined branches
branches = {
    'positive': {'frac': 0.124, 'frac_err': 0.005},
    'negative': {'frac': -0.124, 'frac_err': 0.005}
}

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
    
    # Calculate y = (a^2 sigma) * (r0/a)^2
    y = a2sigma * (r0_a ** 2)
    y_err = y * np.sqrt((a2sigma_err / a2sigma)**2 + (2 * r0_a_err / r0_a)**2)
    
    # Calculate x = (am_l) * (r0/a)
    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)
    
    return x, x_err, y, y_err

def model_func(x, y_0, L_prime_val):
    # y = y_0 - 4*(L') * x
    return y_0 - 4 * L_prime_val * x

# --- Load Data ---
df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda x: "_".join(x.split('_')[-2:])) 
ensembles = df['mass_group'].unique()

results = []
data_types = ['bare', 'smeared']

for group in ensembles:
    subset = df[df['mass_group'] == group]
    
    for prefix in data_types:
        if subset.empty: continue

        x_list, x_err_list, y_list, y_err_list = [], [], [], []
        for _, row in subset.iterrows():
            x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
            x_list.append(x_val)
            x_err_list.append(x_e)
            y_list.append(y_val)
            y_err_list.append(y_e)
            
        x_arr, x_err_arr = np.array(x_list), np.array(x_err_list)
        y_arr, y_err_arr = np.array(y_list), np.array(y_err_list)

        # Loop over the branches
        for branch_name, params in branches.items():
            frac_central = params['frac']
            frac_err = params['frac_err']

            # Central L' computation
            L_prime_central = 2 * B0_MeV * frac_central * r0_MeV_central

            # Effective y error for the central goodness-of-fit check
            effective_y_err_central = np.sqrt(y_err_arr**2 + (4 * L_prime_central * x_err_arr)**2)

            # Central Fit
            try:
                popt_central, _ = curve_fit(
                    lambda x_data, y0: model_func(x_data, y0, L_prime_central), 
                    x_arr, y_arr, sigma=effective_y_err_central, absolute_sigma=True
                )
                y_0_central = popt_central[0]

                # Goodness of fit
                residuals = y_arr - model_func(x_arr, y_0_central, L_prime_central)
                chi_sq = np.sum((residuals / effective_y_err_central)**2)
                dof = len(x_arr) - 1 
                red_chi_sq = chi_sq / dof if dof > 0 else np.nan
                p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
            except Exception as e:
                print(f"Fit failed for group {group}, {prefix}, branch {branch_name}: {e}")
                continue

            # =========================================================================
            # Monte Carlo Error Propagation
            # =========================================================================
            y0_stat_samples = []
            y0_sys_samples = []
            y0_tot_samples = []

            for _ in range(N_MC):
                # 1. Resample data (for Stat & Tot)
                x_mc = np.random.normal(x_arr, x_err_arr)
                y_mc = np.random.normal(y_arr, y_err_arr)

                # 2. Resample theoretical parameters (for Sys & Tot)
                r0_fm_mc = np.random.normal(r0_fm, r0_fm_err)
                r0_MeV_mc = r0_fm_mc / hbar_c
                frac_mc = np.random.normal(frac_central, frac_err)
                L_prime_mc = 2 * B0_MeV * frac_mc * r0_MeV_mc

                # --- STATISTICAL FIT (Vary Data, Fixed L' central) ---
                try:
                    popt_stat, _ = curve_fit(
                        lambda x_data, y0: model_func(x_data, y0, L_prime_central), 
                        x_mc, y_mc, sigma=y_err_arr, absolute_sigma=True
                    )
                    y0_stat_samples.append(popt_stat[0])
                except: pass

                # --- SYSTEMATIC FIT (Fixed Data, Vary L') ---
                try:
                    popt_sys, _ = curve_fit(
                        lambda x_data, y0: model_func(x_data, y0, L_prime_mc), 
                        x_arr, y_arr, sigma=y_err_arr, absolute_sigma=True
                    )
                    y0_sys_samples.append(popt_sys[0])
                except: pass

                # --- TOTAL FIT (Vary Data AND Vary L') ---
                try:
                    popt_tot, _ = curve_fit(
                        lambda x_data, y0: model_func(x_data, y0, L_prime_mc), 
                        x_mc, y_mc, sigma=y_err_arr, absolute_sigma=True
                    )
                    y0_tot_samples.append(popt_tot[0])
                except: pass

            # Compute standard deviations across MC realizations
            y_0_stat_err = np.std(y0_stat_samples) if y0_stat_samples else np.nan
            y_0_sys_err = np.std(y0_sys_samples) if y0_sys_samples else np.nan
            y_0_total_err = np.std(y0_tot_samples) if y0_tot_samples else np.nan

            results.append({
                'Ensemble': group,
                'Data_Type': prefix,
                'Branch': branch_name,
                'y_0_central': y_0_central,
                'y_0_stat_err': y_0_stat_err,
                'y_0_sys_err_lamb': y_0_sys_err,
                'y_0_total_err': y_0_total_err,
                'L_prime_central': L_prime_central,
                'chi_sq': chi_sq,             
                'red_chi_sq': red_chi_sq,     
                'p_value': p_value,
                'y0_tot_list': json.dumps(y0_tot_samples)
            })

# Save and print output
results_df = pd.DataFrame(results).sort_values(by=['Ensemble', 'Data_Type', 'Branch'])
results_df.to_csv(filename, index=False)

print("\n--- DIMENSIONLESS LINEAR FITTING (MONTE CARLO) COMPLETE ---")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters (y_0 = r0^2 * sigma_0):")
print(results_df[['Ensemble', 'Data_Type', 'Branch', 'y_0_central', 'y_0_stat_err', 'y_0_sys_err_lamb', 'y_0_total_err', 'red_chi_sq', 'p_value']].to_string(index=False))