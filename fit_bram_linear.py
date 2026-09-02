# Reads data_bram.csv,
# computes dimensionless variables directly from lattice data (propagating errors),
# performs the fit to find y_0 (dimensionless sigma_0) for each ensemble,
# calculates systematic errors from λ',
# and saves the results to fit_results_linear.csv.

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

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_linear_{A_types[A_index]}.csv'

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
    # Assuming am_l has negligible/zero uncertainty as in the original setup
    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)
    
    return x, x_err, y, y_err

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda x: "_".join(x.split('_')[-2:])) 
ensembles = df['mass_group'].unique()

results = []
data_types = ['bare', 'smeared']

for group in ensembles:
    subset = df[df['mass_group'] == group]
    
    for prefix in data_types:
        x_list, x_err_list, y_list, y_err_list = [], [], [], []
        for _, row in subset.iterrows():
            x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
            x_list.append(x_val)
            x_err_list.append(x_e)
            y_list.append(y_val)
            y_err_list.append(y_e)
            
        x_arr, x_err_arr = np.array(x_list), np.array(x_err_list)
        y_arr, y_err_arr = np.array(y_list), np.array(y_err_list)

        def effective_y_err(L_prime_val):
            # Effective variance combining y and x errors
            return np.sqrt(y_err_arr**2 + (4 * L_prime_val * x_err_arr)**2)

        def model_func(x, y_0, L_prime_val):
            # y = y_0 - 4*(lambda' * r0) * x
            return y_0 - 4 * L_prime_val * x

        # Loop over the branches
        for branch_name, params in branches.items():
            # Dimensionless slope constant L' = 2 * B0 * frac * r0
            L_prime = 2 * B0_MeV * params['frac'] * r0_MeV

            # Propagate error combining both the fraction and r0 uncertainties
            # Using the absolute value to handle the negative branch correctly
            rel_err_frac = params['frac_err'] / params['frac']
            rel_err_r0 = r0_MeV_err / r0_MeV
            
            L_prime_err = abs(L_prime) * np.sqrt(rel_err_frac**2 + rel_err_r0**2)

            # Apply the combined error for the Up/Down systematic fits
            L_prime_up = L_prime + L_prime_err
            L_prime_down = L_prime - L_prime_err

            # Central Fit
            popt_central, pcov_central = curve_fit(
                lambda x_data, y0: model_func(x_data, y0, L_prime), 
                x_arr, y_arr, sigma=effective_y_err(L_prime), absolute_sigma=True
            )
            y_0_opt = popt_central[0]
            y_0_stat_err = np.sqrt(np.diag(pcov_central))[0]

            # GOODNESS OF FIT CALCULATION
            residuals = y_arr - model_func(x_arr, y_0_opt, L_prime)
            chi_sq = np.sum((residuals / effective_y_err(L_prime))**2)
            dof = len(x_arr) - 1 
            red_chi_sq = chi_sq / dof if dof > 0 else np.nan
            p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan

            # Upper Fit 
            popt_up, _ = curve_fit(
                lambda x_data, y0: model_func(x_data, y0, L_prime_up), 
                x_arr, y_arr, sigma=effective_y_err(L_prime_up), absolute_sigma=True
            )
            
            # Lower Fit 
            popt_down, _ = curve_fit(
                lambda x_data, y0: model_func(x_data, y0, L_prime_down), 
                x_arr, y_arr, sigma=effective_y_err(L_prime_down), absolute_sigma=True
            )

            y_0_sys_err = max(abs(y_0_opt - popt_up[0]), abs(y_0_opt - popt_down[0]))
            y_0_total_err = np.sqrt(y_0_stat_err**2 + y_0_sys_err**2)

            results.append({
                'Ensemble': group,
                'Data_Type': prefix,
                'Branch': branch_name,
                'y_0_central': y_0_opt,
                'y_0_stat_err': y_0_stat_err,
                'y_0_sys_err_lamb': y_0_sys_err,
                'y_0_total_err': y_0_total_err,
                'chi_sq': chi_sq,             
                'red_chi_sq': red_chi_sq,     
                'p_value': p_value,
                'y_0_up': popt_up[0],
                'y_0_down': popt_down[0],
                'L_prime_central': L_prime,
                'L_prime_up': L_prime_up,
                'L_prime_down': L_prime_down
            })

results_df = pd.DataFrame(results).sort_values(by=['Ensemble', 'Data_Type', 'Branch'])
results_df.to_csv(filename, index=False)

print("\n--- DIMENSIONLESS FITTING COMPLETE ---")
print("Results successfully saved to", filename)
print("\nSummary of Extracted Parameters (y_0 = r0^2 * sigma_0):")
print(results_df[['Ensemble', 'Data_Type', 'Branch', 'y_0_central', 'y_0_total_err', 'red_chi_sq', 'p_value']].to_string(index=False))