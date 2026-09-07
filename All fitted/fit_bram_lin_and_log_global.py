# fit_bram_lin_and_log_global.py
# Reads data_bram.csv,
# computes dimensionless variables directly from lattice data across ALL ensembles simultaneously,
# performs a strictly GLOBAL fit to find universal dimensionless parameters (y_0, L', L2, c_pipi)
# using either a linear or logarithmic model, depending on user selection.
# Propagates statistical and systematic uncertainties via Monte Carlo resampling of raw lattice data.

import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2

# =============================================================================
# 1. Constants & Configuration
# =============================================================================

# Fix seed for reproducibility
np.random.seed(42)

hbar_c = 197.3269804  # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV_central = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200

A_types = ['Ar0', 'Api12']
fit_types = ['linear', 'logarithmic']

# Selection Settings
A_index = 0  # 0 for Ar0, 1 for Api12
fit_index = 0  # 0 for linear global fit, 1 for logarithmic global fit

N_MC = 1000  # Number of Monte Carlo iterations for error propagation

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_full_global_{fit_types[fit_index]}_{A_types[A_index]}.csv'


# =============================================================================
# 2. Data Processing & Chiral Models
# =============================================================================
def calculate_dimensionless(df_row, prefix):
    """Computes dimensionless variables x and y and propagates their errors

    directly from lattice measurements.
    """
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']

    y = a2sigma * (r0_a**2)
    y_err = y * np.sqrt(
        (a2sigma_err / a2sigma) ** 2 + (2 * r0_a_err / r0_a) ** 2
    )

    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)

    return x, x_err, y, y_err


def linear_model_dim(x, y_0, L_prime):
    """Linear ChPT model: y = y_0 - 4 * L' * x"""
    return y_0 - 4 * L_prime * x


def logarithmic_model_dim(
    x, y_0, L2, L_prime, c_pipi, B0_dim, f_pi_dim, mu_dim
):
    """Logarithmic ChPT model containing linear, chiral log, and analytical L'' terms."""
    term1 = 4 * L_prime * x

    log_arg = (2 * B0_dim * x) / (mu_dim**2)
    log_arg = np.where(log_arg > 0, log_arg, 1e-10)

    term2_coeff = (3 * B0_dim) / (4 * np.pi**2 * f_pi_dim**2)
    term2 = (
        term2_coeff
        * (4 * B0_dim * c_pipi - L_prime)
        * (np.log(log_arg) - 1)
        * (x**2)
    )

    term3 = 2 * L2 * (x**2)

    return y_0 - term1 - term2 - term3


# =============================================================================
# 3. Data Loading
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')
data_types = ['bare', 'smeared']
fit_results = []

# =============================================================================
# 4. Global Fitting Routine across All Ensembles
# =============================================================================
for dtype in data_types:
    # Compute central dimensionless quantities for ALL data points across ensembles
    dim_data = df.apply(lambda row: calculate_dimensionless(row, dtype), axis=1)
    x_data = np.array([d[0] for d in dim_data])
    x_err = np.array([d[1] for d in dim_data])
    y_data = np.array([d[2] for d in dim_data])
    y_err = np.array([d[3] for d in dim_data])

    # Central physical dimensionless constants
    B0_dim_central = B0_MeV * r0_MeV_central
    f_pi_dim_central = f_pi_MeV * r0_MeV_central
    mu_dim_central = mu_MeV * r0_MeV_central

    if fit_index == 0:
        # ======================== GLOBAL LINEAR FIT ========================
        try:
            popt_central, _ = curve_fit(
                linear_model_dim,
                x_data,
                y_data,
                p0=[0.05, 0.001],
                sigma=y_err,
                absolute_sigma=True,
            )
            y0_central, L_prime_central = popt_central

            # Goodness of fit calculation
            y_fit = linear_model_dim(x_data, y0_central, L_prime_central)
            chi_sq = np.sum(((y_data - y_fit) / y_err) ** 2)
            dof = len(x_data) - 2  # 2 fitted parameters: y_0, L'
            red_chi_sq = chi_sq / dof if dof > 0 else np.nan
            p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
        except Exception as e:
            print(f'Global Linear fit failed for {dtype}: {e}')
            continue

        # MC Loop for uncertainties (Resampling raw lattice observables)
        y0_stat_list, L_prime_stat_list = [], []

        for _ in range(N_MC):
            x_mc_list, y_mc_list = [], []
            for _, row in df.iterrows():
                r0_a_s = np.random.normal(
                    row[f'r0_a_{dtype}'], row[f'r0_a_{dtype}_err']
                )
                a2sigma_s = np.random.normal(
                    row[f'a2sigma_{dtype}'], row[f'a2sigma_{dtype}_err']
                )

                x_mc_list.append(row['am_l'] * r0_a_s)
                y_mc_list.append(a2sigma_s * (r0_a_s**2))

            x_mc, y_mc = np.array(x_mc_list), np.array(y_mc_list)

            try:
                p_stat, _ = curve_fit(
                    linear_model_dim,
                    x_mc,
                    y_mc,
                    p0=popt_central,
                    sigma=y_err,
                    absolute_sigma=True,
                )
                y0_stat_list.append(p_stat[0])
                L_prime_stat_list.append(p_stat[1])
            except Exception:
                continue

        fit_results.append({
            'Data_Type': dtype,
            'y0_central': y0_central,
            'L_prime_central': L_prime_central,
            'chi_sq': chi_sq,
            'dof': dof,
            'red_chi_sq': red_chi_sq,
            'p_value': p_value,
            'y0_stat_err': (
                np.std(y0_stat_list) if y0_stat_list else np.nan
            ),
            'y0_sys_err': 0.0,  # No explicit r0 physical scale dependency in linear dimensionless model
            'y0_total_err': (
                np.std(y0_stat_list) if y0_stat_list else np.nan
            ),
            'L_prime_stat_err': (
                np.std(L_prime_stat_list) if L_prime_stat_list else np.nan
            ),
            'L_prime_sys_err': 0.0,
            'L_prime_total_err': (
                np.std(L_prime_stat_list) if L_prime_stat_list else np.nan
            ),
            'y0_tot_list': json.dumps(y0_stat_list),
            'L_prime_tot_list': json.dumps(L_prime_stat_list),
        })

    elif fit_index == 1:
        # ===================== GLOBAL LOGARITHMIC FIT =====================
        try:
            p0_log = [0.05, 0.001, 0.001, 0.001]
            popt_central, _ = curve_fit(
                lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(
                    x_val,
                    y0,
                    l2,
                    l1,
                    cpi,
                    B0_dim_central,
                    f_pi_dim_central,
                    mu_dim_central,
                ),
                x_data,
                y_data,
                p0=p0_log,
                sigma=y_err,
                absolute_sigma=True,
                maxfev=10000,
            )
            y0_central, L2_central, L_prime_central, c_pipi_central = (
                popt_central
            )

            # Goodness of fit calculation
            y_fit = logarithmic_model_dim(
                x_data,
                y0_central,
                L2_central,
                L_prime_central,
                c_pipi_central,
                B0_dim_central,
                f_pi_dim_central,
                mu_dim_central,
            )
            chi_sq = np.sum(((y_data - y_fit) / y_err) ** 2)
            dof = len(x_data) - 4  # 4 fitted parameters
            red_chi_sq = chi_sq / dof if dof > 0 else np.nan
            p_value = chi2.sf(chi_sq, dof) if dof > 0 else np.nan
        except Exception as e:
            print(f'Global Logarithmic fit failed for {dtype}: {e}')
            continue

        # MC Storage Lists
        y0_stat_list, y0_sys_list, y0_tot_list = [], [], []
        L_prime_stat_list, L_prime_sys_list, L_prime_tot_list = [], [], []
        L2_stat_list, L2_sys_list, L2_tot_list = [], [], []
        c_pipi_stat_list, c_pipi_sys_list, c_pipi_tot_list = [], [], []

        for _ in range(N_MC):
            # Resample raw lattice quantities to preserve x-y correlations
            x_mc_list, y_mc_list = [], []
            for _, row in df.iterrows():
                r0_a_s = np.random.normal(
                    row[f'r0_a_{dtype}'], row[f'r0_a_{dtype}_err']
                )
                a2sigma_s = np.random.normal(
                    row[f'a2sigma_{dtype}'], row[f'a2sigma_{dtype}_err']
                )

                x_mc_list.append(row['am_l'] * r0_a_s)
                y_mc_list.append(a2sigma_s * (r0_a_s**2))

            x_mc, y_mc = np.array(x_mc_list), np.array(y_mc_list)

            # Physical constant variation (systematic scale error source)
            r0_MeV_mc = np.random.normal(r0_fm, r0_fm_err) / hbar_c
            B0_dim_mc = B0_MeV * r0_MeV_mc
            f_pi_dim_mc = f_pi_MeV * r0_MeV_mc
            mu_dim_mc = mu_MeV * r0_MeV_mc

            # 1. STATISTICAL: Vary lattice data, keep central physical scale
            try:
                p_stat, _ = curve_fit(
                    lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(
                        x_val,
                        y0,
                        l2,
                        l1,
                        cpi,
                        B0_dim_central,
                        f_pi_dim_central,
                        mu_dim_central,
                    ),
                    x_mc,
                    y_mc,
                    p0=popt_central,
                    sigma=y_err,
                    absolute_sigma=True,
                    maxfev=5000,
                )
                y0_stat_list.append(p_stat[0])
                L2_stat_list.append(p_stat[1])
                L_prime_stat_list.append(p_stat[2])
                c_pipi_stat_list.append(p_stat[3])
            except Exception:
                pass

            # 2. SYSTEMATIC: Keep central lattice data, vary physical scale r0
            try:
                p_sys, _ = curve_fit(
                    lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(
                        x_val,
                        y0,
                        l2,
                        l1,
                        cpi,
                        B0_dim_mc,
                        f_pi_dim_mc,
                        mu_dim_mc,
                    ),
                    x_data,
                    y_data,
                    p0=popt_central,
                    sigma=y_err,
                    absolute_sigma=True,
                    maxfev=5000,
                )
                y0_sys_list.append(p_sys[0])
                L2_sys_list.append(p_sys[1])
                L_prime_sys_list.append(p_sys[2])
                c_pipi_sys_list.append(p_sys[3])
            except Exception:
                pass

            # 3. TOTAL: Vary both lattice data and physical scale r0
            try:
                p_tot, _ = curve_fit(
                    lambda x_val, y0, l2, l1, cpi: logarithmic_model_dim(
                        x_val,
                        y0,
                        l2,
                        l1,
                        cpi,
                        B0_dim_mc,
                        f_pi_dim_mc,
                        mu_dim_mc,
                    ),
                    x_mc,
                    y_mc,
                    p0=popt_central,
                    sigma=y_err,
                    absolute_sigma=True,
                    maxfev=5000,
                )
                y0_tot_list.append(p_tot[0])
                L2_tot_list.append(p_tot[1])
                L_prime_tot_list.append(p_tot[2])
                c_pipi_tot_list.append(p_tot[3])
            except Exception:
                pass

        fit_results.append({
            'Data_Type': dtype,
            'y0_central': y0_central,
            'L_prime_central': L_prime_central,
            'L2_central': L2_central,
            'c_pipi_central': c_pipi_central,
            'chi_sq': chi_sq,
            'dof': dof,
            'red_chi_sq': red_chi_sq,
            'p_value': p_value,
            'y0_stat_err': np.std(y0_stat_list) if y0_stat_list else np.nan,
            'y0_sys_err': np.std(y0_sys_list) if y0_sys_list else np.nan,
            'y0_total_err': np.std(y0_tot_list) if y0_tot_list else np.nan,
            'L_prime_stat_err': (
                np.std(L_prime_stat_list) if L_prime_stat_list else np.nan
            ),
            'L_prime_sys_err': (
                np.std(L_prime_sys_list) if L_prime_sys_list else np.nan
            ),
            'L_prime_total_err': (
                np.std(L_prime_tot_list) if L_prime_tot_list else np.nan
            ),
            'L2_stat_err': np.std(L2_stat_list) if L2_stat_list else np.nan,
            'L2_sys_err': np.std(L2_sys_list) if L2_sys_list else np.nan,
            'L2_total_err': np.std(L2_tot_list) if L2_tot_list else np.nan,
            'c_pipi_stat_err': (
                np.std(c_pipi_stat_list) if c_pipi_stat_list else np.nan
            ),
            'c_pipi_sys_err': (
                np.std(c_pipi_sys_list) if c_pipi_sys_list else np.nan
            ),
            'c_pipi_total_err': (
                np.std(c_pipi_tot_list) if c_pipi_tot_list else np.nan
            ),
            'y0_tot_list': json.dumps(y0_tot_list),
            'L_prime_tot_list': json.dumps(L_prime_tot_list),
            'L2_tot_list': json.dumps(L2_tot_list),
            'c_pipi_tot_list': json.dumps(c_pipi_tot_list),
        })

# =============================================================================
# 5. Output Summary & File Export
# =============================================================================
results_df = pd.DataFrame(fit_results)
if not results_df.empty:
    results_df.to_csv(filename, index=False)
    print(
        f'\n--- GLOBAL DIMENSIONLESS {fit_types[fit_index].upper()} FITTING'
        ' COMPLETE ---'
    )
    print(f"Results successfully saved to '{filename}'.")
    print('\nSummary of Extracted Global Parameters:')

    if fit_index == 0:
        cols_to_print = [
            'Data_Type',
            'y0_central',
            'y0_total_err',
            'L_prime_central',
            'L_prime_total_err',
            'chi_sq',
            'dof',
            'red_chi_sq',
            'p_value',
        ]
    else:
        cols_to_print = [
            'Data_Type',
            'y0_central',
            'y0_total_err',
            'L_prime_central',
            'L_prime_total_err',
            'L2_central',
            'L2_total_err',
            'c_pipi_central',
            'c_pipi_total_err',
            'chi_sq',
            'dof',
            'red_chi_sq',
            'p_value',
        ]

    print(results_df[cols_to_print].to_string(index=False))
else:
    print('\nNo fits were successful or dataset was empty.')