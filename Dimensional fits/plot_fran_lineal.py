# Reads data_fran.csv and fit_results_lineal_fran.csv, 
# and generates a single publication-ready figure.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
# 1. Keep text editable in SVG
mpl.rcParams['svg.fonttype'] = 'none'

# 2. Use a professional Serif/Times font environment
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # Matches the math rendering to the text font
# --- Constants & Error Propagation ---
hbar_c = 197.3269804
B0_MeV = 2700
# t0 parameters
sqrt_t0_fm = 0.1443
sqrt_t0_err_fm = np.sqrt(0.0007**2 + 0.0013**2) 

# Convert t0 to MeV^-1
sqrt_t0_MeV_inv = sqrt_t0_fm / hbar_c
t0_MeV_inv2 = sqrt_t0_MeV_inv**2


data_file = 'data_fran.csv'
filename = 'fit_results_lineal_fran.csv'

# --- Load Data ---
df = pd.read_csv(data_file, sep=r'\s+')
fit_df = pd.read_csv(filename)

# --- Calculate Data Points for Plotting ---
x_data_plot = df['mu_l'].values

# Calculate sigma in physical units (MeV^2)
sigma_arr = df['sigma_t0'].values / t0_MeV_inv2

# Propagate errors for sigma
rel_err_sigma_t0 = df['statistical_error_of_sigma_t0'].values / df['sigma_t0'].values
rel_err_t0 = 2 * (sqrt_t0_err_fm / sqrt_t0_fm)
sigma_err_arr = sigma_arr * np.sqrt(rel_err_sigma_t0**2 + rel_err_t0**2)

# --- Aesthetic Styles ---
branch_styles = {
    'positive': {'color': 'orange'}, 
    'negative': {'color': 'blue'} 
}
data_color = '#1f1f1f' 

# Custom Legend Mapping
branch_labels = {
    'positive': r"Set 1 & 2 $\lambda^\prime > 0$",
    'negative': r"Set 1 & 2 $\lambda^\prime < 0$"
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('String Tension Fits', fontsize=16)

# X-axis range for the fit lines (extending 20% beyond the data points)
x_min, x_max = min(x_data_plot), max(x_data_plot)
x_vals_plot = np.linspace(x_min - (x_max - x_min) * 0.2, x_max + (x_max - x_min) * 0.2, 100)

# 1. Plot the Experimental Data Points 
# Note: xerr is removed since mu_l has no error
ax.errorbar(x_data_plot, sigma_arr, yerr=sigma_err_arr, 
            fmt='o', color=data_color, alpha=0.9, 
            label='Data', capsize=4, zorder=5) 

# 2. Plot Both Branches
for branch in ['positive', 'negative']:
    fit_row = fit_df[fit_df['Branch'] == branch]
    
    if not fit_row.empty:
        s0_cen = fit_row.iloc[0]['sigma_0_central']
        s0_stat_err = fit_row.iloc[0]['sigma_0_stat_err']
        s0_up = fit_row.iloc[0]['sigma_0_up']
        s0_down = fit_row.iloc[0]['sigma_0_down']
        
        lp_cen = fit_row.iloc[0]['lamb_prime_central']
        lp_up = fit_row.iloc[0]['lamb_prime_up']
        lp_down = fit_row.iloc[0]['lamb_prime_down']
        
        y_cen = s0_cen - 4 * lp_cen * x_vals_plot
        y_up_sys = s0_up - 4 * lp_up * x_vals_plot
        y_down_sys = s0_down - 4 * lp_down * x_vals_plot
        
        y_sys_err = np.maximum(np.abs(y_cen - y_up_sys), np.abs(y_cen - y_down_sys))
        y_total_err = np.sqrt(y_sys_err**2 + s0_stat_err**2)
        
        c = branch_styles[branch]['color']
        
        # Create the dynamic equation label using scientific notation
        def fmt_sci(val):
            base, exp = "{:.2e}".format(val).split('e')
            return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"
        
        eq_label = rf"{branch_labels[branch]} ($\sigma_0 = {fmt_sci(s0_cen)}$)"
        
        ax.plot(x_vals_plot, y_cen, linestyle='--', color=c, label=eq_label)
        ax.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=c, alpha=0.25)

# Aesthetics
ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
ax.set_xlabel(r'$\mu_l$', fontsize=14)
ax.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
ax.legend(fontsize=10, loc='best')

fig.tight_layout()
filename_out = 'Plot_Lineal_fran.svg'
fig.savefig(filename_out, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved successfully as {filename_out}")