# Reads data_fran.csv and fit_results_log_fran.csv, 
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
# 2. Chiral Model for Evaluation
# =============================================================================
def chiral_eval(ml, sigma_0, lamb_prime, c_pipi):
    term1 = 4 * lamb_prime * ml
    log_arg = np.where((2 * B0_MeV * ml) / (mu_MeV**2) > 0, (2 * B0_MeV * ml) / (mu_MeV**2), 1e-10)
    term2_coeff = (3 * B0_MeV) / (4 * np.pi**2 * f_pi_MeV**2)
    term2 = (term2_coeff * (4 * B0_MeV * c_pipi - lamb_prime) * ( np.log(log_arg) - 1) ) * (ml**2)
    return sigma_0 - term1 - term2

# =============================================================================
# 3. Load Data & Calculate Physical Units
# =============================================================================
# Experimental data
df = pd.read_csv(data_file, sep=r'\s+')

# Data points for plotting
m_l_data = df['mu_l'].values
sigma_data = df['sigma_t0'].values / t0_MeV_inv2

# Propagate errors for sigma
rel_err_sigma_t0 = df['statistical_error_of_sigma_t0'].values / df['sigma_t0'].values
rel_err_t0 = 2 * (sqrt_t0_err_fm / sqrt_t0_fm) 
sigma_err = sigma_data * np.sqrt(rel_err_sigma_t0**2 + rel_err_t0**2)

# Fit data
fit_df = pd.read_csv(filename)

# =============================================================================
# 4. Plotting Logic & Aesthetics
# =============================================================================
colors = ['orange', 'blue', 'green', 'red']
data_color = '#1f1f1f' 

# Custom Legend Mapping
branch_labels = {
    'Branch_1': r"Set 2 $\lambda^\prime < 0$",
    'Branch_2': r"Set 1 $\lambda^\prime < 0$",
    'Branch_3': r"Set 1 $\lambda^\prime > 0$",
    'Branch_4': r"Set 2 $\lambda^\prime > 0$"
}

# Desired Legend Order
legend_order = ['Branch_3', 'Branch_2', 'Branch_4', 'Branch_1']

branch_colors = {
    'Branch_3': colors[0],
    'Branch_2': colors[1],
    'Branch_4': colors[2],
    'Branch_1': colors[3]
}

# Scientific notation formatting
def fmt_sci(val):
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

# Prepare Figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Chiral Log Fits', fontsize=16)

# X-axis plotting bounds
x_min, x_max = np.min(m_l_data), np.max(m_l_data)
x_vals_plot = np.linspace(x_min - (x_max-x_min)*0.2, x_max + (x_max-x_min)*0.2, 200)

# Plot Experimental Data (No xerr since mu_l is exact)
ax.errorbar(m_l_data, sigma_data, yerr=sigma_err, 
            fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

handles_dict = {}
labels_dict = {}

# Plot the 4 Fit Branches
for _, fit_row in fit_df.iterrows():
    branch = fit_row['Branch']
    c = branch_colors.get(branch, 'black')
    
    y_cen = chiral_eval(x_vals_plot, fit_row['sigma_0_central'], 
                        fit_row['lamb_prime_central'], fit_row['c_pipi_central'])
    
    y_up = chiral_eval(x_vals_plot, fit_row['sigma_0_up'], 
                       fit_row['lamb_prime_up'], fit_row['c_pipi_up'])
    y_down = chiral_eval(x_vals_plot, fit_row['sigma_0_down'], 
                         fit_row['lamb_prime_down'], fit_row['c_pipi_down'])
    
    # Calculate total error band
    y_sys_err = np.maximum(np.abs(y_cen - y_up), np.abs(y_cen - y_down))
    y_total_err = np.sqrt(y_sys_err**2 + fit_row['sigma_0_stat_err']**2)
    
    eq_label = rf"{branch_labels[branch]} ($\sigma_0 = {fmt_sci(fit_row['sigma_0_central'])}$)"
    
    line, = ax.plot(x_vals_plot, y_cen, linestyle='--', color=c)
    ax.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=c, alpha=0.25)
    
    handles_dict[branch] = line
    labels_dict[branch] = eq_label

# Axes Aesthetics
ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
ax.set_xlabel(r'$\mu_l$', fontsize=14)
ax.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)

# Custom Legend Logic
handles, labels = ax.get_legend_handles_labels()
data_handles = [h for h, l in zip(handles, labels) if l == 'Data']

ordered_handles = [data_handles[0]] if data_handles else []
ordered_labels = ['Data'] if data_handles else []

for b in legend_order:
    if b in handles_dict:
        ordered_handles.append(handles_dict[b])
        ordered_labels.append(labels_dict[b])

ax.legend(ordered_handles, ordered_labels, fontsize=10, loc='best')

# Save Plot
fig.tight_layout()
filename_out = "Plot_Log_fran.svg"
fig.savefig(filename_out, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved successfully as '{filename_out}'.")