import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
# 1. Read the data into a pandas DataFrame from the existing CSV file
# Ensure 'data_fran.csv' is in the same directory as this script
df = pd.read_csv('data_fran.csv')

# 3. Define constants and calculate sigma & uncertainties
# sqrt(t_0) = 0.1443 fm, with uncertainties 0.0007 and 0.0013
sqrt_t0 = 0.1443
err_sqrt_t0 = np.sqrt(0.0007**2 + 0.0013**2)

# Calculate t0 and its propagated error
t0 = sqrt_t0**2
err_t0 = 2 * sqrt_t0 * err_sqrt_t0

# Calculate sigma = (\sigma*t_0) / t0
df['sigma'] = df['\sigma*t_0'] / t0

# Error propagation for sigma = (\sigma*t_0) / t0
rel_err_num = df['statistical error of \sigma*t_0'] / df['\sigma*t_0']
rel_err_den = err_t0 / t0

# Final absolute error for sigma
df['sigma_err'] = df['sigma'] * np.sqrt(rel_err_num**2 + rel_err_den**2)

# 4. Create Scatter plot with error bars
plt.figure(figsize=(8, 6))
plt.errorbar(df['\mu_l'], df['sigma'], yerr=df['sigma_err'], fmt='o', 
             capsize=5, markerfacecolor='b', markeredgecolor='b', ecolor='r', 
             markersize=6, label=r'$\sigma$ values')

plt.xlabel(r'$\mu_l$', fontsize=14)
plt.ylabel(r'$\sigma$ (fm$^{-2}$)', fontsize=14)
plt.title(r'Scatter plot of $\sigma$ vs $\mu_l$', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('sigma_vs_m_l_fran.svg')