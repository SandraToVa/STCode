import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
# 1. Load the data
df = pd.read_csv('data_bram.csv', sep=r'\s+')

# Constants
r0 = 0.475 # fm
hbar_c = 0.1973269804 # GeV * fm

# Calculate Lambda in fm^-1
B0_GeV = 2.7 # GeV
B0_fm = B0_GeV / hbar_c # Convert to fm^-1

lambda_val_positive = 0.124 * 2 * B0_fm  # Positive branch
lambda_val_negative = -0.124 * 2 * B0_fm # Negative branch

# 2. Define a function to calculate physical values and errors
def calculate_physics(df_subset, prefix):
    r0_a = df_subset[f'r0_a_{prefix}']
    r0_a_err = df_subset[f'r0_a_{prefix}_err']
    a2sigma = df_subset[f'a2sigma_{prefix}']
    a2sigma_err = df_subset[f'a2sigma_{prefix}_err']
    am_l = df_subset['am_l'] 
    
    a = r0 / r0_a
    a_err = a * (r0_a_err / r0_a)
    
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    
    sigma = a2sigma / (a**2)
    relative_a2sigma_err = a2sigma_err / a2sigma
    relative_a_err = a_err / a
    sigma_err = sigma * np.sqrt(relative_a2sigma_err**2 + (2 * relative_a_err)**2)
    
    return m_l, m_l_err, sigma, sigma_err

# 3. Create a new column to group the ensembles
def assign_group(name):
    if name.endswith('M_iii'): return 'M iii'
    elif name.endswith('M_ii'): return 'M ii'
    elif name.endswith('M_i'): return 'M i'
    return 'Unknown'

df['Group'] = df['Ensemble'].apply(assign_group)

# 4. Set up the figure with 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

groups = ['M i', 'M ii', 'M iii']
titles = ['M i (Physical Ratio)', 'M ii (1/10 Physical Ratio)', 'M iii (1/5 Physical Ratio)']

# 5. Loop through each group and plot it on its respective axis
for i, (group_name, title) in enumerate(zip(groups, titles)):
    subset = df[df['Group'] == group_name]
    
    ml_bare, ml_bare_err, sigma_bare, sigma_bare_err = calculate_physics(subset, 'bare')
    ml_smeared, ml_smeared_err, sigma_smeared, sigma_smeared_err = calculate_physics(subset, 'smeared')
    
    ax = axes[i]
    
    # Plot Bare Links
    ax.errorbar(ml_bare, sigma_bare, 
                xerr=ml_bare_err, yerr=sigma_bare_err, 
                fmt='o', label='Bare Links', capsize=4, alpha=0.8, color='blue')
    
    # Plot Smeared Links
    ax.errorbar(ml_smeared, sigma_smeared, 
                xerr=ml_smeared_err, yerr=sigma_smeared_err, 
                fmt='s', label='Smeared Links', capsize=4, alpha=0.8, color='orange')
    
    # --- GET LIMITS INCLUDING ERROR BARS ---
    # Find absolute mins and maxes by adding/subtracting the error bars
    ml_min_err = min((ml_bare - ml_bare_err).min(), (ml_smeared - ml_smeared_err).min())
    ml_max_err = max((ml_bare + ml_bare_err).max(), (ml_smeared + ml_smeared_err).max())
    
    sigma_min_err = min((sigma_bare - sigma_bare_err).min(), (sigma_smeared - sigma_smeared_err).min())
    sigma_max_err = max((sigma_bare + sigma_bare_err).max(), (sigma_smeared + sigma_smeared_err).max())
    
    # Add a gentle 10% visual buffer outside the very tips of the error bars
    x_buffer = (ml_max_err - ml_min_err) * 0.10
    if x_buffer == 0: x_buffer = ml_min_err * 0.10 
    x_lower = ml_min_err - x_buffer
    x_upper = ml_max_err + x_buffer
    
    y_buffer = (sigma_max_err - sigma_min_err) * 0.10
    if y_buffer == 0: y_buffer = sigma_min_err * 0.10
    y_lower = sigma_min_err - y_buffer
    y_upper = sigma_max_err + y_buffer

    x_vals = np.array([x_lower, x_upper])
    
    # --- THEORETICAL FUNCTIONS ---
    # Extract sigma_0 (using the first bare link value in the current subset)
    sigma_0 = sigma_bare.iloc[0]
    
    # Calculate the functions
    sigma_func_pos = sigma_0 - 4 * lambda_val_positive * x_vals
    sigma_func_neg = sigma_0 - 4 * lambda_val_negative * x_vals
    
    # Plot the lines
    ax.plot(x_vals, sigma_func_pos, 'r--', label=r'$\sigma = \sigma_0 - 4\lambda m_l$ ($\lambda > 0$)')
    ax.plot(x_vals, sigma_func_neg, 'g--', label=r'$\sigma = \sigma_0 - 4\lambda m_l$ ($\lambda < 0$)')
    
    # Apply the perfectly zoomed limits
    ax.set_xlim(left=x_lower, right=x_upper)
    ax.set_ylim(bottom=y_lower, top=y_upper)

    # Subplot Formatting
    ax.set_title(title)
    ax.set_xlabel(r'$m_l$ (fm$^{-1}$)')
    ax.set_ylabel(r'$\sigma$ (fm$^{-2}$)')
        
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

fig.suptitle(r'String Tension ($\sigma$) vs Light Quark Mass ($m_l$)', fontsize=16)

plt.tight_layout()
plt.show()

