import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
# 1. Load the data (Using sep='\s+' to correctly parse the spaces)
df = pd.read_csv('data_bram.csv', sep=r'\s+')

# Fixed constant
r0 = 0.475 # fm

# 2. Define a function to calculate physical values and errors
def calculate_physics(df_subset, prefix):
    """
    prefix should be 'bare' or 'smeared' to read the correct columns.
    Returns ml, ml_err, sigma, sigma_err
    """
    r0_a = df_subset[f'r0_a_{prefix}']
    r0_a_err = df_subset[f'r0_a_{prefix}_err']
    a2sigma = df_subset[f'a2sigma_{prefix}']
    a2sigma_err = df_subset[f'a2sigma_{prefix}_err']
    am_l = df_subset['am_l'] 
    
    # Calculate lattice spacing 'a'
    a = r0 / r0_a
    a_err = a * (r0_a_err / r0_a)
    
    # Calculate physical light quark mass m_l
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    
    # Calculate physical string tension sigma
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
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Define the groups and their corresponding titles based on Table I
groups = ['M i', 'M ii', 'M iii']
titles = ['M i (Physical Ratio)', 'M ii (1/10 Physical Ratio)', 'M iii (1/5 Physical Ratio)']

# 5. Loop through each group and plot it on its respective axis
for i, (group_name, title) in enumerate(zip(groups, titles)):
    # Filter the data for the current group
    subset = df[df['Group'] == group_name]
    
    # Calculate values
    ml_bare, ml_bare_err, sigma_bare, sigma_bare_err = calculate_physics(subset, 'bare')
    ml_smeared, ml_smeared_err, sigma_smeared, sigma_smeared_err = calculate_physics(subset, 'smeared')
    
    ax = axes[i]
    
    # Plot Bare Links
    ax.errorbar(ml_bare, sigma_bare, 
                xerr=ml_bare_err, yerr=sigma_bare_err, 
                fmt='o', label='Bare Links', capsize=4, alpha=0.8)
    
    # Plot Smeared Links
    ax.errorbar(ml_smeared, sigma_smeared, 
                xerr=ml_smeared_err, yerr=sigma_smeared_err, 
                fmt='s', label='Smeared Links', capsize=4, alpha=0.8)
    
    # Subplot Formatting
    ax.set_title(title)
    ax.set_xlabel(r'$m_l$ (fm$^{-1}$)')
    
    # Only put the Y-axis label on the leftmost plot to avoid clutter
    if i == 0:
        ax.set_ylabel(r'$\sigma$ (fm$^{-2}$)')
        
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

# Main title for the entire figure
fig.suptitle(r'String Tension ($\sigma$) vs Light Quark Mass ($m_l$) across Ensembles', fontsize=16)

plt.tight_layout()
plt.show()