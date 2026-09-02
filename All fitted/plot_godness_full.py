# Script to plot a figure comparing the goodness of fit 
# (Full Linear vs Full Logarithmic models), 
# either globally or ensemble by ensemble.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 1. Keep text editable in SVG
mpl.rcParams['svg.fonttype'] = 'none'

# 2. Use a professional Serif/Times font environment
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # Matches the math rendering to the text font

A_types = ['Ar0', 'Api12']
A_latex = ['r_0', r'\pi/12']
# Change index to select r0 or pi12 data
A_index = 0
# Global? (True for Global, False for Ensemble by Ensemble)
glob = False

A_val = A_types[A_index]
A_tex = A_latex[A_index]

if glob:
    # Load the global fit results
    df_lin = pd.read_csv(f'fit_results_full_global_linear_{A_val}.csv')
    df_log = pd.read_csv(f'fit_results_full_global_logarithmic_{A_val}.csv')
    
    # Select relevant columns and rename p_value for the joint table
    df_lin = df_lin[['Data_Type', 'p_value']].rename(columns={'p_value': 'Full Linear'})
    df_log = df_log[['Data_Type', 'p_value']].rename(columns={'p_value': 'Full Logarithmic'})
    
    # Merge both dataframes based on Data_Type
    combined_data = pd.merge(df_lin, df_log, on='Data_Type', how='outer')
    combined_data.set_index('Data_Type', inplace=True)
    
    # Plot configuration
    title = f'Global Models: p-value ($A=A_{{{A_tex}}}$)'
    ylabel = 'Data Type'
    save_filename = f'P_value_Comparison_Global_{A_val}.svg'

else:
    # Load the ensemble fit results
    df_lin = pd.read_csv(f'fit_results_full_linear_{A_val}.csv')
    df_log = pd.read_csv(f'fit_results_full_logarithmic_{A_val}.csv')
    
    # Create a combined configuration label (Ensemble + Data Type)
    df_lin['Config'] = df_lin['Ensemble'] + " (" + df_lin['Data_Type'] + ")"
    df_log['Config'] = df_log['Ensemble'] + " (" + df_log['Data_Type'] + ")"
    
    # Select relevant columns and rename p_value for the joint table
    df_lin = df_lin[['Config', 'p_value']].rename(columns={'p_value': 'Full Linear'})
    df_log = df_log[['Config', 'p_value']].rename(columns={'p_value': 'Full Logarithmic'})
    
    # Merge both dataframes based on Config
    combined_data = pd.merge(df_lin, df_log, on='Config', how='outer')
    combined_data.set_index('Config', inplace=True)
    
    # Plot configuration
    title = f'Ensemble Models: p-value ($A=A_{{{A_tex}}}$)'
    ylabel = 'Ensemble (Data Type)'
    save_filename = f'P_value_Comparison_{A_val}.svg'

# ==========================================
# Figure Generation (Both Models Compared)
# ==========================================
# Adjust height dynamically in case there are many ensembles
fig_height = max(6, len(combined_data) * 0.5) 
fig, ax = plt.subplots(figsize=(8, fig_height))

sns.heatmap(
    combined_data, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    ax=ax, 
    cbar_kws={'label': 'P-values'}, 
    vmin=0, 
    vmax=1.5
)

ax.set_title(title, fontsize=16)
ax.set_ylabel(ylabel, fontsize=14)
ax.set_xlabel('Model', fontsize=14)

# Keep model names horizontal for better readability
plt.xticks(rotation=0)

fig.tight_layout()
fig.savefig(save_filename, bbox_inches='tight')
plt.close(fig)
print(f"Saved '{save_filename}'")