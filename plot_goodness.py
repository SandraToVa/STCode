# Script to plot a figure with the goodness of the different fits
# (linear vs logarithmic) for all branches and ensembles, 
# using the reduced chi-square values.

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
A_latex = ['r_0','\pi/12']
# Change index to select r0 or pi12 data
A_index = 1
# Global?
glob = True

if glob == True:
    # Load the fit results
    df_lin = pd.read_csv(f'fit_results_linear_global_{A_types[A_index]}.csv')
    df_log = pd.read_csv(f'fit_results_log_global_{A_types[A_index]}.csv')
    df_ST = pd.read_csv(f'fit_results_ST_global_{A_types[A_index]}.csv')
    #When global fit is performed
    pivot_lin = df_lin.pivot(index='Data_Type', columns='Branch', values='p_value')
    pivot_log = df_log.pivot(index='Data_Type', columns='Branch', values='p_value')
    pivot_ST = df_ST.set_index('Data_Type')[['p_value']]
if glob == False:
    # Load the fit results
    df_lin = pd.read_csv(f'fit_results_linear_{A_types[A_index]}.csv')
    df_log = pd.read_csv(f'fit_results_log_{A_types[A_index]}.csv')
    df_ST = pd.read_csv(f'fit_results_ST_{A_types[A_index]}.csv')
    # Create a combined configuration label (Ensemble + Data Type)
    #When separated by ensambles
    df_lin['Config'] = df_lin['Ensemble'] + " (" + df_lin['Data_Type'] + ")"
    df_log['Config'] = df_log['Ensemble'] + " (" + df_log['Data_Type'] + ")"
    df_ST['Config'] = df_ST['Ensemble'] + " (" + df_ST['Data_Type'] + ")"

    pivot_lin = df_lin.pivot(index='Config', columns='Branch', values='p_value')
    pivot_log = df_log.pivot(index='Config', columns='Branch', values='p_value')
    pivot_ST = df_ST.pivot(index='Ensemble', columns='Data_Type', values='p_value')

# ==========================================
# 1. Linear Model Figure
# ==========================================
fig_lin, ax_lin = plt.subplots(figsize=(8, 6))

sns.heatmap(
    pivot_lin, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    ax=ax_lin, 
    cbar_kws={'label': f'P-values'}, 
    vmin=0, 
    vmax=1.5
)
if glob == True:
    ax_lin.set_title(f'Global Linear Model: p-value ($A=A_{{{A_latex[A_index]}}}$)', fontsize=16)
    ax_lin.set_ylabel('Data Type', fontsize=14)
    ax_lin.set_xlabel('Branch', fontsize=12)

    fig_lin.tight_layout()
    fig_lin.savefig(f'P_value_Linear_Global_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_lin)
    print(f"Saved 'P_value_Linear_Global_{A_types[A_index]}.svg'")
if glob == False:
    ax_lin.set_title(f'Linear Model: p-value ($A=A_{{{A_latex[A_index]}}}$)', fontsize=16)
    ax_lin.set_ylabel('Ensemble (Data Type)', fontsize=14)
    ax_lin.set_xlabel('Branch', fontsize=12)

    fig_lin.tight_layout()
    fig_lin.savefig(f'P_value_Linear_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_lin)
    print(f"Saved 'P_value_Linear_{A_types[A_index]}.svg'")


# ==========================================
# 2. Extended Logarithmic Model Figure
# ==========================================
fig_log, ax_log = plt.subplots(figsize=(8, 6))

sns.heatmap(
    pivot_log, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    ax=ax_log, 
    cbar_kws={'label': f'P-values'}, 
    vmin=0, 
    vmax=1.5
)
if glob == True:
    ax_log.set_title(f'Global Extended Log Model: p-value ($A=A_{{{A_latex[A_index]}}}$)', fontsize=16)
    ax_log.set_ylabel('Data Type', fontsize=14)
    ax_log.set_xlabel('Branch', fontsize=12)

    fig_log.tight_layout()
    fig_log.savefig(f'P_value_Log_Global_v2_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_log)
    print(f"Saved 'P_value_Log_Global_v2_{A_types[A_index]}.svg'")
if glob == False:
    ax_log.set_title(f'Extended Log Model: p-value ($A=A_{{{A_latex[A_index]}}}$)', fontsize=16)
    ax_log.set_ylabel('Ensemble (Data Type)', fontsize=14)
    ax_log.set_xlabel('Branch', fontsize=12)

    fig_log.tight_layout()
    fig_log.savefig(f'P_value_Log_v2_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_log)
    print(f"Saved 'P_value_Log_v2_{A_types[A_index]}.svg'")

# ==========================================
# 3. Soto-Tarrús model
# ==========================================
fig_ST, ax_ST = plt.subplots(figsize=(8, 6))

sns.heatmap(
    pivot_ST, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    ax=ax_ST, 
    cbar_kws={'label': f'P-values'}, 
    vmin=0, 
    vmax=1.5
)
if glob == True:
    ax_ST.set_title(f'Global ST Model: p-value ($A=A_{{{A_latex[A_index]}}}$)', fontsize=16)
    ax_ST.set_ylabel('Data Type', fontsize=14)
    ax_ST.set_xlabel('p-value', fontsize=14)
    ax_ST.set_xticklabels(ax_ST.get_xticklabels(), rotation=0)
    ax_ST.set_yticklabels(ax_ST.get_yticklabels(), rotation=0)

    fig_ST.tight_layout()
    fig_ST.savefig(f'P_value_ST_Global_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_ST)
    print(f"Saved 'P_value_ST_Global_{A_types[A_index]}.svg'")
if glob == False:
    ax_ST.set_title(f'ST Model: p-value ($A=A_{{{A_latex[A_index]}}}$)', fontsize=16)
    ax_ST.set_ylabel('Ensemble', fontsize=14)
    ax_ST.set_xlabel('Data Type', fontsize=14)

    fig_ST.tight_layout()
    fig_ST.savefig(f'P_value_ST_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_ST)
    print(f"Saved 'P_value_ST_{A_types[A_index]}.svg'")