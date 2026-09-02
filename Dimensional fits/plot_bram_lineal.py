# Reads both data_bram.csv and fit_results_lineal.csv, 
# and generates the final publication-ready figures.

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
# --- Constants & Data Processing Function ---
hbar_c = 197.3269804
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\pi/12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_lineal_{A_types[A_index]}.csv'


def calculate_physics(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    a = r0_MeV / r0_a
    a_err = a * (r0_a_err / r0_a + r0_MeV_err / r0_MeV)  # Combine relative errors of r0_a and r0_MeV
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    sigma = a2sigma / (a**2)
    sigma_err = sigma * np.sqrt((a2sigma_err/a2sigma)**2 + (2*a_err/a)**2)
    return m_l, m_l_err, sigma, sigma_err

# --- Load Data ---
df = pd.read_csv(data_file, sep=r'\s+')
fit_df = pd.read_csv(filename)

df['mass_group'] = df['Ensemble'].apply(lambda x: "_".join(x.split('_')[-2:])) 
ensembles = df['mass_group'].unique()
data_types = ['bare', 'smeared']

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
for prefix in data_types:
    # Prepare Grouped Figure
    fig_grp, axes_grp = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig_grp.suptitle(f'String Tension Fits - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=18, fontweight='bold', y=1.02)
    
    for ax_grp, group in zip(axes_grp, ensembles):
        subset = df[df['mass_group'] == group]
        ax_grp.set_title(f'Ensemble: {group}', fontsize=16)
        
        # Prepare Individual Figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Ensemble: {group}', fontsize=16)
        
        m_l_list, m_l_err_list, sigma_list, sigma_err_list = [], [], [], []
        for _, row in subset.iterrows():
            ml, ml_e, sig, sig_e = calculate_physics(row, prefix)
            m_l_list.append(ml)
            m_l_err_list.append(ml_e)
            sigma_list.append(sig)
            sigma_err_list.append(sig_e)
            
        m_l_arr = np.array(m_l_list)
        m_l_err_arr = np.array(m_l_err_list)
        sigma_arr = np.array(sigma_list)
        sigma_err_arr = np.array(sigma_err_list)
        
        x_data_plot = m_l_arr
        x_err_plot = m_l_err_arr
            
        x_min, x_max = min(m_l_arr), max(m_l_arr)
        x_vals_math = np.linspace(x_min - (x_max-x_min)*0.2, x_max + (x_max-x_min)*0.2, 100)
        x_vals_plot = x_vals_math 
        
        # 1. Plot the Experimental Data Points on BOTH axes
        for axis in [ax, ax_grp]:
            axis.errorbar(x_data_plot, sigma_arr, xerr=x_err_plot, yerr=sigma_err_arr, 
                        fmt='o', color=data_color, alpha=0.9, 
                        label='Data', capsize=4, zorder=5) 
        
        # 2. Plot Both Branches
        for branch in ['positive', 'negative']:
            fit_row = fit_df[(fit_df['Ensemble'] == group) & 
                             (fit_df['Data_Type'] == prefix) & 
                             (fit_df['Branch'] == branch)]
            
            if not fit_row.empty:
                s0_cen = fit_row.iloc[0]['sigma_0_central']
                s0_stat_err = fit_row.iloc[0]['sigma_0_stat_err']
                s0_up = fit_row.iloc[0]['sigma_0_up']
                s0_down = fit_row.iloc[0]['sigma_0_down']
                
                lp_cen = fit_row.iloc[0]['lamb_prime_central']
                lp_up = fit_row.iloc[0]['lamb_prime_up']
                lp_down = fit_row.iloc[0]['lamb_prime_down']
                
                y_cen = s0_cen - 4 * lp_cen * x_vals_math
                y_up_sys = s0_up - 4 * lp_up * x_vals_math
                y_down_sys = s0_down - 4 * lp_down * x_vals_math
                
                y_sys_err = np.maximum(np.abs(y_cen - y_up_sys), np.abs(y_cen - y_down_sys))
                y_total_err = np.sqrt(y_sys_err**2 + s0_stat_err**2)
                
                c = branch_styles[branch]['color']
                
                # Create the dynamic equation label using scientific notation
                def fmt_sci(val):
                    base, exp = "{:.2e}".format(val).split('e')
                    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"
                
                eq_label = rf"{branch_labels[branch]} ($\sigma_0 = {fmt_sci(s0_cen)}$)"
                
                for axis in [ax, ax_grp]:
                    axis.plot(x_vals_plot, y_cen, linestyle='--', color=c, label=eq_label)
                    axis.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=c, alpha=0.25)

        # Aesthetics for BOTH axes
        for axis in [ax, ax_grp]:
            axis.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
            axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) # Scientific notation
            axis.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
            axis.legend(fontsize=10, loc='best')

        ax.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
        
        fig.tight_layout()
        indiv_filename = f'Plot_Lineal_{group}_{prefix}_{A_types[A_index]}.svg'
        fig.savefig(indiv_filename, bbox_inches='tight')
        plt.close(fig)

    axes_grp[0].set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
    fig_grp.tight_layout()
    grp_filename = f'Plot_Lineal_Grouped_{prefix}_{A_types[A_index]}.svg'
    fig_grp.savefig(grp_filename, bbox_inches='tight')
    plt.close(fig_grp)