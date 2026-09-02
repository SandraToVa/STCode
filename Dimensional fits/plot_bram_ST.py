# Reads both data_bram.csv and fit_results_ST.csv, 
# and generates the final individual and grouped publication-ready figures.

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
mu = np.sqrt(210)         # MeV sqrt(sigma) from paper Soto-Tarrús
gamma_E = np.euler_gamma  # Euler-Mascheroni constant

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_ST_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
# Extracts 'M_i', 'M_ii', etc. to match the fit results row keys
df['mass_group'] = df['Ensemble'].apply(lambda x: "_".join(x.split('_')[-2:]))
fit_df = pd.read_csv(filename)

def calculate_physics(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    a = r0_MeV / r0_a
    a_err = a * (r0_a_err / r0_a + r0_MeV_err / r0_MeV)
    m_l = df_row['am_l'] / a
    m_l_err = m_l * (a_err / a)
    sigma = a2sigma / (a**2)
    sigma_err = sigma * np.sqrt((a2sigma_err/a2sigma)**2 + (2*a_err/a)**2)
    return m_l, m_l_err, sigma, sigma_err

ensembles = df['mass_group'].unique()
data_types = ['bare', 'smeared']
data_color = '#1f1f1f'
fit_color = 'blue'

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val):
        return "N/A"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    # Prepare Grouped Figure
    fig_grp, axes_grp = plt.subplots(1, len(ensembles), figsize=(18, 6), sharey=True)
    fig_grp.suptitle(f'Soto-Tarrus Model Fits - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=18, fontweight='bold', y=1.02)
    
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
            
        m_l_arr, m_l_err_arr = np.array(m_l_list), np.array(m_l_err_list)
        sigma_arr, sigma_err_arr = np.array(sigma_list), np.array(sigma_err_list)
            
        x_min, x_max = min(m_l_arr), max(m_l_arr)
        x_vals_plot = np.linspace(x_min - (x_max-x_min)*0.2, x_max + (x_max-x_min)*0.2, 100)
        
        # 1. Plot Data points on BOTH axes
        for axis in [ax, ax_grp]:
            axis.errorbar(m_l_arr, sigma_arr, xerr=m_l_err_arr, yerr=sigma_err_arr, 
                        fmt='o', color=data_color, alpha=0.9, 
                        label='Data', capsize=4, zorder=5) 
        
        # 2. Plot Fit and Error Band (Updated Column Keys for ST model)
        fit_row = fit_df[(fit_df['Ensemble'] == group) & (fit_df['Data_Type'] == prefix)]
        
        if not fit_row.empty:
            s0_cen = fit_row.iloc[0]['sigma_0']
            s0_err = fit_row.iloc[0]['sigma_0_err']
            cl_cen = fit_row.iloc[0]['c_l']
            cl_err = fit_row.iloc[0]['c_l_err']
            
            # Logarithmic model calculation
            log_arg = (cl_cen**2 * x_vals_plot**2) / (4 * np.pi * mu**2)
            log_arg = np.maximum(log_arg, 1e-15)
            
            term1 = (cl_cen**2 / (2 * np.pi)) * x_vals_plot**2
            term2 = 1 + gamma_E - np.log(log_arg)
            y_cen = s0_cen - term1 * term2
            
            # Analytical error band propagation via parameter derivatives
            if np.isinf(s0_err) or np.isinf(cl_err):
                y_total_err = np.zeros_like(x_vals_plot)
            else:
                df_dcl = - (cl_cen / np.pi) * x_vals_plot**2 * (gamma_E - np.log(log_arg))
                y_total_err = np.sqrt(s0_err**2 + (df_dcl * cl_err)**2)
            
            eq_label = rf"Fit ($\sigma_0 = {fmt_sci(s0_cen)}$, $c_l = {cl_cen:.3f}$)"
            
            for axis in [ax, ax_grp]:
                axis.plot(x_vals_plot, y_cen, linestyle='--', color=fit_color, label=eq_label)
                if not np.all(y_total_err == 0):
                    axis.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=fit_color, alpha=0.25)

        # Aesthetics for BOTH axes
        for axis in [ax, ax_grp]:
            axis.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
            axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) 
            axis.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
            axis.legend(fontsize=10, loc='best')

        ax.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
        
        fig.tight_layout()
        indiv_filename = f'Plot_ST_{group}_{prefix}_{A_types[A_index]}.svg'
        fig.savefig(indiv_filename, bbox_inches='tight')
        plt.close(fig)

    axes_grp[0].set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
    fig_grp.tight_layout()
    grp_filename = f'Plot_ST_Grouped_{prefix}_{A_types[A_index]}.svg'
    fig_grp.savefig(grp_filename, bbox_inches='tight')
    plt.close(fig_grp)