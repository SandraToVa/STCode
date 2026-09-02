# Reads data_bram.csv and the globally fit quadratic results,
# and generates a GridSpec Figure with a main overview and 3 zooms.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

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
nf = 3

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_quad_global_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
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

data_types = ['bare', 'smeared']
groups = df['mass_group'].unique()
data_color = '#1f1f1f'
fit_color = 'blue'

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val):
        return "N/A"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, len(groups), height_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
    
    ax_main = fig.add_subplot(gs[0, :])
    zoom_axes = []
    
    for i in range(len(groups)):
        if i == 0:
            zoom_axes.append(fig.add_subplot(gs[1, i]))
        else:
            zoom_axes.append(fig.add_subplot(gs[1, i], sharey=zoom_axes[0]))
    
    fig.suptitle(f'Universal Quadratic Fit - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=20, fontweight='bold', y=0.95)
    
    # Pre-calculate physical data
    phys_data_all = df.apply(lambda row: calculate_physics(row, prefix), axis=1)
    df['m_l_calc'] = [x[0] for x in phys_data_all]
    df['m_l_err'] = [x[1] for x in phys_data_all]
    df['sigma_calc'] = [x[2] for x in phys_data_all]
    df['sigma_err'] = [x[3] for x in phys_data_all]
    
    # Filter fits by current prefix (bare or smeared)
    dtype_fits = fit_df[fit_df['Data_Type'] == prefix]
    
    # ---------------------------------------------------------
    # 1. Main Global Figure (Top Panel)
    # ---------------------------------------------------------
    ax_main.set_title('Global Overview', fontsize=16)
    x_min_main, x_max_main = df['m_l_calc'].min(), df['m_l_calc'].max()
    x_vals_main = np.linspace(x_min_main - 0.5, x_max_main + 0.5, 500)
    
    # Scatter all data
    ax_main.errorbar(df['m_l_calc'], df['sigma_calc'], xerr=df['m_l_err'], yerr=df['sigma_err'], 
                     fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

    # Branch filter removed. Directly use dtype_fits and exact column names.
    if not dtype_fits.empty:
        s0_cen = dtype_fits.iloc[0]['sigma_0']
        s0_err = dtype_fits.iloc[0]['sigma_0_err']
        eps_cen = dtype_fits.iloc[0]['epsilon']
        eps_err = dtype_fits.iloc[0]['epsilon_err']
        
        y_cen = s0_cen - (nf / (2 * np.pi * eps_cen)) * x_vals_main**2
        
        if np.isinf(s0_err) or np.isinf(eps_err):
            y_total_err = np.zeros_like(x_vals_main)
        else:
            y_total_err = np.sqrt(s0_err**2 + ((nf/(2*np.pi * eps_cen**2)) * x_vals_main**2 * eps_err)**2)
        
        eq_label = rf"Quadratic Fit ($\sigma_0 = {fmt_sci(s0_cen)}$, $\epsilon = {fmt_sci(eps_cen)}$)"
        ax_main.plot(x_vals_main, y_cen, linestyle='--', color=fit_color, label=eq_label)
        if not np.all(y_total_err == 0):
            ax_main.fill_between(x_vals_main, y_cen - y_total_err, y_cen + y_total_err, color=fit_color, alpha=0.25)

    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
    ax_main.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
    ax_main.legend(fontsize=12, loc='best')

    # ---------------------------------------------------------
    # 2. Plot Zoom Figures (Bottom Panels)
    # ---------------------------------------------------------
    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=14)
        
        x_min_z, x_max_z = subset['m_l_calc'].min(), subset['m_l_calc'].max()
        x_vals_z = np.linspace(x_min_z - 0.2, x_max_z + 0.2, 100)
        
        ax_z.errorbar(subset['m_l_calc'], subset['sigma_calc'], xerr=subset['m_l_err'], yerr=subset['sigma_err'], 
                      fmt='o', color=data_color, alpha=0.9, capsize=4, zorder=5)
        
        if not dtype_fits.empty:
            y_cen_z = s0_cen - (nf / (2 * np.pi * eps_cen)) * x_vals_z**2
            
            if np.isinf(s0_err) or np.isinf(eps_err):
                y_total_err_z = np.zeros_like(x_vals_z)
            else:
                y_total_err_z = np.sqrt(s0_err**2 + ((nf/(2*np.pi * eps_cen**2)) * x_vals_z**2 * eps_err)**2)
            
            ax_z.plot(x_vals_z, y_cen_z, linestyle='--', color=fit_color)
            if not np.all(y_total_err_z == 0):
                ax_z.fill_between(x_vals_z, y_cen_z - y_total_err_z, y_cen_z + y_total_err_z, color=fit_color, alpha=0.25)

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
        if idx == 0:
            ax_z.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)


    grp_filename = f'Plot_Quad_Global_{prefix}_{A_types[A_index]}.svg'
    fig.savefig(grp_filename, bbox_inches='tight')
    plt.close(fig)