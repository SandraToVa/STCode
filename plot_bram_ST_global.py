import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import ast  # To parse matrix strings from CSV if needed

# Aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

# Constants
hbar_c = 197.3269804
r0_fm = 0.4547
r0_MeV = r0_fm / hbar_c
  

A_types = ['Ar0', 'Api12']
A_latex = ['r_0', '\\pi/12']
A_index = 0

LQCD =  False

# File Paths
data_file = f'Data/data_bram_{A_types[A_index]}.csv'
lqcd_str = "_full" if LQCD else ""
fit_file = f'fit_results{lqcd_str}_ST_global_{A_types[A_index]}.csv'

if not os.path.exists(data_file) or not os.path.exists(fit_file):
    # Fallback to local execution directory if subfolder isn't present
    data_file = f'data_bram_{A_types[A_index]}.csv'
    fit_file = f'fit_results_ST_global_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: "_".join(s.split('_')[-2:]))
fit_df = pd.read_csv(fit_file)

def calculate_dimensionless(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    
    y = a2sigma * (r0_a ** 2)
    y_err = y * np.sqrt((a2sigma_err / a2sigma)**2 + (2 * r0_a_err / r0_a)**2)
    x = am_l * r0_a
    x_err = x * (r0_a_err / r0_a)
    return x, x_err, y, y_err

# Model functions
def model_func_dim(x, y_0, c_l, gamma):
    term1 = (c_l**2 / (2 * np.pi)) * x**2 * np.log(x**2)
    term2 = gamma * x**2
    
    return y_0 + term1 * term2

def model_func2_dim(x, y_0, c_l, c2_l, gamma):
    z = x + c_l/c2_l
    term1 = (c_l**2 / (2 * np.pi)) * z**2 * np.log(z**2)
    term2 = gamma * z**2
    
    return y_0 + term1 * term2


def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    if abs(val) < 1e-2 or abs(val) > 1e3:
        base, exp = "{:.2e}".format(val).split('e')
        return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"
    return f"{val:.4f}"

data_types = ['bare', 'smeared']
groups = sorted(df['mass_group'].unique())

# Color palette across groups
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))

for prefix in data_types:
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, len(groups), height_ratios=[1.6, 1], hspace=0.35, wspace=0.25)
    
    ax_main = fig.add_subplot(gs[0, :])
    zoom_axes = [fig.add_subplot(gs[1, i]) for i in range(len(groups))]

    lqcd_title = "Full" if LQCD else ""
    fig.suptitle(
        f'{lqcd_title}Global Soto-Tarrús Fit - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', 
        fontsize=18, fontweight='bold', y=0.96
    )
    
    # Calculate x, y and errors
    phys_data = df.apply(lambda row: calculate_dimensionless(row, prefix), axis=1)
    df['x_calc'], df['x_err'] = zip(*[(d[0], d[1]) for d in phys_data])
    df['y_calc'], df['y_err'] = zip(*[(d[2], d[3]) for d in phys_data])
    
    dtype_fits = fit_df[fit_df['Data_Type'] == prefix] if 'Data_Type' in fit_df.columns else fit_df
    
    # Plot Data on Main Axis with Group Colors
    for group in groups:
        grp_data = df[df['mass_group'] == group]
        ax_main.errorbar(
            grp_data['x_calc'], grp_data['y_calc'], 
            xerr=grp_data['x_err'], yerr=grp_data['y_err'], 
            fmt='o', color=group_color_map[group], alpha=0.85, 
            capsize=3, label=f'Ensamble: {group}', zorder=4
        )
    
    # Global Fit Curve and Band Propagation from fit_df
    if not dtype_fits.empty:
        fit_row = dtype_fits.iloc[0]
        y0_cen = fit_row['y_0']
        cl_cen = fit_row['c_l']
        gamma_cen = fit_row['gamma']
        
        
        x_min_main, x_max_main = min(df['x_calc']), max(df['x_calc'])
        x_pad = (x_max_main - x_min_main) * 0.05
        x_vals_main = np.linspace(x_min_main - x_pad, x_max_main + x_pad, 500)
        

        if LQCD:
            c2l_cen = fit_row['c2_l']
            y_cen = model_func2_dim(x_vals_main, y0_cen, cl_cen, c2l_cen, gamma_cen)
            eq_label = rf"$y_0 = {y0_cen:.2f}$, $c_1 = {cl_cen:.2f}$, $c_2 = {c2l_cen:.2f}$, $\gamma = {gamma_cen:.2f}$"
        else:
            y_cen = model_func_dim(x_vals_main, y0_cen, cl_cen, gamma_cen)
            eq_label = rf"$y_0 = {y0_cen:.2f}$, $c_l = {cl_cen:.2f}$, $\gamma = {gamma_cen:.2f}$"
        
        ax_main.plot(x_vals_main, y_cen, linestyle='-', color='#003366', lw=2, label=eq_label, zorder=5)
        
        # --- OPTIONAL: Plot Error Band using pre-computed pcov from CSV ---
        if 'pcov' in fit_row and pd.notna(fit_row['pcov']):
            pcov = np.array(ast.literal_eval(str(fit_row['pcov'])))
                        
            if LQCD:
                z = x_vals_main + c2l_cen/cl_cen
                df_dp = np.array([
                    np.ones_like(x_vals_main),
                    (z * (cl_cen**3 * x_vals_main * np.log(z**2) - c2l_cen * (cl_cen**2 + 2 * gamma_cen * np.pi))) / (cl_cen**2 * np.pi),
                    (z * (cl_cen**2 * np.log(z**2) + cl_cen**2 + 2 * gamma_cen * np.pi)) / (cl_cen * np.pi),
                    (z**2)
                ])
            else:
                df_dp = np.array([
                    np.ones_like(x_vals_main),
                    (cl_cen / np.pi) * x_vals_main**2 * np.log(x_vals_main**2),
                    2 * x_vals_main**2
                ])
        
            variance_y = np.einsum('ik,ij,jk->k', df_dp, pcov, df_dp)
            y_total_err = np.sqrt(np.maximum(variance_y, 0))

            ax_main.fill_between(x_vals_main, y_cen - y_total_err, y_cen + y_total_err, color='#003366', alpha=0.2, zorder=3)
        
        ax_main.set_xlim(x_min_main - x_pad, x_max_main + x_pad)

    #ax_main.set_title('Global Overview', fontsize=15)
    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=13)
    ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=13)
    ax_main.legend(fontsize=11, loc='best', framealpha=0.9)

    # Subplots per Group
    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=12)
        x_min_z, x_max_z = min(subset['x_calc']), max(subset['x_calc'])
        x_pad_z = (x_max_z - x_min_z) * 0.1 if x_max_z > x_min_z else x_min_z * 0.05
        x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 150)
        
        ax_z.errorbar(
            subset['x_calc'], subset['y_calc'], 
            xerr=subset['x_err'], yerr=subset['y_err'], 
            fmt='o', color=group_color_map[group], alpha=0.9, capsize=3, zorder=4
        )
        
        if not dtype_fits.empty:
            if LQCD:
                c2l_cen = fit_row['c2_l']
                y_cen = model_func2_dim(x_vals_main, y0_cen, cl_cen, c2l_cen, gamma_cen)
            else:
                y_cen = model_func_dim(x_vals_main, y0_cen, cl_cen, gamma_cen)
            
            ax_z.plot(x_vals_main, y_cen, linestyle='-', color='#003366', lw=2, zorder=5)
        
            # --- OPTIONAL: Plot Error Band using pre-computed pcov from CSV ---
            if 'pcov' in fit_row and pd.notna(fit_row['pcov']):
                pcov = np.array(ast.literal_eval(str(fit_row['pcov'])))
                        
                if LQCD:
                    z = x_vals_main + c2l_cen/cl_cen
                    df_dp = np.array([
                        np.ones_like(x_vals_main),
                        (z * (cl_cen**3 * x_vals_main * np.log(z**2) - c2l_cen * (cl_cen**2 + 2 * gamma_cen * np.pi))) / (cl_cen**2 * np.pi),
                        (z * (cl_cen**2 * np.log(z**2) + cl_cen**2 + 2 * gamma_cen * np.pi)) / (cl_cen * np.pi),
                        (z**2)
                    ])
                else:
                    df_dp = np.array([
                        np.ones_like(x_vals_main),
                        (cl_cen / np.pi) * x_vals_main**2 * np.log(x_vals_main**2),
                        2 * x_vals_main**2
                    ])
        
                variance_y = np.einsum('ik,ij,jk->k', df_dp, pcov, df_dp)
                y_total_err = np.sqrt(np.maximum(variance_y, 0))

                ax_z.fill_between(x_vals_main, y_cen - y_total_err, y_cen + y_total_err, color='#003366', alpha=0.2, zorder=3)
        
        ax_main.set_xlim(x_min_main - x_pad, x_max_main + x_pad)

        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=5)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=12)
        if idx == 0: 
            ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=12)

    output_name = f'Plot_ST{lqcd_str}_Global_{prefix}_{A_types[A_index]}.svg'
    fig.savefig(output_name, bbox_inches='tight')
    plt.close(fig)