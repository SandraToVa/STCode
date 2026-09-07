import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
LQCD = False

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
lqcd_str = "_full" if LQCD else ""
filename = f'fit_results{lqcd_str}_ST_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: "_".join(s.split('_')[-2:]))
fit_df = pd.read_csv(filename)

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

data_types = ['bare', 'smeared']

groups = ['M_i', 'M_ii', 'M_iii']

# Color palette across groups
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))
# Marker shapes across groups
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
group_marker_map = dict(zip(groups, markers[:len(groups)]))
line_styles = ['-', '--', '-.', ':']

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    ensembles = df['mass_group'].unique()
    fig_grp, axes_grp = plt.subplots(1, len(ensembles), figsize=(18, 6), sharey=False)
    lqcd_title = "Full" if LQCD else ""
    fig_grp.suptitle(f'{lqcd_title} Soto-Tarrús Fits - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=18, fontweight='bold', y=1.02)
    
    for ax_grp, group in zip(axes_grp, ensembles):
        subset = df[df['mass_group'] == group]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Ensemble: {group}', fontsize=16)
        ax_grp.set_title(f'Ensemble: {group}', fontsize=16)
        
        x_list, x_err_list, y_list, y_err_list = [], [], [], []
        for _, row in subset.iterrows():
            x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
            x_list.append(x_val)
            x_err_list.append(x_e)
            y_list.append(y_val)
            y_err_list.append(y_e)
            
        x_arr, x_err_arr = np.array(x_list), np.array(x_err_list)
        y_arr, y_err_arr = np.array(y_list), np.array(y_err_list)
        
        x_min, x_max = min(x_arr), max(x_arr)
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else x_min * 0.05
        x_vals_plot = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        
        for axis in [ax, ax_grp]:
            axis.errorbar(x_arr, y_arr, xerr=x_err_arr, yerr=y_err_arr, fmt=group_marker_map[group], color=group_color_map[group], alpha=0.9, label='Data', capsize=4, zorder=5) 
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
        
        fit_row = fit_df[(fit_df['Ensemble'] == group) & (fit_df['Data_Type'] == prefix)]
        if not fit_row.empty:
            row_data = fit_row.iloc[0]
            y0_cen = row_data['y_0']
            cl_cen = row_data['c_l']
            gamma_cen = row_data['gamma']
            
            if LQCD:
                c2l_cen = row_data['c2_l']
                y_cen = model_func2_dim(x_vals_plot, y0_cen, cl_cen, c2l_cen, gamma_cen)
                eq_label = rf"$y_0 = {y0_cen:.2f}$, $c_1 = {cl_cen:.2f}$, $c_2 = {c2l_cen:.2f}$, $\gamma = {gamma_cen:.2f}$"
            else:
                y_cen = model_func_dim(x_vals_plot, y0_cen, cl_cen, gamma_cen)
                eq_label = rf"$y_0 = {y0_cen:.2f}$, $c_l = {cl_cen:.2f}$, $\gamma = {gamma_cen:.2f}$"

            # Plot fitting curve directly from CSV parameter values
            for axis in [ax, ax_grp]:
                axis.plot(x_vals_plot, y_cen, linestyle='-', color='#003366', lw=2, label=eq_label, zorder=5)

            # --- OPTIONAL: Plot Error Band using pre-computed pcov from CSV ---
            if 'pcov' in row_data and pd.notna(row_data['pcov']):
                pcov = np.array(ast.literal_eval(str(row_data['pcov'])))
                
                if LQCD:
                    z = x_vals_plot + c2l_cen/cl_cen
                    df_dp = np.array([
                        np.ones_like(x_vals_plot),
                        (z * (cl_cen**3 * x_vals_plot * np.log(z**2) - c2l_cen * (cl_cen**2 + 2 * gamma_cen * np.pi))) / (cl_cen**2 * np.pi),
                        (z * (cl_cen**2 * np.log(z**2) + cl_cen**2 + 2 * gamma_cen * np.pi)) / (cl_cen * np.pi),
                        (z**2)
                    ])
                else:
                    df_dp = np.array([
                        np.ones_like(x_vals_plot),
                        (cl_cen / np.pi) * x_vals_plot**2 * np.log(x_vals_plot**2),
                        2 * x_vals_plot**2
                    ])

                variance_y = np.einsum('ik,ij,jk->k', df_dp, pcov, df_dp)
                y_total_err = np.sqrt(np.maximum(variance_y, 0))

                for axis in [ax, ax_grp]:
                    axis.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color='#003366', alpha=0.2, zorder=3)

        for axis in [ax, ax_grp]:
            axis.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
            axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) 
            axis.set_xlabel(r'$x = m_l r_0$', fontsize=14)
            axis.legend(fontsize=10, loc='best')

        ax.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'Plot_ST{lqcd_str}_{group}_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
        plt.close(fig)

    axes_grp[0].set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    fig_grp.tight_layout()
    fig_grp.savefig(f'Plot_ST{lqcd_str}_Grouped_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_grp)