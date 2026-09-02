import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

# Aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix' 

# Constants
hbar_c = 197.3269804
r0_fm = 0.4547
r0_MeV = r0_fm / hbar_c
mu_dim = np.sqrt(210) * r0_MeV
gamma_E = np.euler_gamma  

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
A_index = 0
# Flag LQCD
LQCD = True  # Canvia a True per carregar els resultats de fit_results_ST_full_...

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'

# Selecció dinàmica del nom de fitxer segons la variable LQCD
lqcd_str = "_full" if LQCD else ""
filename = f'fit_results_ST{lqcd_str}_{A_types[A_index]}.csv'

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

# Updated Model Function to conditionally include c2_l
def model_func_dim(x, y_0, c_l, c2_l=0.0):
    log_arg = np.maximum((c_l**2 * x**2) / (4 * np.pi * mu_dim**2), 1e-15)
    term1 = (c_l**2 / (2 * np.pi)) * x**2
    term2 = 1 + gamma_E - np.log(log_arg)
    return y_0 - term1 * term2 + c2_l * (x**2)

data_types = ['bare', 'smeared']
data_color = '#1f1f1f'
fit_color = 'blue'

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    ensembles = df['mass_group'].unique()
    fig_grp, axes_grp = plt.subplots(1, len(ensembles), figsize=(18, 6), sharey=False)
    fig_grp.suptitle(f'Dimensionless ST Fits - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=18, fontweight='bold', y=1.02)
    
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
            axis.errorbar(x_arr, y_arr, xerr=x_err_arr, yerr=y_err_arr, fmt='o', color=data_color, alpha=0.9, label='Data', capsize=4, zorder=5) 
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
        
        fit_row = fit_df[(fit_df['Ensemble'] == group) & (fit_df['Data_Type'] == prefix)]
        if not fit_row.empty:
            y0_cen = fit_row.iloc[0]['y_0']
            cl_cen = fit_row.iloc[0]['c_l']
            
            # Check if c2_l is part of this fit
            has_c2l = 'c2_l' in fit_row.columns
            c2l_cen = fit_row.iloc[0]['c2_l'] if has_c2l else 0.0
            
            cov_y0_y0, cov_cl_cl, cov_y0_cl = 0, 0, 0
            cov_c2l_c2l, cov_y0_c2l, cov_cl_c2l = 0, 0, 0
            
            try:
                # Dynamically set function and p0 based on whether we are tracking c2_l
                if has_c2l:
                    def fit_func(x, y0, cl, c2l): return model_func_dim(x, y0, cl, c2l)
                    p0_vals = [y0_cen, cl_cen, c2l_cen]
                else:
                    def fit_func(x, y0, cl): return model_func_dim(x, y0, cl, 0.0)
                    p0_vals = [y0_cen, cl_cen]

                _, pcov = curve_fit(fit_func, x_arr, y_arr, sigma=y_err_arr, absolute_sigma=True, p0=p0_vals)
                
                if not np.isinf(pcov).any():
                    cov_y0_y0, cov_cl_cl, cov_y0_cl = pcov[0,0], pcov[1,1], pcov[0,1]
                    if has_c2l:
                        cov_c2l_c2l = pcov[2,2]
                        cov_y0_c2l = pcov[0,2]
                        cov_cl_c2l = pcov[1,2]
            except Exception:
                pass
            
            log_arg = np.maximum((cl_cen**2 * x_vals_plot**2) / (4 * np.pi * mu_dim**2), 1e-15)
            y_cen = model_func_dim(x_vals_plot, y0_cen, cl_cen, c2l_cen)
            
            df_dy0 = 1.0
            df_dcl = - (cl_cen / np.pi) * x_vals_plot**2 * (gamma_E - np.log(log_arg))
            df_dc2l = x_vals_plot**2 if has_c2l else 0.0
            
            # Expanded variance taking into account all possible covariances
            variance_y = (
                (df_dy0**2 * cov_y0_y0) + 
                (df_dcl**2 * cov_cl_cl) + 
                (df_dc2l**2 * cov_c2l_c2l) +
                (2 * df_dy0 * df_dcl * cov_y0_cl) +
                (2 * df_dy0 * df_dc2l * cov_y0_c2l) +
                (2 * df_dcl * df_dc2l * cov_cl_c2l)
            )
            y_total_err = np.sqrt(np.maximum(variance_y, 0))
            
            if has_c2l:
                eq_label = rf"Fit ($y_0 = {fmt_sci(y0_cen)}$, $|c_l| = {abs(cl_cen):.2f}$, $c_{{2,l}} = {c2l_cen:.2f}$)"
            else:
                eq_label = rf"Fit ($y_0 = {fmt_sci(y0_cen)}$, $|c_l| = {abs(cl_cen):.2f}$)"
                
            for axis in [ax, ax_grp]:
                axis.plot(x_vals_plot, y_cen, linestyle='-', color=fit_color, label=eq_label)
                if not np.all(y_total_err == 0):
                    axis.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=fit_color, alpha=0.25)

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