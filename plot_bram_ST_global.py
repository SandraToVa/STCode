import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning

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
A_index = 1
# Flag LQCD
LQCD = True  # Set to True to load the full fit results

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/data_bram_{A_types[A_index]}.csv'

# Selecció dinàmica del nom de fitxer segons la variable LQCD
lqcd_str = "_full" if LQCD else ""
filename = f'fit_results_ST{lqcd_str}_global_{A_types[A_index]}.csv'

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
    # Adding the analytic term. Using x**2 as standard for O(m_l^2) terms.
    # If your specific EFT relies on linear m_l, change (x**2) to x.
    return y_0 - term1 * term2 + c2_l * (x**2)

data_types = ['bare', 'smeared']
groups = df['mass_group'].unique()
data_color = '#1f1f1f'
fit_color = 'blue'

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, len(groups), height_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
    ax_main = fig.add_subplot(gs[0, :])
    
    zoom_axes = []
    for i in range(len(groups)):
        if i == 0:
            ax = fig.add_subplot(gs[1, i])
        else:
            ax = fig.add_subplot(gs[1, i], sharey=zoom_axes[0])
        zoom_axes.append(ax)
    
    fig.suptitle(f'Global Dimensionless ST Fit - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=20, fontweight='bold', y=0.95)
    
    phys_data_all = df.apply(lambda row: calculate_dimensionless(row, prefix), axis=1)
    df['x_calc'], df['x_err'] = zip(*[(d[0], d[1]) for d in phys_data_all])
    df['y_calc'], df['y_err'] = zip(*[(d[2], d[3]) for d in phys_data_all])
    
    dtype_fits = fit_df[fit_df['Data_Type'] == prefix]
    
    ax_main.set_title('Global Overview', fontsize=16)
    x_min_main, x_max_main = min(df['x_calc']), max(df['x_calc'])
    x_pad_main = (x_max_main - x_min_main) * 0.05
    x_vals_main = np.linspace(x_min_main - x_pad_main, x_max_main + x_pad_main, 500)
    ax_main.set_xlim(x_min_main - x_pad_main, x_max_main + x_pad_main)
    
    ax_main.errorbar(df['x_calc'], df['y_calc'], xerr=df['x_err'], yerr=df['y_err'], fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

    if not dtype_fits.empty:
        y0_cen = dtype_fits.iloc[0]['y_0']
        cl_cen = dtype_fits.iloc[0]['c_l']
        
        # Check if c2_l is part of this fit
        has_c2l = 'c2_l' in dtype_fits.columns
        c2l_cen = dtype_fits.iloc[0]['c2_l'] if has_c2l else 0.0
        
        log_arg_eff = np.maximum((cl_cen**2 * np.array(df['x_calc'])**2) / (4 * np.pi * mu_dim**2), 1e-15)
        df_dx = - (cl_cen**2 / np.pi) * np.array(df['x_calc']) * (gamma_E - np.log(log_arg_eff))
        if has_c2l: df_dx += 2 * c2l_cen * np.array(df['x_calc'])
        eff_y_err = np.sqrt(np.array(df['y_err'])**2 + (df_dx * np.array(df['x_err']))**2)
        
        cov_y0_y0, cov_cl_cl, cov_y0_cl = 0, 0, 0
        cov_c2l_c2l, cov_y0_c2l, cov_cl_c2l = 0, 0, 0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            try:
                # Dynamically set function and p0 based on whether we are tracking c2_l
                if has_c2l:
                    def fit_func(x, y0, cl, c2l): return model_func_dim(x, y0, cl, c2l)
                    p0_vals = [y0_cen, cl_cen, c2l_cen]
                else:
                    def fit_func(x, y0, cl): return model_func_dim(x, y0, cl, 0.0)
                    p0_vals = [y0_cen, cl_cen]

                _, pcov = curve_fit(fit_func, df['x_calc'].values, df['y_calc'].values, 
                                    sigma=eff_y_err, absolute_sigma=True, p0=p0_vals)
                
                if not np.isinf(pcov).any():
                    cov_y0_y0, cov_cl_cl, cov_y0_cl = pcov[0,0], pcov[1,1], pcov[0,1]
                    if has_c2l:
                        cov_c2l_c2l = pcov[2,2]
                        cov_y0_c2l = pcov[0,2]
                        cov_cl_c2l = pcov[1,2]
            except Exception:
                pass 
            
        log_arg_main = np.maximum((cl_cen**2 * x_vals_main**2) / (4 * np.pi * mu_dim**2), 1e-15)
        y_cen = model_func_dim(x_vals_main, y0_cen, cl_cen, c2l_cen)
        
        df_dy0 = 1.0
        df_dcl_main = - (cl_cen / np.pi) * x_vals_main**2 * (gamma_E - np.log(log_arg_main))
        df_dc2l_main = x_vals_main**2 if has_c2l else 0.0
        
        # Expanded variance taking into account all possible covariances
        variance_y_main = (
            (df_dy0**2 * cov_y0_y0) + 
            (df_dcl_main**2 * cov_cl_cl) + 
            (df_dc2l_main**2 * cov_c2l_c2l) +
            (2 * df_dy0 * df_dcl_main * cov_y0_cl) +
            (2 * df_dy0 * df_dc2l_main * cov_y0_c2l) +
            (2 * df_dcl_main * df_dc2l_main * cov_cl_c2l)
        )
        y_total_err = np.sqrt(np.maximum(variance_y_main, 0))
        
        if has_c2l:
            eq_label = rf"Universal Fit ($y_0 = {fmt_sci(y0_cen)}$, $c_l = {cl_cen:.3f}$, $c_{{2,l}} = {c2l_cen:.3f}$)"
        else:
            eq_label = rf"Universal Fit ($y_0 = {fmt_sci(y0_cen)}$, $c_l = {cl_cen:.3f}$)"
            
        ax_main.plot(x_vals_main, y_cen, linestyle='-', color=fit_color, label=eq_label)
        
        if not np.all(y_total_err == 0):
            ax_main.fill_between(x_vals_main, y_cen - y_total_err, y_cen + y_total_err, color=fit_color, alpha=0.25)

    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=14)
    ax_main.legend(fontsize=12, loc='best')

    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=14)
        x_min_z, x_max_z = min(subset['x_calc']), max(subset['x_calc'])
        x_pad_z = (x_max_z - x_min_z) * 0.05 if x_max_z > x_min_z else x_min_z * 0.05
        x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 100)
        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        
        ax_z.errorbar(subset['x_calc'], subset['y_calc'], xerr=subset['x_err'], yerr=subset['y_err'], fmt='o', color=data_color, alpha=0.9, capsize=4, zorder=5)
        
        if not dtype_fits.empty:
            log_arg_z = np.maximum((cl_cen**2 * x_vals_z**2) / (4 * np.pi * mu_dim**2), 1e-15)
            y_cen_z = model_func_dim(x_vals_z, y0_cen, cl_cen, c2l_cen)
            
            df_dcl_z = - (cl_cen / np.pi) * x_vals_z**2 * (gamma_E - np.log(log_arg_z))
            df_dc2l_z = x_vals_z**2 if has_c2l else 0.0
            
            variance_y_z = (
                (df_dy0**2 * cov_y0_y0) + 
                (df_dcl_z**2 * cov_cl_cl) + 
                (df_dc2l_z**2 * cov_c2l_c2l) +
                (2 * df_dy0 * df_dcl_z * cov_y0_cl) +
                (2 * df_dy0 * df_dc2l_z * cov_y0_c2l) +
                (2 * df_dcl_z * df_dc2l_z * cov_cl_c2l)
            )
            y_total_err_z = np.sqrt(np.maximum(variance_y_z, 0))
            
            ax_z.plot(x_vals_z, y_cen_z, linestyle='-', color=fit_color)
            if not np.all(y_total_err_z == 0):
                ax_z.fill_between(x_vals_z, y_cen_z - y_total_err_z, y_cen_z + y_total_err_z, color=fit_color, alpha=0.25)

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=14)
        if idx == 0: ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)

    fig.savefig(f'Plot_ST{lqcd_str}_Global_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig)