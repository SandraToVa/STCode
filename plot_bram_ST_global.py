import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

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
A_latex = ['r_0', '\\pi/12']
A_index = 0  

LQCD = False  

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

def model_func_dim(x, y_0, c_l, c2_l=0.0):
    log_arg = np.maximum((c_l**2 * x**2) / (4 * np.pi * mu_dim**2), 1e-15)
    term1 = (c_l**2 / (2 * np.pi)) * (x**2)
    term2 = 1 + gamma_E - np.log(log_arg)
    return y_0 - term1 * term2 + c2_l * (x**2)

def predict_global_fit(x_vals, y0, cl, c2l, cov):
    """Calculates y_pred and uncertainty bands via Jacobian covariance propagation."""
    y_pred = model_func_dim(x_vals, y0, cl, c2l)
    
    log_arg = np.maximum((cl**2 * x_vals**2) / (4 * np.pi * mu_dim**2), 1e-15)
    
    # Partial derivatives
    df_dy0 = np.ones_like(x_vals)
    df_dcl = - (cl / np.pi) * (x_vals**2) * (gamma_E - np.log(log_arg))
    df_dc2l = x_vals**2
    
    J = np.array([df_dy0, df_dcl, df_dc2l])  # Shape (3, N)
    var_y = np.einsum('ik,ij,jk->k', J, cov, J)
    y_err = np.sqrt(np.maximum(var_y, 0))
    
    return y_pred, y_err

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
    
    fig.suptitle(
        f'Global Dimensionless ST Fit - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', 
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
            capsize=3, label=f'Group {group}', zorder=4
        )
    
    # Global Fit Curve and Band Propagation from fit_df
    if not dtype_fits.empty:
        fit_row = dtype_fits.iloc[0]
        y0_cen = fit_row['y_0']
        cl_cen = fit_row['c_l']
        c2l_cen = fit_row['c2_l'] if 'c2_l' in fit_row else 0.0
        
        # Build Covariance Matrix directly from fit results CSV
        cov = np.zeros((3, 3))
        cov[0, 0] = fit_row.get('y_0_err', 0.0)**2
        cov[1, 1] = fit_row.get('c_l_err', 0.0)**2
        cov[2, 2] = fit_row.get('c2_l_err', 0.0)**2
        
        cov[0, 1] = cov[1, 0] = fit_row.get('cov_y0_cl', 0.0)
        cov[0, 2] = cov[2, 0] = fit_row.get('cov_y0_c2l', 0.0)
        cov[1, 2] = cov[2, 1] = fit_row.get('cov_cl_c2l', 0.0)
        
        x_min_main, x_max_main = min(df['x_calc']), max(df['x_calc'])
        x_pad = (x_max_main - x_min_main) * 0.05
        x_vals_main = np.linspace(x_min_main - x_pad, x_max_main + x_pad, 500)
        
        y_cen, y_err = predict_global_fit(x_vals_main, y0_cen, cl_cen, c2l_cen, cov)
        
        c2_str = f", $c_{{2,l}} = {c2l_cen:.3f}$" if 'c2_l' in fit_row else ""
        eq_label = rf"Global Fit ($y_0 = {fmt_sci(y0_cen)}$, $c_l = {cl_cen:.3f}${c2_str})"
        
        ax_main.plot(x_vals_main, y_cen, linestyle='-', color='#003366', lw=2, label=eq_label, zorder=5)
        if not np.all(y_err == 0):
            ax_main.fill_between(x_vals_main, y_cen - y_err, y_cen + y_err, color='#003366', alpha=0.2, zorder=3)
            
        ax_main.set_xlim(x_min_main - x_pad, x_max_main + x_pad)

    ax_main.set_title('Global Overview', fontsize=15)
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
            y_cen_z, y_err_z = predict_global_fit(x_vals_z, y0_cen, cl_cen, c2l_cen, cov)
            ax_z.plot(x_vals_z, y_cen_z, linestyle='-', color='#003366', lw=1.8, zorder=5)
            if not np.all(y_err_z == 0):
                ax_z.fill_between(x_vals_z, y_cen_z - y_err_z, y_cen_z + y_err_z, color='#003366', alpha=0.2, zorder=3)
        
        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=5)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=12)
        if idx == 0: 
            ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=12)

    output_name = f'Plot_ST{lqcd_str}_Global_{prefix}_{A_types[A_index]}.svg'
    fig.savefig(output_name, bbox_inches='tight')
    plt.close(fig)