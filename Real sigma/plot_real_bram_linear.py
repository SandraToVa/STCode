# In this plot I compare the linear fit y(x) obtained completely for the real sigma values and the real pion mass
# to the brambilla data. In this case there is no more free parameters so it's only a comparison.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.stats import chi2

# Plotting aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix' 

# -----------------------------------------------------------------------------
# 1. Physical Input Parameters & Constants
# -----------------------------------------------------------------------------
hbarc = 0.1973269804      # GeV * fm
m_pi = 0.140              # GeV (140 MeV)
B_0 = 2.7                 # GeV (2700 MeV)
sigma = 0.187             # GeV^2
r0_fm = 0.4547            # fm
r0_fm_err = 0.0064        # fm

# Convert both value and error to GeV^-1 
r0_gev = r0_fm / hbarc
r0_gev_err = r0_fm_err / hbarc

A_types = ['Ar0', 'Api12']
A_latex = ['r_0', '\\pi/12']
A_index = 1


# Two branches for f
branch_params = {
    'positive': {'f': 0.124,  'f_err': 0.005, 'color': '#003366', 'ls': '-'},
    'negative': {'f': -0.124, 'f_err': 0.005, 'color': '#3B82C4', 'ls': '--'}
}

# -----------------------------------------------------------------------------
# 2. Helper Functions for Theoretical Curve & Error Propagation
# -----------------------------------------------------------------------------
def compute_branch_theory(f_val, f_err, x_vals):
    """
    Returns 5 values:
    y0        : Central intercept
    y0_err    : Uncertainty on y0
    lambda_p  : Slope parameter lambda'
    y_pred    : Predicted y(x) array
    y_err     : Uncertainty array y_err(x)
    """
    y0 = (r0_gev**2) * (sigma + 4 * f_val * (m_pi**2))
    lambda_p = 2 * B_0 * f_val * r0_gev
    y_pred = y0 - 4 * lambda_p * x_vals
    
    # Uncertainty on y0 (evaluated at x = 0)
    dy0_dr0_gev = 2 * r0_gev * sigma + 8 * f_val * (m_pi**2) * r0_gev
    dy0_df = 4 * (m_pi**2) * (r0_gev**2)
    y0_err = np.sqrt((dy0_dr0_gev * r0_gev_err)**2 + (dy0_df * f_err)**2)
    
    # Point-by-point uncertainty on y(x)
    dy_dr0_gev = 2 * r0_gev * sigma + 8 * f_val * (m_pi**2) * r0_gev - 8 * B_0 * f_val * x_vals
    dy_df = 4 * (m_pi**2) * (r0_gev**2) - 8 * B_0 * r0_gev * x_vals
    y_err = np.sqrt((dy_dr0_gev * r0_gev_err)**2 + (dy_df * f_err)**2)
    
    return y0, y0_err, lambda_p, y_pred, y_err


def calculate_chi2_dof(x_data, y_data, x_err, y_err, y0, lambda_p):
    """
    Computes chi^2, dof, reduced chi^2, and p-value using the effective variance method.
    """
    y_pred = y0 - 4 * lambda_p * x_data
    dy_dx = -4 * lambda_p
    
    # Effective variance accounting for both x and y errors
    sigma_eff2 = (y_err**2) + ((dy_dx * x_err)**2)
    
    chi2_val = np.sum(((y_data - y_pred)**2) / sigma_eff2)
    dof = len(x_data) # Since parameters are externally fixed, dof = N_data
    chi2_red = chi2_val / dof if dof > 0 else np.nan
    
    # p-value: probability of obtaining a chi^2 as extreme as observed by chance
    p_value = chi2.sf(chi2_val, dof) if dof > 0 else np.nan
    
    return chi2_val, dof, chi2_red, p_value

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

# -----------------------------------------------------------------------------
# 3. Data Loading & Preparation
# -----------------------------------------------------------------------------

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: "_".join(s.split('_')[-2:]))

groups = df['mass_group'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
group_marker_map = dict(zip(groups, markers[:len(groups)]))

data_types = ['bare', 'smeared']

# -----------------------------------------------------------------------------
# 4. Plotting Loop for Bare and Smeared Data
# -----------------------------------------------------------------------------
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
    
    fig.suptitle(f'Theoretical Linear Plot - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Calculate x and y data points for this data type
    phys_data_all = df.apply(lambda row: calculate_dimensionless(row, prefix), axis=1)
    df['x_calc'], df['x_err'] = [d[0] for d in phys_data_all], [d[1] for d in phys_data_all]
    df['y_calc'], df['y_err'] = [d[2] for d in phys_data_all], [d[3] for d in phys_data_all]
    
    # Axis bounds
    x_min_main, x_max_main = df['x_calc'].min(), df['x_calc'].max()
    x_pad_main = (x_max_main - x_min_main) * 0.05
    x_vals_main = np.linspace(x_min_main - x_pad_main, x_max_main + x_pad_main, 500)
    ax_main.set_xlim(x_min_main - x_pad_main, x_max_main + x_pad_main)
    
    # Plot ensemble data points on main axis
    for group in groups:
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
        ax_main.errorbar(
            subset['x_calc'], subset['y_calc'],
            xerr=subset['x_err'], yerr=subset['y_err'],
            fmt=group_marker_map[group], color=group_color_map[group],
            alpha=0.9, capsize=4, label=f'Ensemble: {group}', zorder=5
        )

    # Plot theoretical predictions for both branches
    for branch, p in branch_params.items():
        y0, y0_err, lambda_p, y_main, y_err_main = compute_branch_theory(p['f'], p['f_err'], x_vals_main)
        
        # Calculate chi^2 / dof against the current dataset
        chi2_val, dof, chi2_red, p_val = calculate_chi2_dof(
            df['x_calc'].values, df['y_calc'].values,
            df['x_err'].values, df['y_err'].values,
            y0, lambda_p
        )

        label = (rf"{branch.capitalize()} Branch ($f={p['f']:+.3f}$): " 
            rf"$y_0 = {y0:.3f} \pm {y0_err:.3f}$, "
            rf"$\chi^2/\mathrm{{dof}} = {chi2_red:.2f}$, $p = {p_val:.3f}$")
        
        # Main plot line & band
        ax_main.plot(x_vals_main, y_main, linestyle=p['ls'], color=p['color'], label=label, lw=2, zorder=6)
        ax_main.fill_between(x_vals_main, y_main - y_err_main, y_main + y_err_main, color=p['color'], alpha=0.18, zorder=3)
        
        # Zoom subplots lines & bands
        for idx, group in enumerate(groups):
            ax_z = zoom_axes[idx]
            subset = df[df['mass_group'] == group]
            if subset.empty: continue
            
            x_min_z, x_max_z = subset['x_calc'].min(), subset['x_calc'].max()
            x_pad_z = (x_max_z - x_min_z) * 0.05 if x_max_z > x_min_z else x_min_z * 0.05
            x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 100)
            
            _, _, _, y_z, y_err_z = compute_branch_theory(p['f'], p['f_err'], x_vals_z)
            
            ax_z.plot(x_vals_z, y_z, linestyle=p['ls'], color=p['color'], lw=2, zorder=6)
            ax_z.fill_between(x_vals_z, y_z - y_err_z, y_z + y_err_z, color=p['color'], alpha=0.18, zorder=3)

    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=14)
    ax_main.legend(fontsize=11, loc='best')

    # Format zoom subplots with data points
    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=14)
        x_min_z, x_max_z = subset['x_calc'].min(), subset['x_calc'].max()
        x_pad_z = (x_max_z - x_min_z) * 0.05 if x_max_z > x_min_z else x_min_z * 0.05
        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        
        ax_z.errorbar(
            subset['x_calc'], subset['y_calc'],
            xerr=subset['x_err'], yerr=subset['y_err'],
            fmt=group_marker_map[group], color=group_color_map[group],
            alpha=0.9, capsize=4, zorder=5
        )

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=14)
        if idx == 0: ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)

    fig.savefig(f'Plot_Linear_Theoretical_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig)