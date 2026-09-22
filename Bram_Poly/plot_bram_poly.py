# Plots the results of the fits performed by fit_bram_poly.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import json

# Plotting aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix' 

# =============================================================================
# 1. Constants & Scale Conversions
# =============================================================================
# Physical constants matching the fit script
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_MeV = r0_fm / hbar_c

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
A_index = 1

# =============================================================================
# 2. Data Processing Function & Dimensionless Models
# =============================================================================
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

def poly0(x, y_0):
    return np.full_like(x, y_0, dtype=float)

def poly1(x, y_0, a, b):
    return y_0 + a * x + b * x**2

def poly2(x, y_0, b):
    return y_0 + b * x**2

# Helper function to compute error bands using the covariance matrix
def get_fit_band(x_vals, model_name, popt, cov, n_samples=1000):
    samples = np.random.multivariate_normal(popt, cov, n_samples)
    y_samples = []
    
    for s in samples:
        if model_name == 'Poly0':
            y_samples.append(poly0(x_vals, s[0]))
        elif model_name == 'Poly1':
            y_samples.append(poly1(x_vals, s[0], s[1], s[2]))
        elif model_name == 'Poly2':
            y_samples.append(poly2(x_vals, s[0], s[1]))
            
    return np.std(y_samples, axis=0)

# =============================================================================
# 3. Load Data & Define Registries
# =============================================================================
data_file = f'Data/data_bram_{A_types[A_index]}.csv'
filename = f'Bram_Poly/fit_results_poly_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: "_".join(s.split('_')[-2:]))
fit_df = pd.read_csv(filename)

data_types = ['bare', 'smeared']
groups = df['mass_group'].unique() 

fit_colors = {
    'Poly0': '#003366', 
    'Poly1': '#1D5288',
    'Poly2': '#437BB5'
}

# New dictionary for line styles
fit_linestyles = {
    'Poly0': '-', 
    'Poly1': '--',
    'Poly2': '-.'
}

# Color and marker palettes across groups
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
group_marker_map = dict(zip(groups, markers[:len(groups)]))

# =============================================================================
# 4. Plotting Routine
# =============================================================================
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
    
    fig.suptitle(f'Polynomial Fits - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=20, fontweight='bold', y=0.95)

    # Process and append calculated columns to the dataframe cleanly
    df[['x_calc', 'x_err', 'y_calc', 'y_err']] = df.apply(
        lambda row: pd.Series(calculate_dimensionless(row, prefix)), axis=1
    )
    
    dtype_fits = fit_df[fit_df['Data_Type'] == prefix]
    
    # ---------------------------------------------------------
    # MAIN AXIS (GLOBAL OVERVIEW)
    # ---------------------------------------------------------
    x_min_main, x_max_main = df['x_calc'].min(), df['x_calc'].max()
    x_pad_main = (x_max_main - x_min_main) * 0.05
    x_vals_main = np.linspace(x_min_main - x_pad_main, x_max_main + x_pad_main, 500)
    ax_main.set_xlim(x_min_main - x_pad_main, x_max_main + x_pad_main)
    
    # Plot experimental data
    for group in groups:
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
        ax_main.errorbar(
            subset['x_calc'], subset['y_calc'],
            xerr=subset['x_err'], yerr=subset['y_err'],
            fmt=group_marker_map[group], color=group_color_map[group],
            alpha=0.9, capsize=4, label=f'Ensemble: {group}', zorder=5
        )

    # Plot fits and error bands
    if not dtype_fits.empty:
        for _, fit_row in dtype_fits.iterrows():
            model = fit_row['Model']
            if model not in fit_colors: continue

            cov = np.array(json.loads(fit_row['cov_matrix']))
            p_val = fit_row['p_value']
            
            # Extract parameters, evaluate function, and build the custom legend label
            if model == 'Poly0':
                popt = [fit_row['y_0']]
                y_cen = poly0(x_vals_main, *popt)
                label_str = f"Poly0: y0={popt[0]:.3g}, p-value={p_val:.3g}"
            elif model == 'Poly1':
                popt = [fit_row['y_0'], fit_row['a'], fit_row['b']]
                y_cen = poly1(x_vals_main, *popt)
                label_str = f"Poly1: y0={popt[0]:.3g}, a={popt[1]:.3g}, b={popt[2]:.3g}, p-value={p_val:.3g}"
            elif model == 'Poly2':
                popt = [fit_row['y_0'], fit_row['b']]
                y_cen = poly2(x_vals_main, *popt)
                label_str = f"Poly2: y0={popt[0]:.3g}, b={popt[1]:.3g}, p-value={p_val:.3g}"

            y_err = get_fit_band(x_vals_main, model, popt, cov)
            c_color = fit_colors[model]
            l_style = fit_linestyles[model]
            
            # Plot center line with the designated linestyle and dynamic label
            ax_main.plot(x_vals_main, y_cen, color=c_color, linestyle=l_style, lw=2, label=label_str, zorder=4)
            ax_main.fill_between(x_vals_main, y_cen - y_err, y_cen + y_err, color=c_color, alpha=0.2, zorder=3)

    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=14)
    # Using a slightly smaller font size for the legend so the equations fit nicely
    ax_main.legend(fontsize=9, loc='best')

    # ---------------------------------------------------------
    # ZOOM AXES (INDIVIDUAL ENSEMBLES)
    # ---------------------------------------------------------
    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=14)
        x_min_z, x_max_z = subset['x_calc'].min(), subset['x_calc'].max()
        x_pad_z = (x_max_z - x_min_z) * 0.1 if x_max_z > x_min_z else x_min_z * 0.1
        x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 100)
        
        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        
        # Plot experimental data for zoom
        ax_z.errorbar(
            subset['x_calc'], subset['y_calc'], 
            xerr=subset['x_err'], yerr=subset['y_err'], 
            fmt=group_marker_map[group], color=group_color_map[group], 
            alpha=0.9, capsize=4, zorder=5
        )
        
        # Plot fits and error bands for zoom
        if not dtype_fits.empty:
            for _, fit_row in dtype_fits.iterrows():
                model = fit_row['Model']
                if model not in fit_colors: continue

                cov = np.array(json.loads(fit_row['cov_matrix']))
                
                if model == 'Poly0':
                    popt = [fit_row['y_0']]
                    y_cen_z = poly0(x_vals_z, *popt)
                elif model == 'Poly1':
                    popt = [fit_row['y_0'], fit_row['a'], fit_row['b']]
                    y_cen_z = poly1(x_vals_z, *popt)
                elif model == 'Poly2':
                    popt = [fit_row['y_0'], fit_row['b']]
                    y_cen_z = poly2(x_vals_z, *popt)

                y_err_z = get_fit_band(x_vals_z, model, popt, cov)
                c_color = fit_colors[model]
                l_style = fit_linestyles[model]
                
                ax_z.plot(x_vals_z, y_cen_z, color=c_color, linestyle=l_style, lw=2, zorder=4)
                ax_z.fill_between(x_vals_z, y_cen_z - y_err_z, y_cen_z + y_err_z, color=c_color, alpha=0.2, zorder=3)

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=14)
        if idx == 0: 
            ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)

    # Save output for each data type
    out_name = f'Bram_Poly/Plot_Poly_{prefix}_{A_types[A_index]}.svg'
    fig.savefig(out_name, bbox_inches='tight')
    print(f"Saved plot: {out_name}")
    plt.close(fig)