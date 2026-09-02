import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

# Plotting aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix' 

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
A_index = 1

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_linear_global_{A_types[A_index]}.csv'

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

data_types = ['bare', 'smeared']
groups = df['mass_group'].unique()
data_color = '#1f1f1f'
branch_colors = {'positive': 'blue', 'negative': 'red'}

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    if val == 0: return "0"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, len(groups), height_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
    ax_main = fig.add_subplot(gs[0, :])
    
    # CORRECTED BLOCK: Standard loop to avoid NameError
    zoom_axes = []
    for i in range(len(groups)):
        if i == 0:
            ax = fig.add_subplot(gs[1, i])
        else:
            ax = fig.add_subplot(gs[1, i], sharey=zoom_axes[0])
        zoom_axes.append(ax)
    
    fig.suptitle(f'Global Dimensionless Linear Fit - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=20, fontweight='bold', y=0.95)
    
    phys_data_all = df.apply(lambda row: calculate_dimensionless(row, prefix), axis=1)
    df['x_calc'], df['x_err'] = [d[0] for d in phys_data_all], [d[1] for d in phys_data_all]
    df['y_calc'], df['y_err'] = [d[2] for d in phys_data_all], [d[3] for d in phys_data_all]
    
    dtype_fits = fit_df[fit_df['Data_Type'] == prefix]
    
    ax_main.set_title('Global Overview', fontsize=16)
    x_min_main, x_max_main = df['x_calc'].min(), df['x_calc'].max()
    x_pad_main = (x_max_main - x_min_main) * 0.05
    x_vals_main = np.linspace(x_min_main - x_pad_main, x_max_main + x_pad_main, 500)
    ax_main.set_xlim(x_min_main - x_pad_main, x_max_main + x_pad_main)
    
    ax_main.errorbar(df['x_calc'], df['y_calc'], xerr=df['x_err'], yerr=df['y_err'], fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

    if not dtype_fits.empty:
        for _, fit_row in dtype_fits.iterrows():
            branch = fit_row['Branch']
            y0_cen = fit_row['y_0_central']
            y0_err = fit_row['y_0_total_err']
            L_prime = fit_row['L_prime_central']
            
            y_cen = y0_cen - 4 * L_prime * x_vals_main
            y_total_err = np.ones_like(x_vals_main) * y0_err
            
            c = branch_colors.get(branch, 'green')
            eq_label = rf"{branch.capitalize()} Universal Fit ($y_0 = {fmt_sci(y0_cen)}$)"
            ax_main.plot(x_vals_main, y_cen, linestyle='-', color=c, label=eq_label)
            if not np.all(y_total_err == 0):
                ax_main.fill_between(x_vals_main, y_cen - y_total_err, y_cen + y_total_err, color=c, alpha=0.25)

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
        x_min_z, x_max_z = subset['x_calc'].min(), subset['x_calc'].max()
        x_pad_z = (x_max_z - x_min_z) * 0.05 if x_max_z > x_min_z else x_min_z * 0.05
        x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 100)
        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        
        ax_z.errorbar(subset['x_calc'], subset['y_calc'], xerr=subset['x_err'], yerr=subset['y_err'], fmt='o', color=data_color, alpha=0.9, capsize=4, zorder=5)
        
        if not dtype_fits.empty:
            for _, fit_row in dtype_fits.iterrows():
                branch = fit_row['Branch']
                y0_cen = fit_row['y_0_central']
                y0_err = fit_row['y_0_total_err']
                L_prime = fit_row['L_prime_central']
                
                y_cen_z = y0_cen - 4 * L_prime * x_vals_z
                y_total_err_z = np.ones_like(x_vals_z) * y0_err
                
                c = branch_colors.get(branch, 'green')
                ax_z.plot(x_vals_z, y_cen_z, linestyle='-', color=c)
                if not np.all(y_total_err_z == 0):
                    ax_z.fill_between(x_vals_z, y_cen_z - y_total_err_z, y_cen_z + y_total_err_z, color=c, alpha=0.25)

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=14)
        if idx == 0: ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)

    fig.savefig(f'Plot_Linear_Global_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig)