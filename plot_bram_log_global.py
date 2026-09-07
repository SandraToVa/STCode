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

# Physical constants matching the fit script
hbar_c = 197.3269804
r0_fm = 0.4547
r0_MeV_central = r0_fm / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200
B0_dim_cen = B0_MeV * r0_MeV_central
f_pi_dim_cen = f_pi_MeV * r0_MeV_central
mu_dim_cen = mu_MeV * r0_MeV_central

branch_colors = {
    'Branch_1': '#003366', 'Branch_2': '#1D5288',
    'Branch_3': '#437BB5', 'Branch_4': '#8CB2E2'
}

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
A_index = 0

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_global_{A_types[A_index]}.csv'

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

def chiral_model_universal_dim(x, y_0, L2, L_prime, c_pipi, b0, fpi, mu):
    term1 = 4 * L_prime * x
    log_arg = np.maximum((2 * b0 * x) / (mu**2), 1e-10)
    term2_coeff = (3 * b0) / (4 * np.pi**2 * fpi**2)
    term2 = term2_coeff * (4 * b0 * c_pipi - L_prime) * (np.log(log_arg) - 1) * (x**2)
    term3 = 2 * L2 * (x**2)
    return y_0 - term1 - term2 - term3

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

data_types = ['bare', 'smeared']
groups = df['mass_group'].unique()

# Color palette across groups
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))
# Marker shapes across groups
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
group_marker_map = dict(zip(groups, markers[:len(groups)]))
# Colorblind-safe blue gradients anchored at #003366
#fit_colors = {
#    1: ['#003366'],
#    2: ['#003366', '#3B82C4'],
#    3: [None, '#003366', '#1D5288', '#437BB5', '#8CB2E2']
#}
line_styles = ['-', '--', '-.', ':']

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
    
    fig.suptitle(f'Global Logarithmic Fit - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=20, fontweight='bold', y=0.95)

    # Plotting the data points with error bars
    phys_data_all = df.apply(lambda row: calculate_dimensionless(row, prefix), axis=1)
    df['x_calc'], df['x_err'] = [d[0] for d in phys_data_all], [d[1] for d in phys_data_all]
    df['y_calc'], df['y_err'] = [d[2] for d in phys_data_all], [d[3] for d in phys_data_all]
    
    dtype_fits = fit_df[fit_df['Data_Type'] == prefix]
    
    #ax_main.set_title('Global Overview', fontsize=16)
    x_min_main, x_max_main = df['x_calc'].min(), df['x_calc'].max()
    x_pad_main = (x_max_main - x_min_main) * 0.05
    x_vals_main = np.linspace(x_min_main - x_pad_main, x_max_main + x_pad_main, 500)
    ax_main.set_xlim(x_min_main - x_pad_main, x_max_main + x_pad_main)
    
    for group in groups:
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
        ax_main.errorbar(
            subset['x_calc'], subset['y_calc'],
            xerr=subset['x_err'], yerr=subset['y_err'],
            fmt=group_marker_map[group], color=group_color_map[group],
            alpha=0.9, capsize=4, label=f'Ensemble: {group}', zorder=5
        )

    if not dtype_fits.empty:
        for _, fit_row in dtype_fits.iterrows():
            branch = fit_row['Branch']
            if branch not in branch_colors: continue
                
            y0_cen = fit_row['y0_central']
            L2_cen = fit_row['L2_central']
            L_prime_cen = fit_row['L_prime_central']
            c_pipi_cen = fit_row['c_pipi_central']
            
            y0_traces = json.loads(fit_row['y0_traces'])
            L2_traces = json.loads(fit_row['L2_traces'])
            L_prime_traces = json.loads(fit_row['L_prime_traces'])
            c_pipi_traces = json.loads(fit_row['c_pipi_traces'])
            
            c_color = branch_colors[branch]
            y_cen = chiral_model_universal_dim(x_vals_main, y0_cen, L2_cen, L_prime_cen, c_pipi_cen, B0_dim_cen, f_pi_dim_cen, mu_dim_cen)
            
            # Reconstruct bootstrap error band
            y_band_samples = []
            for y0_s, L2_s, Lp_s, cp_s in zip(y0_traces, L2_traces, L_prime_traces, c_pipi_traces):
                y_band_samples.append(chiral_model_universal_dim(x_vals_main, y0_s, L2_s, Lp_s, cp_s, B0_dim_cen, f_pi_dim_cen, mu_dim_cen))
            y_total_err = np.std(y_band_samples, axis=0)
            
            eq_label = rf"{branch.replace('_', ' ')} ($y_0 = {y0_cen:.2f}$, $\lambda'' = {L2_cen:.2f}$)"
            branch_index = int(branch.rsplit('_', 1)[1])
            ax_main.plot(x_vals_main, y_cen, linestyle=line_styles[branch_index - 1], color=c_color, label=eq_label)
            if not np.all(y_total_err == 0):
                ax_main.fill_between(x_vals_main, y_cen - y_total_err, y_cen + y_total_err, color=c_color, alpha=0.2)

    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=14)
    ax_main.legend(fontsize=10, loc='best')

    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=14)
        x_min_z, x_max_z = subset['x_calc'].min(), subset['x_calc'].max()
        x_pad_z = (x_max_z - x_min_z) * 0.05 if x_max_z > x_min_z else x_min_z * 0.05
        x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 100)
        ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
        
        ax_z.errorbar(subset['x_calc'], subset['y_calc'], xerr=subset['x_err'], yerr=subset['y_err'], fmt=group_marker_map[group], color=group_color_map[group], alpha=0.9, capsize=4, zorder=5)
        
        if not dtype_fits.empty:
            for _, fit_row in dtype_fits.iterrows():
                branch = fit_row['Branch']
                if branch not in branch_colors: continue
                branch_index = int(branch.rsplit('_', 1)[1])
                
                y0_cen = fit_row['y0_central']
                L2_cen = fit_row['L2_central']
                L_prime_cen = fit_row['L_prime_central']
                c_pipi_cen = fit_row['c_pipi_central']
                
                y0_traces = json.loads(fit_row['y0_traces'])
                L2_traces = json.loads(fit_row['L2_traces'])
                L_prime_traces = json.loads(fit_row['L_prime_traces'])
                c_pipi_traces = json.loads(fit_row['c_pipi_traces'])
                
                c_color = branch_colors[branch]
                y_cen_z = chiral_model_universal_dim(x_vals_z, y0_cen, L2_cen, L_prime_cen, c_pipi_cen, B0_dim_cen, f_pi_dim_cen, mu_dim_cen)
                
                y_band_samples_z = []
                for y0_s, L2_s, Lp_s, cp_s in zip(y0_traces, L2_traces, L_prime_traces, c_pipi_traces):
                    y_band_samples_z.append(chiral_model_universal_dim(x_vals_z, y0_s, L2_s, Lp_s, cp_s, B0_dim_cen, f_pi_dim_cen, mu_dim_cen))
                y_total_err_z = np.std(y_band_samples_z, axis=0)
                
                ax_z.plot(x_vals_z, y_cen_z, linestyle=line_styles[branch_index - 1], color=c_color, lw=2, zorder=5)
                if not np.all(y_total_err_z == 0):
                    ax_z.fill_between(x_vals_z, y_cen_z - y_total_err_z, y_cen_z + y_total_err_z, color=c_color, alpha=0.2, zorder=3)

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=14)
        if idx == 0: ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)

    fig.savefig(f'Plot_Log_Global_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig)