import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

# Plotting aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix' 

# Physical constants
hbar_c = 197.3269804
r0_fm = 0.4547
r0_MeV_central = r0_fm / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200

# Central derived constants from source script
B0_dim_central = B0_MeV * r0_MeV_central
f_pi_dim_central = f_pi_MeV * r0_MeV_central
mu_dim_central = mu_MeV * r0_MeV_central

# Branch configurations
param_sets = {
    'Branch_1': {'means': [0.00978833, 0.0262025, -0.124474], 'color': '#003366'},
    'Branch_2': {'means': [-0.0458536, 0.0815424, -0.124023], 'color': '#1D5288'},
    'Branch_3': {'means': [0.0458536, -0.0815424, 0.124023], 'color': '#437BB5'},
    'Branch_4': {'means': [-0.00978833, -0.0262025, 0.124474], 'color': '#8CB2E2'}
}

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\\pi/12']
A_index = 0

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: "_".join(s.split('_')[-2:]) if 'M_' in s else 'Unknown')
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

def chiral_model_dim(x, y_0, L2, L_prime, c_pipi):
    term1 = 4 * L_prime * x
    log_arg = np.maximum((2 * B0_dim_central * x) / (mu_dim_central**2), 1e-10) 
    term2_coeff = (3 * B0_dim_central) / (4 * np.pi**2 * f_pi_dim_central**2)
    term2 = term2_coeff * (4 * B0_dim_central * c_pipi - L_prime) * (np.log(log_arg) - 1) * (x**2)
    term3 = 2 * L2 * (x**2)
    return y_0 - term1 - term2 - term3

data_types = ['bare', 'smeared']

groups = ['M_i', 'M_ii', 'M_iii']

# Color palette across groups
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))
# Marker shapes across groups
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
group_marker_map = dict(zip(groups, markers[:len(groups)]))
# Colorblind-safe blue gradients anchored at #003366
fit_colors = {
    1: ['#003366'],
    2: ['#003366', '#3B82C4'],
    3: [None, '#003366', '#1D5288', '#437BB5', '#8CB2E2']
}
line_styles = ['-', '--', '-.', ':']

def fmt_sci(val):
    if np.isinf(val) or np.isnan(val): return "N/A"
    if val == 0: return "0"
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for prefix in data_types:
    ensembles = [g for g in groups if g in df['mass_group'].unique()]
    fig_grp, axes_grp = plt.subplots(1, len(ensembles), figsize=(18, 6), sharey=True)
    if len(ensembles) == 1: axes_grp = [axes_grp]
    
    fig_grp.suptitle(f'Dimensionless Log Fits - {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=18, fontweight='bold', y=1.02)
    
    for ax_grp, group in zip(axes_grp, ensembles):
        subset = df[df['mass_group'] == group]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Ensemble: {group}', fontsize=16)
        ax_grp.set_title(f'Ensemble: {group}', fontsize=16)
        
        x_list, x_err_list, y_list, y_err_list = [], [], [], []
        for _, row in subset.iterrows():
            x_val, x_e, y_val, y_e = calculate_dimensionless(row, prefix)
            x_list.append(x_val), x_err_list.append(x_e)
            y_list.append(y_val), y_err_list.append(y_e)
            
        x_arr, y_arr = np.array(x_list), np.array(y_list)
        x_err_arr, y_err_arr = np.array(x_err_list), np.array(y_err_list)
        
        x_min, x_max = min(x_arr), max(x_arr)
        x_pad = (x_max - x_min) * 0.05
        x_vals_plot = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        
        for axis in [ax, ax_grp]:
            axis.errorbar(x_arr, y_arr, xerr=x_err_arr, yerr=y_err_arr, fmt=group_marker_map[group], color=group_color_map[group], alpha=0.9, label=f'({group})', capsize=4, zorder=5) 
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
        
        group_fits = fit_df[(fit_df['Ensemble'] == group) & (fit_df['Data_Type'] == prefix)]
        
        for _, fit_row in group_fits.iterrows():
            branch = fit_row['Branch']
            if branch not in param_sets: continue
            
            y0_cen = fit_row['y0_central']
            L2_cen = fit_row['L2_central']
            y0_list = json.loads(fit_row['y0_tot_list'])
            L2_list = json.loads(fit_row['L2_tot_list'])
            
            means = param_sets[branch]['means']
            c_pipi_cen = (means[0] + means[1]/2) / 2
            L_prime_cen = 2 * B0_MeV * means[2] * r0_MeV_central
            branch_index = int(branch.rsplit('_', 1)[1])
            c = fit_colors[3][branch_index]
            line_style = line_styles[branch_index - 1]
            
            # Central evaluation
            y_cen = chiral_model_dim(x_vals_plot, y0_cen, L2_cen, L_prime_cen, c_pipi_cen)
            
            # Reconstruct error band from JSON bootstrap parameters
            y_band_samples = []
            for y0_s, L2_s in zip(y0_list, L2_list):
                y_band_samples.append(chiral_model_dim(x_vals_plot, y0_s, L2_s, L_prime_cen, c_pipi_cen))
            y_total_err = np.std(y_band_samples, axis=0)
            
            eq_label = rf"{branch.replace('_', ' ')} ($y_0 = {fmt_sci(y0_cen)}$, $\lambda'' = {fmt_sci(L2_cen)}$)"
            for axis in [ax, ax_grp]:
                axis.plot(x_vals_plot, y_cen, linestyle=line_style, color=c, lw=2, label=eq_label, zorder=5)
                if not np.all(y_total_err == 0):
                    axis.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=c, alpha=0.2, zorder=3)

        for axis in [ax, ax_grp]:
            axis.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
            axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) 
            axis.set_xlabel(r'$x = m_l r_0$', fontsize=14)
            axis.legend(fontsize=8, loc='best')

        ax.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'Plot_Log_{group}_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
        plt.close(fig)

    axes_grp[0].set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    fig_grp.tight_layout()
    fig_grp.savefig(f'Plot_Log_Grouped_{prefix}_{A_types[A_index]}.svg', bbox_inches='tight')
    plt.close(fig_grp)