import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import matplotlib.gridspec as gridspec

# 1. Keep text editable in SVG
mpl.rcParams['svg.fonttype'] = 'none'

# 2. Use a professional Serif/Times font environment
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # Matches the math rendering to the text font

# =============================================================================
# 1. Constants & Functions
# =============================================================================
hbar_c = 197.3269804 
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200 

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\pi/12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_global_{A_types[A_index]}.csv'

def calculate_physics(df_row, prefix):
    r0_a, r0_a_err = df_row[f'r0_a_{prefix}'], df_row[f'r0_a_{prefix}_err']
    a2sigma, a2sigma_err = df_row[f'a2sigma_{prefix}'], df_row[f'a2sigma_{prefix}_err']
    am_l = df_row['am_l']
    a = r0_MeV / r0_a
    a_err = a * (r0_a_err / r0_a + r0_MeV_err / r0_MeV)  # Combine relative errors of r0_a and r0_MeV
    m_l = am_l / a
    m_l_err = m_l * (a_err / a)
    sigma = a2sigma / (a**2)
    sigma_err = sigma * np.sqrt((a2sigma_err/a2sigma)**2 + (2*a_err/a)**2)
    return m_l, m_l_err, sigma, sigma_err

def chiral_eval(ml, sigma_0, lamb_2prime, lamb_prime, c_pipi):
    term1 = 4 * lamb_prime * ml
    log_arg = np.where((2 * B0_MeV * ml) / (mu_MeV**2) > 0, (2 * B0_MeV * ml) / (mu_MeV**2), 1e-10)
    term2_coeff = (3 * B0_MeV) / (4 * np.pi**2 * f_pi_MeV**2)
    term2 = (term2_coeff * (4 * B0_MeV * c_pipi - lamb_prime) * (np.log(log_arg) - 1)) * (ml**2)
    term3 = 2 * lamb_2prime * (ml**2)
    return sigma_0 - term1 - term2 - term3

def fmt_sci(val):
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

# =============================================================================
# 2. Load Data
# =============================================================================
df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda x: 'M_iii' if 'M_iii' in x else ('M_ii' if 'M_ii' in x else 'M_i'))
fit_df = pd.read_csv(filename)

groups = ['M_i', 'M_ii', 'M_iii']
data_types = ['bare', 'smeared']
colors = ['orange', 'blue', 'green', 'red']
data_color = '#1f1f1f' 

branch_labels = {
    'Branch_1': r"Set 2 $\lambda^\prime < 0$", 'Branch_2': r"Set 1 $\lambda^\prime < 0$",
    'Branch_3': r"Set 1 $\lambda^\prime > 0$", 'Branch_4': r"Set 2 $\lambda^\prime > 0$"
}
legend_order = ['Branch_3', 'Branch_2', 'Branch_4', 'Branch_1']
branch_colors = {'Branch_3': colors[0], 'Branch_2': colors[1], 'Branch_4': colors[2], 'Branch_1': colors[3]}

# =============================================================================
# 3. Plotting Logic (GridSpec for Main Plot + Zooms)
# =============================================================================
for dtype in data_types:
    # Set up the figure and GridSpec
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
    
    # Define axes: Top spans all columns, bottom are individual
    ax_main = fig.add_subplot(gs[0, :])
    ax_i = fig.add_subplot(gs[1, 0])
    ax_ii = fig.add_subplot(gs[1, 1], sharey=ax_i) # Optional: remove sharey if ranges are vastly different
    ax_iii = fig.add_subplot(gs[1, 2], sharey=ax_i)
    zoom_axes = [ax_i, ax_ii, ax_iii]
    
    fig.suptitle(f'Universal Chiral Log Fit - {dtype.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', fontsize=20, fontweight='bold', y=0.95)
    
    # Pre-calculate physical data for the entire dataset
    phys_data_all = df.apply(lambda row: calculate_physics(row, dtype), axis=1)
    df['m_l_calc'] = [x[0] for x in phys_data_all]
    df['m_l_err'] = [x[1] for x in phys_data_all]
    df['sigma_calc'] = [x[2] for x in phys_data_all]
    df['sigma_err'] = [x[3] for x in phys_data_all]
    
    dtype_fits = fit_df[fit_df['Data_Type'] == dtype]
    
    # ---------------------------------------------------------
    # 3a. Plot Main Global Figure (Top Panel)
    # ---------------------------------------------------------
    ax_main.set_title('Global Overview', fontsize=16)
    x_min_main = df['m_l_calc'].min()
    x_max_main = df['m_l_calc'].max()
    x_vals_main = np.linspace(x_min_main - 0.5, x_max_main + 0.5, 500)
    
    # Scatter all data
    ax_main.errorbar(df['m_l_calc'], df['sigma_calc'], xerr=df['m_l_err'], yerr=df['sigma_err'], 
                     fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

    handles_dict_main = {}
    labels_dict_main = {}

    for _, fit_row in dtype_fits.iterrows():
        branch = fit_row['Branch']
        c = branch_colors.get(branch, 'black')
        
        s0_cen = fit_row['sigma_0_central']
        l2_cen = fit_row['lamb_2prime_central']
        
        y_cen = chiral_eval(x_vals_main, s0_cen, l2_cen, fit_row['lamb_prime_central'], fit_row['c_pipi_central'])
        
        # Calculate error band traces for main plot
        s0_traces = np.array(json.loads(fit_row['s0_traces']))
        l2_traces = np.array(json.loads(fit_row['l2_traces']))
        lp_traces = np.array(json.loads(fit_row['lamb_prime_traces']))
        cp_traces = np.array(json.loads(fit_row['c_pipi_traces']))
        
        y_traces = np.zeros((len(s0_traces), len(x_vals_main)))
        for i in range(len(s0_traces)):
            y_traces[i, :] = chiral_eval(x_vals_main, s0_traces[i], l2_traces[i], lp_traces[i], cp_traces[i])
            
        y_err = np.std(y_traces, axis=0)
        eq_label = rf"{branch_labels[branch]} ($\sigma_{{0}} = {fmt_sci(s0_cen)}$, $\lambda^{{\prime\prime}} = {l2_cen:.0f}$)"
        
        line, = ax_main.plot(x_vals_main, y_cen, linestyle='--', color=c)
        ax_main.fill_between(x_vals_main, y_cen - y_err, y_cen + y_err, color=c, alpha=0.25)
        
        handles_dict_main[branch] = line
        labels_dict_main[branch] = eq_label

    ax_main.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
    ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax_main.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
    ax_main.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
    
    # Legend for Main Plot
    ordered_handles = [ax_main.get_legend_handles_labels()[0][0]]
    ordered_labels = ['Data']
    for b in legend_order:
        if b in handles_dict_main:
            ordered_handles.append(handles_dict_main[b])
            ordered_labels.append(labels_dict_main[b])
    ax_main.legend(ordered_handles, ordered_labels, fontsize=12, loc='best')

    # ---------------------------------------------------------
    # 3b. Plot Zoom Figures (Bottom Panels)
    # ---------------------------------------------------------
    for idx, group in enumerate(groups):
        ax_z = zoom_axes[idx]
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        ax_z.set_title(f'Zoom: {group}', fontsize=14)
        
        x_min_z = subset['m_l_calc'].min()
        x_max_z = subset['m_l_calc'].max()
        pad = (x_max_z - x_min_z) * 0.2 if (x_max_z - x_min_z) > 0 else 0.5
        x_vals_z = np.linspace(x_min_z - pad, x_max_z + pad, 200)

        # Scatter zoom data
        ax_z.errorbar(subset['m_l_calc'], subset['sigma_calc'], xerr=subset['m_l_err'], yerr=subset['sigma_err'], 
                      fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

        for _, fit_row in dtype_fits.iterrows():
            branch = fit_row['Branch']
            c = branch_colors.get(branch, 'black')
            
            s0_cen = fit_row['sigma_0_central']
            l2_cen = fit_row['lamb_2prime_central']
            
            y_cen = chiral_eval(x_vals_z, s0_cen, l2_cen, fit_row['lamb_prime_central'], fit_row['c_pipi_central'])
            
            # Recalculate error bands for the specific zoomed x-axis
            s0_traces = np.array(json.loads(fit_row['s0_traces']))
            l2_traces = np.array(json.loads(fit_row['l2_traces']))
            lp_traces = np.array(json.loads(fit_row['lamb_prime_traces']))
            cp_traces = np.array(json.loads(fit_row['c_pipi_traces']))
            
            y_traces = np.zeros((len(s0_traces), len(x_vals_z)))
            for i in range(len(s0_traces)):
                y_traces[i, :] = chiral_eval(x_vals_z, s0_traces[i], l2_traces[i], lp_traces[i], cp_traces[i])
                
            y_err = np.std(y_traces, axis=0)
            
            ax_z.plot(x_vals_z, y_cen, linestyle='--', color=c)
            ax_z.fill_between(x_vals_z, y_cen - y_err, y_cen + y_err, color=c, alpha=0.25)

        ax_z.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
        ax_z.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax_z.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
        
        if idx == 0:
            ax_z.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)

    fig.savefig(f"Plot_Log_Global_{dtype}_{A_types[A_index]}.svg", bbox_inches='tight')
    plt.close(fig)

print("GridSpec Universal Fit PNGs generated.")