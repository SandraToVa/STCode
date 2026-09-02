# Reads both data_bram.csv and fit_results_log.csv, 
# and generates the final publication-ready figures.

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import matplotlib as mpl
# 1. Keep text editable in SVG
mpl.rcParams['svg.fonttype'] = 'none'

# 2. Use a professional Serif/Times font environment
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['STIXGeneral', 'Times New Roman', 'Times', 'Nimbus Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'  # Matches the math rendering to the text font
# =============================================================================
# 1. Constants & Unit Conversions (Converting GeV to fm^-1)
# =============================================================================
hbar_c = 197.3269804 # MeV*fm
r0_fm = 0.4547
r0_fm_err = 0.0064
r0_MeV = r0_fm / hbar_c
r0_MeV_err = r0_fm_err / hbar_c
B0_MeV = 2700
f_pi_MeV = 92
mu_MeV = 200 # order of magnitude of pion mass in MeV, used as scale for chiral logs

A_types = ['Ar0', 'Api12']
A_latex = ['r_0','\pi/12']
# Change index to select r0 or pi12 data
A_index = 1

data_file = f'data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_log_{A_types[A_index]}_v2.csv'


# =============================================================================
# 2. Data Processing Function & Chiral Model
# =============================================================================
# Same as in the fit_model_log.py, but included here for completeness and to avoid circular imports
def calculate_physics(df_row, prefix):
    r0_a = df_row[f'r0_a_{prefix}']
    r0_a_err = df_row[f'r0_a_{prefix}_err']
    a2sigma = df_row[f'a2sigma_{prefix}']
    a2sigma_err = df_row[f'a2sigma_{prefix}_err']
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
    term2 = (term2_coeff * (4 * B0_MeV * c_pipi - lamb_prime) * ( np.log(log_arg) - 1) ) * (ml**2)
    term3 = 2 * lamb_2prime * (ml**2)
    return sigma_0 - term1 - term2 - term3

# =============================================================================
# 3. Load dada and fit results
# =============================================================================

# Experimental data
df = pd.read_csv(data_file, sep=r'\s+')

def get_ensemble_group(ens_str):
    if 'M_iii' in ens_str: return 'M_iii'
    if 'M_ii' in ens_str:  return 'M_ii'
    if 'M_i' in ens_str:   return 'M_i'
    return 'Unknown'

df['mass_group'] = df['Ensemble'].apply(get_ensemble_group)

# Fit data
fit_df = pd.read_csv(filename)

# =============================================================================
# 4. Plotting Logic
# =============================================================================
groups = ['M_i', 'M_ii', 'M_iii']
data_types = ['bare', 'smeared']

# Custom Color Palette
# Pretty colors for the branches (but they give bad mixing)
#colors = ["#F4B942", "#4A89C7", "#4DB37B", "#F27457", "#8E63AC", "#5E6C71"]
colors = ['orange', 'blue', 'green', 'red']
data_color = '#1f1f1f' 

# Custom Legend Mapping
branch_labels = {
    'Branch_1': r"Set 2 $\lambda^\prime < 0$",
    'Branch_2': r"Set 1 $\lambda^\prime < 0$",
    'Branch_3': r"Set 1 $\lambda^\prime > 0$",
    'Branch_4': r"Set 2 $\lambda^\prime > 0$"
}

# Desired Legend Order
legend_order = ['Branch_3', 'Branch_2', 'Branch_4', 'Branch_1']

# Map branches to colors based on the requested order
branch_colors = {
    'Branch_3': colors[0],
    'Branch_2': colors[1],
    'Branch_4': colors[2],
    'Branch_1': colors[3]
}

# Cientific notation in the plot
def fmt_sci(val):
    base, exp = "{:.2e}".format(val).split('e')
    return f"{float(base):.2f} \\times 10^{{{int(exp)}}}"

for dtype in data_types:
    # Prepare Grouped Figure - matched to first script (sharey=True, figsize)
    fig_grp, axes_grp = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig_grp.suptitle(f'Chiral Log Fits - {dtype.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)', 
                     fontsize=18, fontweight='bold', y=1.02)
    
    for idx, group in enumerate(groups):
        subset = df[df['mass_group'] == group]
        if subset.empty: continue
            
        phys_data = subset.apply(lambda row: calculate_physics(row, dtype), axis=1)
        m_l_data = np.array([x[0] for x in phys_data])
        m_l_err = np.array([x[1] for x in phys_data])
        sigma_data = np.array([x[2] for x in phys_data])
        sigma_err = np.array([x[3] for x in phys_data])
        
        # Prepare Individual Figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax_grp = axes_grp[idx]
        
        # Apply title to grouped subplot
        ax_grp.set_title(f'Ensemble: {group}', fontsize=16)
        ax.set_title(f'Ensemble: {group}', fontsize=16)
        
        # Consistent X-axis range (0.2 padding from script 1)
        x_min, x_max = np.min(m_l_data), np.max(m_l_data)
        x_vals_math = np.linspace(x_min - (x_max-x_min)*0.2, x_max + (x_max-x_min)*0.2, 200)
        x_vals_plot = x_vals_math

        # Plot experimental data
        for axis in [ax, ax_grp]:
            axis.errorbar(m_l_data, sigma_data, xerr=m_l_err, yerr=sigma_err, 
                          fmt='o', color=data_color, alpha=0.9, capsize=4, label='Data', zorder=5)

        group_fits = fit_df[(fit_df['Ensemble'] == group) & (fit_df['Data_Type'] == dtype)]
        
        # Dictionary for custom legend sorting
        handles_dict = {}
        labels_dict = {}
        
        for _, fit_row in group_fits.iterrows():
            branch = fit_row['Branch']
            c = branch_colors.get(branch, 'black')
            
            # 1. Evaluate central line
            y_cen = chiral_eval(x_vals_math, fit_row['sigma_0_central'], fit_row['lamb_2prime_central'],
                                fit_row['lamb_prime_central'], fit_row['c_pipi_central'])
            
            # 2. Load the 1000 Monte Carlo parameter traces
            s0_traces = np.array(json.loads(fit_row['s0_tot_list']))
            l2_traces = np.array(json.loads(fit_row['l2_tot_list']))
            lp_traces = np.array(json.loads(fit_row['lamb_prime_list']))
            cp_traces = np.array(json.loads(fit_row['c_pipi_list']))
            
            # 3. Calculate all 1000 y-curves
            y_traces = np.zeros((len(s0_traces), len(x_vals_math)))
            for i in range(len(s0_traces)):
                y_traces[i, :] = chiral_eval(x_vals_math, s0_traces[i], l2_traces[i], 
                                             lp_traces[i], cp_traces[i])
            
            # 4. The true 1-sigma error band (accounts for all correlations!)
            y_total_err = np.std(y_traces, axis=0)
            
            # Added mu to the aesthetic label (formatted to 0 decimal places for cleanliness)
            eq_label = rf"{branch_labels[branch]} ($\sigma_0 = {fmt_sci(fit_row['sigma_0_central'])}$, $\lambda^{{\prime\prime}} = {fit_row['lamb_2prime_central']:.0f}$)"

            for axis in [ax, ax_grp]:
                line, = axis.plot(x_vals_plot, y_cen, linestyle='--', color=c)
                axis.fill_between(x_vals_plot, y_cen - y_total_err, y_cen + y_total_err, color=c, alpha=0.25)
                
                if axis == ax:
                    handles_dict[branch] = line
                    labels_dict[branch] = eq_label

        # Formatting loop
        for axis in [ax, ax_grp]:
            axis.tick_params(direction='in', top=True, right=True, bottom=True, left=True, length=6)
            # Scientific Notation directly to Y axis logic
            axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
            axis.set_xlabel(r'$m_l$ (MeV)', fontsize=14)
            
            # Custom Legend Logic
            handles, labels = axis.get_legend_handles_labels()
            data_handles = [h for h, l in zip(handles, labels) if l == 'Data']
            
            ordered_handles = [data_handles[0]] if data_handles else []
            ordered_labels = ['Data'] if data_handles else []
            
            for b in legend_order:
                if b in handles_dict:
                    ordered_handles.append(handles_dict[b])
                    ordered_labels.append(labels_dict[b])
            
            axis.legend(ordered_handles, ordered_labels, fontsize=10, loc='best')

        
        # Save Individual Plot
        ax.set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
        fig.tight_layout()
        fig.savefig(f"Plot_Log_{group}_{dtype}_{A_types[A_index]}_v2.svg", bbox_inches='tight')
        plt.close(fig)

        
    # Final aesthetics for grouped figure
    axes_grp[0].set_ylabel(r'$\sigma$ (MeV$^{2}$)', fontsize=14)
    fig_grp.tight_layout()
    fig_grp.savefig(f"Plot_Log_Grouped_{dtype}_{A_types[A_index]}_v2.svg", bbox_inches='tight')
    plt.close(fig_grp)

print("Visual style matched. PNGs generated.")

