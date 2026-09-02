# plot_results_ensemble.py
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plotting aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = [
    'STIXGeneral',
    'Times New Roman',
    'Times',
    'Nimbus Roman',
]
mpl.rcParams['mathtext.fontset'] = 'stix'

# Physical Constants for logarithmic evaluation
hbar_c = 197.3269804
r0_fm = 0.4547
r0_MeV_central = r0_fm / hbar_c
B0_dim_central = 2700 * r0_MeV_central
f_pi_dim_central = 92 * r0_MeV_central
mu_dim_central = 200 * r0_MeV_central

A_types = ['Ar0', 'Api12']
A_latex = ['r_0', '\\pi/12']
fit_types = ['linear', 'logarithmic']

# Selection Settings
A_index = 1  # 0 for Ar0, 1 for Api12
fit_index = 1  # 0 for linear, 1 for logarithmic

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_full_{fit_types[fit_index]}_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(lambda s: '_'.join(s.split('_')[-2:]))
fit_df = pd.read_csv(filename)


def calculate_dimensionless(df_row, prefix):
  r0_a = df_row[f'r0_a_{prefix}']
  r0_a_err = df_row[f'r0_a_{prefix}_err']
  a2sigma = df_row[f'a2sigma_{prefix}']
  a2sigma_err = df_row[f'a2sigma_{prefix}_err']
  am_l = df_row['am_l']

  y = a2sigma * (r0_a**2)
  y_err = y * np.sqrt((a2sigma_err / a2sigma) ** 2 + (2 * r0_a_err / r0_a) ** 2)
  x = am_l * r0_a
  x_err = x * (r0_a_err / r0_a)
  return x, x_err, y, y_err


def eval_model(x_vals, fit_row):
  y0 = fit_row['y0_central']
  L_prime = fit_row['L_prime_central']

  if fit_index == 0:  # Linear
    return y0 - 4 * L_prime * x_vals

  # Logarithmic
  L2 = fit_row['L2_central']
  c_pipi = fit_row['c_pipi_central']

  term1 = 4 * L_prime * x_vals
  log_arg = np.maximum(
      (2 * B0_dim_central * x_vals) / (mu_dim_central**2), 1e-10
  )
  term2_coeff = (3 * B0_dim_central) / (4 * np.pi**2 * f_pi_dim_central**2)
  term2 = (
      term2_coeff
      * (4 * B0_dim_central * c_pipi - L_prime)
      * (np.log(log_arg) - 1)
      * (x_vals**2)
  )
  term3 = 2 * L2 * (x_vals**2)
  return y0 - term1 - term2 - term3


def fmt_sci(val):
  if np.isinf(val) or np.isnan(val):
    return 'N/A'
  if val == 0:
    return '0'
  base, exp = '{:.2e}'.format(val).split('e')
  exp_int = int(exp)
  if -2 <= exp_int <= 1:
    return f'{val:.3g}'
  return f'{float(base):.2f} \\times 10^{{{exp_int}}}'


data_types = ['bare', 'smeared']
data_color = '#1f1f1f'
fit_color = '#0055ff'

for prefix in data_types:
  ensembles = df['mass_group'].unique()
  fig_grp, axes_grp = plt.subplots(
      1, len(ensembles), figsize=(18, 6), sharey=True
  )
  fig_grp.suptitle(
      f'Dimensionless {fit_types[fit_index].capitalize()} Fits -'
      f' {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)',
      fontsize=18,
      fontweight='bold',
      y=1.02,
  )

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
      axis.errorbar(
          x_arr,
          y_arr,
          xerr=x_err_arr,
          yerr=y_err_arr,
          fmt='o',
          color=data_color,
          alpha=0.9,
          label='Data',
          capsize=4,
          zorder=5,
      )
      axis.set_xlim(x_min - x_pad, x_max + x_pad)

    fit_rows = fit_df[
        (fit_df['Ensemble'] == group) & (fit_df['Data_Type'] == prefix)
    ]

    for _, fit_row in fit_rows.iterrows():
      y0_cen = fit_row['y0_central']
      y0_err = fit_row['y0_total_err']
      L_prime_cen = fit_row['L_prime_central']

      y_cen = eval_model(x_vals_plot, fit_row)
      y_total_err = np.ones_like(x_vals_plot) * y0_err

      if fit_index == 0:  # Linear
        eq_label = (
            rf'Fit ($y_0 = {fmt_sci(y0_cen)}$, $L^\prime ='
            rf' {fmt_sci(L_prime_cen)}$)'
        )
      else:  # Logarithmic
        L2_cen = fit_row['L2_central']
        c_pipi_cen = fit_row['c_pipi_central']
        eq_label = (
            rf'Fit ($y_0 = {fmt_sci(y0_cen)}$, $L^\prime ='
            rf' {fmt_sci(L_prime_cen)}$,\n$L_2 = {fmt_sci(L2_cen)}$,'
            rf' c_{{\pi\pi}} = {fmt_sci(c_pipi_cen)}$)'
        )

      for axis in [ax, ax_grp]:
        axis.plot(
            x_vals_plot, y_cen, linestyle='-', color=fit_color, label=eq_label
        )
        if not np.all(np.isnan(y_total_err)):
          axis.fill_between(
              x_vals_plot,
              y_cen - y_total_err,
              y_cen + y_total_err,
              color=fit_color,
              alpha=0.25,
          )

    for axis in [ax, ax_grp]:
      axis.tick_params(
          direction='in', top=True, right=True, bottom=True, left=True, length=6
      )
      axis.ticklabel_format(
          style='sci', axis='y', scilimits=(0, 0), useMathText=True
      )
      axis.set_xlabel(r'$x = m_l r_0$', fontsize=14)
      axis.legend(fontsize=9, loc='best')

    ax.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
    fig.tight_layout()
    fig.savefig(
        f'Plot_{fit_types[fit_index].capitalize()}_{group}_{prefix}_{A_types[A_index]}.svg',
        bbox_inches='tight',
    )
    plt.close(fig)

  axes_grp[0].set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
  fig_grp.tight_layout()
  fig_grp.savefig(
      f'Plot_{fit_types[fit_index].capitalize()}_Grouped_{prefix}_{A_types[A_index]}.svg',
      bbox_inches='tight',
  )
  plt.close(fig_grp)