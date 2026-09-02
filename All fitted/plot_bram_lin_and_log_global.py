# plot_results_global.py
import json
import matplotlib as mpl
import matplotlib.gridspec as gridspec
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
fit_index = 1  # 0 for linear global, 1 for logarithmic global

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/data_bram_{A_types[A_index]}.csv'
filename = f'fit_results_full_global_{fit_types[fit_index]}_{A_types[A_index]}.csv'

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
groups = df['mass_group'].unique()
data_color = '#1f1f1f'
fit_color = '#0055ff'

for prefix in data_types:
  fig = plt.figure(figsize=(18, 12))
  gs = gridspec.GridSpec(
      2, len(groups), height_ratios=[1.5, 1], hspace=0.3, wspace=0.2
  )
  ax_main = fig.add_subplot(gs[0, :])

  zoom_axes = []
  for i in range(len(groups)):
    if i == 0:
      ax = fig.add_subplot(gs[1, i])
    else:
      ax = fig.add_subplot(gs[1, i], sharey=zoom_axes[0])
    zoom_axes.append(ax)

  fig.suptitle(
      f'Global Dimensionless {fit_types[fit_index].capitalize()} Fit -'
      f' {prefix.capitalize()} Data ($A=A_{{{A_latex[A_index]}}}$)',
      fontsize=20,
      fontweight='bold',
      y=0.95,
  )

  phys_data_all = df.apply(
      lambda row: calculate_dimensionless(row, prefix), axis=1
  )
  df['x_calc'], df['x_err'] = [d[0] for d in phys_data_all], [
      d[1] for d in phys_data_all
  ]
  df['y_calc'], df['y_err'] = [d[2] for d in phys_data_all], [
      d[3] for d in phys_data_all
  ]

  dtype_fits = fit_df[fit_df['Data_Type'] == prefix]

  ax_main.set_title('Global Overview', fontsize=16)
  x_min_main, x_max_main = df['x_calc'].min(), df['x_calc'].max()
  x_pad_main = (x_max_main - x_min_main) * 0.05
  x_vals_main = np.linspace(
      x_min_main - x_pad_main, x_max_main + x_pad_main, 500
  )
  ax_main.set_xlim(x_min_main - x_pad_main, x_max_main + x_pad_main)

  ax_main.errorbar(
      df['x_calc'],
      df['y_calc'],
      xerr=df['x_err'],
      yerr=df['y_err'],
      fmt='o',
      color=data_color,
      alpha=0.9,
      capsize=4,
      label='Data',
      zorder=5,
  )

  if not dtype_fits.empty:
    for _, fit_row in dtype_fits.iterrows():
      y0_cen = fit_row['y0_central']
      y0_err = fit_row['y0_total_err']
      L_prime_cen = fit_row['L_prime_central']

      y_cen = eval_model(x_vals_main, fit_row)
      y_total_err = np.ones_like(x_vals_main) * y0_err

      if fit_index == 0:  # Linear
        eq_label = (
            rf'Global Fit ($y_0 = {fmt_sci(y0_cen)}$, $L^\prime ='
            rf' {fmt_sci(L_prime_cen)}$)'
        )
      else:  # Logarithmic
        L2_cen = fit_row['L2_central']
        c_pipi_cen = fit_row['c_pipi_central']
        eq_label = (
            rf'Global Fit ($y_0 = {fmt_sci(y0_cen)}$, $L^\prime ='
            rf' {fmt_sci(L_prime_cen)}$, $L_2 = {fmt_sci(L2_cen)}$,'
            rf' $c_{{\pi\pi}} = {fmt_sci(c_pipi_cen)}$)'
        )

      ax_main.plot(
          x_vals_main, y_cen, linestyle='-', color=fit_color, label=eq_label
      )
      if not np.all(np.isnan(y_total_err)):
        ax_main.fill_between(
            x_vals_main,
            y_cen - y_total_err,
            y_cen + y_total_err,
            color=fit_color,
            alpha=0.25,
        )

  ax_main.tick_params(
      direction='in', top=True, right=True, bottom=True, left=True, length=6
  )
  ax_main.ticklabel_format(
      style='sci', axis='y', scilimits=(0, 0), useMathText=True
  )
  ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)
  ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=14)
  ax_main.legend(fontsize=10, loc='best')

  for idx, group in enumerate(groups):
    ax_z = zoom_axes[idx]
    subset = df[df['mass_group'] == group]
    if subset.empty:
      continue

    ax_z.set_title(f'Zoom: {group}', fontsize=14)
    x_min_z, x_max_z = subset['x_calc'].min(), subset['x_calc'].max()
    x_pad_z = (
        (x_max_z - x_min_z) * 0.05 if x_max_z > x_min_z else x_min_z * 0.05
    )
    x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 100)
    ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)

    ax_z.errorbar(
        subset['x_calc'],
        subset['y_calc'],
        xerr=subset['x_err'],
        yerr=subset['y_err'],
        fmt='o',
        color=data_color,
        alpha=0.9,
        capsize=4,
        zorder=5,
    )

    if not dtype_fits.empty:
      for _, fit_row in dtype_fits.iterrows():
        y0_cen = fit_row['y0_central']
        y0_err = fit_row['y0_total_err']

        y_cen_z = eval_model(x_vals_z, fit_row)
        y_total_err_z = np.ones_like(x_vals_z) * y0_err

        ax_z.plot(x_vals_z, y_cen_z, linestyle='-', color=fit_color)
        if not np.all(np.isnan(y_total_err_z)):
          ax_z.fill_between(
              x_vals_z,
              y_cen_z - y_total_err_z,
              y_cen_z + y_total_err_z,
              color=fit_color,
              alpha=0.25,
          )

    ax_z.tick_params(
        direction='in', top=True, right=True, bottom=True, left=True, length=6
    )
    ax_z.ticklabel_format(
        style='sci', axis='y', scilimits=(0, 0), useMathText=True
    )
    ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=14)
    if idx == 0:
      ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=14)

  fig.savefig(
      f'Plot_{fit_types[fit_index].capitalize()}_Global_{prefix}_{A_types[A_index]}.svg',
      bbox_inches='tight',
  )
  plt.close(fig)