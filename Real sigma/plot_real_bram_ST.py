# Follows the concept of fit_real_bram_ST.py
# Uses pre-computed y0, c_l, c2_l, gamma traces from CSV to reconstruct physical error bands

import ast  # To parse matrix strings from CSV if needed
import os
import warnings
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Aesthetics
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = [
    'STIXGeneral',
    'Times New Roman',
    'Times',
    'Nimbus Roman',
]
mpl.rcParams['mathtext.fontset'] = 'stix'

# =============================================================================
# 1. Constants & Scale Conversions (Matching Fitting Script)
# =============================================================================
hbar_c = 0.1973269804  # GeV*fm
r0_fm = 0.4547
r0_GeV_central = r0_fm / hbar_c

# Physical point constraints used to compute y0 theoretically
m2_lq_phys = 0.051  # GeV^2
sigma_phys = 0.21  # GeV^2

A_types = ['Ar0', 'Api12']
A_latex = ['r_0', '\\pi/12']
A_index = 0
LQCD = False

# File Paths
data_file = f'Data/data_bram_{A_types[A_index]}.csv'
lqcd_str = '_full' if LQCD else ''
fit_file = f'Real sigma/fit_results_theo{lqcd_str}_ST_{A_types[A_index]}.csv'

df = pd.read_csv(data_file, sep=r'\s+')
df['mass_group'] = df['Ensemble'].apply(
    lambda s: '_'.join(s.split('_')[-2:])
)
fit_df = pd.read_csv(fit_file)


# =============================================================================
# 2. Helper Functions & Dynamic Model Evaluation
# =============================================================================
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


# Dynamic y0 constraints
def compute_y0(c_l, gamma, r0_GeV):
  x2_phys = (m2_lq_phys / c_l**2) * (r0_GeV**2)
  y_phys = sigma_phys * (r0_GeV**2)
  log_arg = np.maximum(x2_phys, 1e-15)
  term1 = (c_l**2 / (2.0 * np.pi)) * x2_phys * np.log(log_arg)
  term2 = gamma * x2_phys
  return y_phys - term1 - term2


def compute_y02(c_l, c2_l, gamma, r0_GeV):
  m_lq = np.sqrt(m2_lq_phys)
  x_phys = (m_lq / c_l) * r0_GeV
  z_phys = x_phys + (c2_l / c_l)
  z2_phys = z_phys**2
  y_phys = sigma_phys * (r0_GeV**2)
  log_arg = np.maximum(z2_phys, 1e-15)
  term1 = (c_l**2 / (2.0 * np.pi)) * z2_phys * np.log(log_arg)
  term2 = gamma * z2_phys
  return y_phys - term1 - term2


# Model functions
def model_func_dim(x, y_0, c_l, gamma):
  term1 = (c_l**2 / (2.0 * np.pi)) * x**2 * np.log(np.maximum(x**2, 1e-15))
  term2 = gamma * x**2
  return y_0 + term1 + term2


def model_func2_dim(x, y_0, c_l, c2_l, gamma):
  z = x + c2_l / c_l
  log_arg = np.maximum(z**2, 1e-15)
  term1 = (c_l**2 / (2.0 * np.pi)) * z**2 * np.log(log_arg)
  term2 = gamma * z**2
  return y_0 + term1 + term2


def eval_model(x, popt):
  """Evaluates the model with y0 derived dynamically from fit parameters."""
  if not LQCD:
    c_l, gamma = popt
    y0 = compute_y0(c_l, gamma, r0_GeV_central)
    return model_func_dim(x, y0, c_l, gamma)
  else:
    c_l, c2_l, gamma = popt
    y0 = compute_y02(c_l, c2_l, gamma, r0_GeV_central)
    return model_func2_dim(x, y0, c_l, c2_l, gamma)


def compute_error_band(x_vals, popt, pcov):
  """Computes statistical error band via Jacobian matrix J using central finite differences."""
  n_params = len(popt)
  J = np.zeros((n_params, len(x_vals)))
  eps = 1e-6

  for i in range(n_params):
    p_plus = np.array(popt, dtype=float)
    p_minus = np.array(popt, dtype=float)
    h = max(abs(popt[i]) * eps, 1e-8)
    p_plus[i] += h
    p_minus[i] -= h
    J[i] = (eval_model(x_vals, p_plus) - eval_model(x_vals, p_minus)) / (2.0 * h)

  variance_y = np.einsum('ik,ij,jk->k', J, pcov, J)
  return np.sqrt(np.maximum(variance_y, 0))


# =============================================================================
# 3. Main Plotting Routine
# =============================================================================
data_types = ['bare', 'smeared']
groups = sorted(df['mass_group'].unique())

colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
group_color_map = dict(zip(groups, colors))

for prefix in data_types:
  fig = plt.figure(figsize=(18, 10))
  gs = gridspec.GridSpec(
      2, len(groups), height_ratios=[1.6, 1], hspace=0.35, wspace=0.25
  )

  ax_main = fig.add_subplot(gs[0, :])
  zoom_axes = [fig.add_subplot(gs[1, i]) for i in range(len(groups))]

  lqcd_title = 'Full ' if LQCD else ''
  fig.suptitle(
      f'Theoretical {lqcd_title}Soto-Tarrús Fit - {prefix.capitalize()} Data'
      f' ($A=A_{{{A_latex[A_index]}}}$)',
      fontsize=18,
      fontweight='bold',
      y=0.96,
  )

  # Calculate x, y and errors
  phys_data = df.apply(lambda row: calculate_dimensionless(row, prefix), axis=1)
  df['x_calc'], df['x_err'] = zip(*[(d[0], d[1]) for d in phys_data])
  df['y_calc'], df['y_err'] = zip(*[(d[2], d[3]) for d in phys_data])

  dtype_fits = (
      fit_df[fit_df['Data_Type'] == prefix]
      if 'Data_Type' in fit_df.columns
      else fit_df
  )

  # Plot Data on Main Axis
  for group in groups:
    grp_data = df[df['mass_group'] == group]
    ax_main.errorbar(
        grp_data['x_calc'],
        grp_data['y_calc'],
        xerr=grp_data['x_err'],
        yerr=grp_data['y_err'],
        fmt='o',
        color=group_color_map[group],
        alpha=0.85,
        capsize=3,
        label=f'Ensamble: {group}',
        zorder=4,
    )

  # Global Fit Curve and Band Propagation from fit_df
  if not dtype_fits.empty:
    fit_row = dtype_fits.iloc[0]
    y0_cen = fit_row['y_0']
    cl_cen = fit_row['c_l']
    gamma_cen = fit_row['gamma']

    x_min_main, x_max_main = min(df['x_calc']), max(df['x_calc'])
    x_pad = (x_max_main - x_min_main) * 0.05
    x_vals_main = np.linspace(x_min_main - x_pad, x_max_main + x_pad, 500)

    if LQCD:
      c2l_cen = fit_row['c2_l']
      popt = [cl_cen, c2l_cen, gamma_cen]
      y_cen = model_func2_dim(x_vals_main, y0_cen, cl_cen, c2l_cen, gamma_cen)
      eq_label = (
          rf'$y_0 = {y0_cen:.3f}$, $c_1 = {cl_cen:.2f}$, $c_2 ='
          rf' {c2l_cen:.2f}$, $\gamma = {gamma_cen:.2f}$'
      )
    else:
      popt = [cl_cen, gamma_cen]
      y_cen = model_func_dim(x_vals_main, y0_cen, cl_cen, gamma_cen)
      eq_label = (
          rf'$y_0 = {y0_cen:.3f}$, $c_l = {cl_cen:.2f}$, $\gamma ='
          rf' {gamma_cen:.2f}$'
      )

    ax_main.plot(
        x_vals_main,
        y_cen,
        linestyle='-',
        color='#003366',
        lw=2,
        label=eq_label,
        zorder=5,
    )

    # Plot Error Band using pre-computed pcov from CSV
    if 'pcov' in fit_row and pd.notna(fit_row['pcov']):
      pcov = np.array(ast.literal_eval(str(fit_row['pcov'])))
      y_total_err = compute_error_band(x_vals_main, popt, pcov)
      ax_main.fill_between(
          x_vals_main,
          y_cen - y_total_err,
          y_cen + y_total_err,
          color='#003366',
          alpha=0.2,
          zorder=3,
      )

    ax_main.set_xlim(x_min_main - x_pad, x_max_main + x_pad)

  ax_main.tick_params(
      direction='in', top=True, right=True, bottom=True, left=True, length=6
  )
  ax_main.ticklabel_format(
      style='sci', axis='y', scilimits=(0, 0), useMathText=True
  )
  ax_main.set_ylabel(r'$y = \sigma r_0^2$', fontsize=13)
  ax_main.set_xlabel(r'$x = m_l r_0$', fontsize=13)
  ax_main.legend(fontsize=11, loc='best', framealpha=0.9)

  # Subplots per Group
  for idx, group in enumerate(groups):
    ax_z = zoom_axes[idx]
    subset = df[df['mass_group'] == group]
    if subset.empty:
      continue

    ax_z.set_title(f'Zoom: {group}', fontsize=12)
    x_min_z, x_max_z = min(subset['x_calc']), max(subset['x_calc'])
    x_pad_z = (
        (x_max_z - x_min_z) * 0.1 if x_max_z > x_min_z else x_min_z * 0.05
    )
    x_vals_z = np.linspace(x_min_z - x_pad_z, x_max_z + x_pad_z, 150)

    ax_z.errorbar(
        subset['x_calc'],
        subset['y_calc'],
        xerr=subset['x_err'],
        yerr=subset['y_err'],
        fmt='o',
        color=group_color_map[group],
        alpha=0.9,
        capsize=3,
        zorder=4,
    )

    if not dtype_fits.empty:
      if LQCD:
        y_cen_z = model_func2_dim(x_vals_z, y0_cen, cl_cen, c2l_cen, gamma_cen)
      else:
        y_cen_z = model_func_dim(x_vals_z, y0_cen, cl_cen, gamma_cen)

      ax_z.plot(
          x_vals_z, y_cen_z, linestyle='-', color='#003366', lw=2, zorder=5
      )

      if 'pcov' in fit_row and pd.notna(fit_row['pcov']):
        pcov = np.array(ast.literal_eval(str(fit_row['pcov'])))
        y_total_err_z = compute_error_band(x_vals_z, popt, pcov)
        ax_z.fill_between(
            x_vals_z,
            y_cen_z - y_total_err_z,
            y_cen_z + y_total_err_z,
            color='#003366',
            alpha=0.2,
            zorder=3,
        )

    ax_z.set_xlim(x_min_z - x_pad_z, x_max_z + x_pad_z)
    ax_z.tick_params(
        direction='in', top=True, right=True, bottom=True, left=True, length=5
    )
    ax_z.ticklabel_format(
        style='sci', axis='y', scilimits=(0, 0), useMathText=True
    )
    ax_z.set_xlabel(r'$x = m_l r_0$', fontsize=12)
    if idx == 0:
      ax_z.set_ylabel(r'$y = \sigma r_0^2$', fontsize=12)

  output_name = f'Plot_ST{lqcd_str}_Theoretical_{prefix}_{A_types[A_index]}.svg'
  fig.savefig(output_name, bbox_inches='tight')
  plt.close(fig)

print('Plotting complete!')