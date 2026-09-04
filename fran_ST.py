# Just plots! Bulava data with the transcription of a ST fit in m_l units instead of am_l.
# In here we use all the values of the parameters obtained with our ST global fit of Brambilla data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Set Parameters ---
A_types = ['Ar0', 'Api12']
A_index = 0     # Set manually (0 or 1)
LQCD = False    # Set manually (True or False)

gamma_E = 0.5772156649
a_fm = 0.0633          # Lattice spacing parameter
hbar_c = 197.3269804 
a = a_fm / hbar_c  # Lattice spacing in MeV^-1
B0 = 2700 #MeV
f_pi = 92 #MeV
l3_bar = 1.8
sigma0 = -0.3
m_k_bar = 485.0 #MeV

lqcd_str = "_full" if LQCD else ""
st_fit_file = f'fit_results{lqcd_str}_ST_global_{A_types[A_index]}.csv'
data_file = '/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/sbdata_Francesco/best_sigma_fit.csv'

ensembles = ["D200", "N200", "N203"]
m_pi_ens = {'D200': 200, 'N200': 280, 'N203': 340}
m_k_ens = {'D200': 480, 'N200': 460, 'N203': 440}

def mu_l(m_pi, m_k):
    return 3 * m_pi**2 / (m_pi**2 + 2 * m_k**2)

# --- Load Bulava Data ---
df_bulava = pd.read_csv(data_file, index_col=0)
x_data, y_data, y_err = [], [], []

for ens in ensembles:
    if ens in df_bulava.index:
        x_data.append(mu_l(m_pi_ens[ens], m_k_ens[ens]))
        y_data.append(float(df_bulava.loc[ens, 'Sigma_a2']))
        y_err.append(float(df_bulava.loc[ens, 'Error_Sigma_a2']))

x_data = np.array(x_data)
y_data = np.array(y_data)
y_err = np.array(y_err)

# --- Load ST Fit Results ---
df_st = pd.read_csv(st_fit_file)

def st_model_false(mu_l_val, y0, c_l, mu_scale):
    log_arg = (c_l**2 * a**2 * m_k_bar**4) / (36.0 * np.pi * B0**2 * mu_scale**2)
    B = - (c_l**2 * a**2 * m_k_bar**4) / (18.0 * np.pi * B0**2) * (1.0 + gamma_E - np.log(log_arg))
    C = (c_l**2 * a**2 * m_k_bar**4) / (9.0 * np.pi * B0**2)
    return y0 + B * (mu_l_val**2) + C * (mu_l_val**2) * np.log(mu_l_val)

def st_model_true(mu_l_val, y0, c1, c2, mu_scale):
    log_arg = (c2**2) / (4.0 * np.pi * mu_scale**2)
    y0_tilde = y0 - (c2**2 / (2.0 * np.pi)) * (1.0 + gamma_E - np.log(log_arg))
    
    A = - (c1 * c2 * a * m_k_bar**2) / (3.0 * np.pi * B0) * (gamma_E - np.log(log_arg))
    
    term1 = - (c1**2 * a**2 * m_k_bar**4) / (18.0 * np.pi * B0**2) * (gamma_E - 1.0 - np.log(log_arg))
    term2 = - (c1 * c2 * a * m_k_bar**2) / (9.0 * np.pi * B0) * (gamma_E - np.log(log_arg)) * \
            ((m_k_bar**2 * l3_bar) / (16.0 * np.pi**2 * f_pi**2) + 1.0 - 4.0 * sigma0)
    B = term1 + term2
    
    return y0_tilde + A * mu_l_val + B * (mu_l_val**2)

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='black', capsize=4, label='Bulava Data', zorder=5)

x_fit = np.linspace(min(x_data) * 0.9, max(x_data) * 1.1, 200)

for idx, row in df_st.iterrows():
    branch = row.get('Branch', f'Branch {idx}')
    dtype = row.get('Data_Type', '')
    
    y0 = float(row['y_0'])
    c_l = float(row['c_l'])
    mu_scale = float(row['mu'])
    
    if not LQCD:
        y_fit = st_model_false(x_fit, y0, c_l, mu_scale)
        y_pred = st_model_false(x_data, y0, c_l, mu_scale)
    else:
        c2_l = float(row['c2_l'])
        y_fit = st_model_true(x_fit, y0, c_l, c2_l, mu_scale)
        y_pred = st_model_true(x_data, y0, c_l, c2_l, mu_scale)
        
    chi2 = np.sum(((y_data - y_pred) / y_err)**2)
    
    plt.plot(x_fit, y_fit, label=f"{dtype} - {branch} ($\chi^2={chi2:.2f}$)")

plt.xlabel(r'$\mu_l$')
plt.ylabel(r'$\sigma a^2$')
title_st = "LQCD-dependent ST Model" if LQCD else "Standard ST Model"
plt.title(f'{title_st} Predictions vs Bulava Data ({A_types[A_index]})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()