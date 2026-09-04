# Just plots! Bulava data with the transcription of a logarithmic fit in m_l units instead of am_l.
# In here we use all the values of the parameters obtained with our logarithmic global fit of Brambilla data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Set Index & Files ---
A_types = ['Ar0', 'Api12']
A_index = 0  # Set manually (0 or 1)

a_fm = 0.0633          # Lattice spacing parameter
hbar_c = 197.3269804 
a = a_fm / hbar_c  # Lattice spacing in MeV^-1
B0 = 2700 #MeV 
f_pi = 92 #MeV
l3_bar = 1.8
sigma0 = -0.3
m_k_bar = 485.0 #MeV
mu_scale = 200  # Renormalization scale \mu MeV

data_file = '/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/sbdata_Francesco/best_sigma_fit.csv'
log_fit_file = f'fit_results_log_global_{A_types[A_index]}.csv'

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

# --- Load Log Fit Results ---
df_log = pd.read_csv(log_fit_file)

def log_model(mu, y0, L_prime, L2, c_pipi):
    A = -4 * L_prime * a * (m_k_bar**2) / (3 * B0)
    B = -4 * L_prime * a * (m_k_bar**2) / (9 * B0) * ((m_k_bar**2 * l3_bar) / (16 * np.pi**2 * f_pi**2) + 1.0 - 4.0 * sigma0) \
        - 2 * L2 * (a**2) * (m_k_bar**4) / (9 * B0**2)
    C = - (a**2 * m_k_bar**4) / (12 * np.pi**2 * f_pi**2 * B0) * (4 * B0 * c_pipi - L_prime)
    
    log_term = np.log((2 * a * m_k_bar**2 * mu) / (3 * mu_scale**2)) - 1.0
    return y0 + A * mu + B * (mu**2) + C * (mu**2) * log_term

# --- Plotting ---
plt.figure(figsize=(8, 6))
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='black', capsize=4, label='Bulava Data', zorder=5)

x_fit = np.linspace(min(x_data) * 0.9, max(x_data) * 1.1, 200)

for idx, row in df_log.iterrows():
    branch = row.get('Branch', f'Branch {idx}')
    dtype = row.get('Data_Type', '')
    
    y0_cen = float(row['y0_central'])
    L_prime_cen = float(row['L_prime_central'])
    L2_cen = float(row['L2_central'])
    c_pipi_cen = float(row['c_pipi_central'])
    
    y_fit = log_model(x_fit, y0_cen, L_prime_cen, L2_cen, c_pipi_cen)
    
    # Calculate chi2 against Bulava data
    y_pred = log_model(x_data, y0_cen, L_prime_cen, L2_cen, c_pipi_cen)
    chi2 = np.sum(((y_data - y_pred) / y_err)**2)
    
    label_str = f"{dtype} - {branch} ($\chi^2={chi2:.2f}$)"
    plt.plot(x_fit, y_fit, label=label_str)

plt.xlabel(r'$\mu_l$')
plt.ylabel(r'$\sigma a^2$')
plt.title(f'Logarithmic Model Predictions vs Bulava Data ({A_types[A_index]})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()