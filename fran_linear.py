# This script fits Bulava data with the trenascription of a linear fit in m_l units instead of am_l.
# Hence in the end its a quadratic fit but in terms of m_l is linear.
# The parameters obtained y0, A and B are to be compared with the ones obtained from Brambilla data fits.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Physical Constants & Input Setup ---
a_fm = 0.0633          # Lattice spacing parameter
hbar_c = 197.3269804 
a = a_fm / hbar_c  # Lattice spacing in MeV^-1
B0 = 2700          # ChPT constant B_0 MeV
f_pi = 92       # Pion decay constant MeV
L3 = 600      # Lambda_3 \sim 0.6 GeV
ck = 0.4     # Scalar parameter ck, can vary between 0.4, 0.5, 0.6
m_k_bar = 485.0   # Reference Kaon mass \bar{m}_K MeV

data_file = '/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/sbdata_Francesco/best_sigma_fit.csv'

ensembles = ["D200", "N200", "N203"]
m_pi_ens = {'D200': 200, 'N200': 280, 'N203': 340}
m_k_ens = {'D200': 480, 'N200': 460, 'N203': 440}

def mu_l(m_pi, m_k):
    return 3 * m_pi**2 / (m_pi**2 + 2 * m_k**2)

# --- Load Data ---
df = pd.read_csv(data_file, index_col=0)
x_data, y_data, y_err = [], [], []

for ens in ensembles:
    if ens in df.index:
        x_data.append(mu_l(m_pi_ens[ens], m_k_ens[ens]))
        y_data.append(float(df.loc[ens, 'Sigma_a2']))
        y_err.append(float(df.loc[ens, 'Error_Sigma_a2']))

x_data = np.array(x_data)
y_data = np.array(y_data)
y_err = np.array(y_err)

# --- Define Model Function ---

def func_C(A):
    return - A * m_k_bar**2 / (48 * np.pi**2 * f_pi**2)

def func_B(A,C):
    return (A/3) * (1 + ck) + func_C(A) * np.log(2 * m_k_bar**2  / (3 * L3**2))

def linear_model(mu, y0, A):
    C = func_C(A)
    B = func_B(A, C)
    return y0 + A * mu + B * (mu**2) + C * (mu**2) * np.log(mu)

# --- Perform Fit ---
popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True, p0=[1.0, 0.1])
y0_fit, A_fit = popt
perr = np.sqrt(np.diag(pcov))

# --- Calculate Chi^2 ---
res = y_data - linear_model(x_data, *popt)
chi2 = np.sum((res / y_err)**2)
dof = len(x_data) - len(popt)
print(f"Fit Results:\n  y0 = {y0_fit:.6f} +/- {perr[0]:.6f}\n  A = {A_fit:.6f} +/- {perr[1]:.6f}\n B = {func_B(A_fit, func_C(A_fit)):.6f}\n C = {func_C(A_fit):.6f}")
print(f"  chi2/dof = {chi2:.4f} / {dof}")

# --- Plotting ---
x_fit = np.linspace(min(x_data) * 0.9, max(x_data) * 1.1, 200)
y_fit = linear_model(x_fit, *popt)

# Parameter uncertainty band via sampling
samples = np.random.multivariate_normal(popt, pcov, 1000)
y_samples = np.array([linear_model(x_fit, *p) for p in samples])
y_band = np.std(y_samples, axis=0)

plt.figure(figsize=(7, 5))
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='black', ecolor='black', capsize=4, label='Bulava Data')
plt.plot(x_fit, y_fit, 'r-', label=f'Linear Fit ($\chi^2/dof = {chi2:.2f}/{dof}$)')
plt.fill_between(x_fit, y_fit - y_band, y_fit + y_band, color='red', alpha=0.25, label='$1\sigma$ uncertainty')

plt.xlabel(r'$\mu_l$')
plt.ylabel(r'$\sigma a^2$')
plt.title('Bulava Data: Linear Model Fit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()