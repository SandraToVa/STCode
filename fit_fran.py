import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2



def mu_l(m_pi, m_k):
    """Definition of the quark mass parameter in 2403.00754"""
    return 3 * m_pi**2 / (m_pi**2 + 2 * m_k**2)


def linear_model(mu, y0, Lambda):
    """Linear fit model: y = y0 + Lambda * mu_l"""
    return y0 + Lambda * mu

# =============================================================================
# 1. Constants & Parameter Sets
# =============================================================================
# Obtain the data from the CSV file
ensembles = ["D200", "N200", "N203"]
m_pi_ens = {'D200': 200, 'N200': 280, 'N203': 340}
m_k_ens = {'D200': 480, 'N200': 460, 'N203': 440}

data_file = f'/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/STCode/Data/sbdata_Francesco/best_sigma_fit.csv'

try:

    df = pd.read_csv(data_file).set_index('Ensemble')

    x_data = []
    y_data = []
    y_err = []

    for ens in ensembles:
        if ens in df.index:
            m_pi = m_pi_ens[ens]
            m_k = m_k_ens[ens]

            x_data.append(mu_l(m_pi, m_k))
            y_data.append(float(df.loc[ens, 'Sigma_a2']))
            y_err.append(float(df.loc[ens, 'Error_Sigma_a2']))

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_err = np.array(y_err)

    # =============================================================================
    # 2. Data Fitting
    # =============================================================================

    # Perform Weighted Least Squares (WLS) fit
    popt, pcov = curve_fit(
        linear_model,
        x_data,
        y_data,
        sigma=y_err,
        absolute_sigma=True,
    )

    y0, Lambda = popt
    y0_err, Lambda_err = np.sqrt(np.diag(pcov))

    # Calculate Chi-squared and Goodness of Fit
    residuals = y_data - linear_model(x_data, y0, Lambda)
    chi2_stat = np.sum((residuals / y_err) ** 2)
    dof = len(x_data) - len(popt)  # 3 data points - 2 fit parameters = 1
    chi2_red = chi2_stat / dof
    p_value = chi2.sf(chi2_stat, dof)

    # =============================================================================
    # 3. Save Data
    # =============================================================================

    fit_summary = pd.DataFrame([{
        'linear_fit_y0': y0,
        'linear_fit_y0_err': y0_err,
        'linear_fit_slope': Lambda,
        'linear_fit_slope_err': Lambda_err,
        'cov_y0_y0': pcov[0, 0],
        'cov_y0_slope': pcov[0, 1],
        'cov_slope_y0': pcov[1, 0],
        'cov_slope_slope': pcov[1, 1],
        'chi2': chi2_stat,
        'dof': dof,
        'chi2_dof': chi2_red,
        'p_value': p_value,
    }])

    fit_summary.to_csv('fran_results_linear.csv', index=False)

    # Display Results
    print("================ LINEAR FIT RESULTS ================")
    print(f"y0 (Chiral Limit)  = {y0:.6f} ± {y0_err:.6f}")
    print(f"Lambda (Slope)     = {Lambda:.6f} ± {Lambda_err:.6f}")
    print(f"Covariance(y0, L)  = {pcov[0, 1]:.6e}")
    print("----------------------------------------------------")
    print(f"chi^2              = {chi2_stat:.4f}")
    print(f"Degrees of Freedom = {dof}")
    print(f"chi^2 / dof        = {chi2_red:.4f}")
    print(f"p-value            = {p_value:.4f}")
    print("====================================================")
    print("\nResults saved to 'fran_results_linear.csv'")

except FileNotFoundError:
    print("Error: 'best_sigma_fit.csv' file not found.")