# Evaluates the linear regime of the V_2 potential after the string breaking. 
# Fits this linear function to finc the corresponding string tension (sigma) for each ensemble (D200, N200, N203) creating file "result_sigma_fit.csv" with the results of the fits for different r_min values.
# Obtain the best fit for each ensemble by selecting the one with Chi2/dof closest to 1 creating file "best_sigma_fit.csv" with the best fits for each ensemble.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Base directory and data paths for each ensemble
base_dir = "/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/Code/Data/sbdata_Francesco"
data_paths = {
    "D200": os.path.join(base_dir, "D200"),
    "N200": os.path.join(base_dir, "N200"),
    "N203": os.path.join(base_dir, "N203"),
}

# Evaluate a range of r_min values for each ensemble
ensembles = ["D200", "N200", "N203"]
r_min_values = list(range(18, 24))  # 18, 19, 20, 21, 22, 23

# Build the list of files automatically from the naming convention potentials{ens}.dat
files = [
    {
        "ensemble": ens,
        "nom": os.path.join(data_paths[ens], f"potentials{ens}.dat"),
        "r_min": r_min,
    }
    for ens in ensembles
    for r_min in r_min_values
]


def func_linear(x, m, c):
    return m * x + c


results = []

for file in files:
    ensemble = file["ensemble"]
    nom_arxiu = file["nom"]
    r_min = file["r_min"]

    try:
        # Llegim el fitxer. La primera línia comença amb '%' i és un comentari del fitxer
        # Cols: 0:(r/a)^2, 1:V0, 2:errV0, 3:V1, 4:errV1, 5:V2, 6:errV2
        dades = np.loadtxt(nom_arxiu, comments='%')

        # Calculem r/a traient l'arrel quadrada de la primera columna
        r_a = np.sqrt(dades[:, 0])
        v2 = dades[:, 5]
        err_v2 = dades[:, 6]

        # Apliquem la màscara per agafar només les distàncies llargues
        mascara = r_a >= r_min
        x_ajust = r_a[mascara]
        y_ajust = v2[mascara]
        e_ajust = err_v2[mascara]

        if len(x_ajust) < 3:
            print(f"Alerta: No hi ha prou punts a {nom_arxiu} per r_min >= {r_min}")
            continue

        # Ajustem la corba ponderant amb l'error estadístic (sigma)
        popt, pcov = curve_fit(func_linear, x_ajust, y_ajust, sigma=e_ajust, absolute_sigma=True)

        pendent, constant = popt
        error_pendent, error_constant = np.sqrt(np.diag(pcov))

        # Calculem el Chi-quadrat reduït per a validar l'ajust
        chi2 = np.sum(((y_ajust - func_linear(x_ajust, pendent, constant)) / e_ajust) ** 2)
        dof = len(x_ajust) - 2
        chi2_reduit = chi2 / dof

        results.append({
            "Ensemble": ensemble,
            "r_min": r_min,
            "Sigma_a2": pendent,
            "Error_Sigma_a2": error_pendent,
            "Constant": constant,
            "Error_Constant": error_constant,
            "Chi2/dof": chi2_reduit,
            "Num_Punts": len(x_ajust)
        })

    except FileNotFoundError:
        print(f"No s'ha trobat el fitxer {nom_arxiu}.")

# Convertim els resultats en un DataFrame i els guardem en un CSV
df_resultats = pd.DataFrame(results)
df_resultats = df_resultats.sort_values(["Ensemble", "r_min"]).reset_index(drop=True)
print(df_resultats.to_string(index=False))

df_resultats.to_csv("result_sigma_fit.csv", index=False)
print("\nResultats guardats correctament a 'result_sigma_fit.csv'")

# Select, for each ensemble, the fit whose Chi2/dof is closest to 1
best_rows = []
for ensemble in ensembles:
    df_ens = df_resultats[df_resultats["Ensemble"] == ensemble].copy()
    if df_ens.empty:
        continue
    df_ens["chi2_diff_from_1"] = (df_ens["Chi2/dof"] - 1.0).abs()
    best_row = df_ens.loc[df_ens["chi2_diff_from_1"].idxmin()].drop(columns=["chi2_diff_from_1"])
    best_rows.append(best_row)

best_df = pd.DataFrame(best_rows)
if not best_df.empty:
    best_df = best_df.sort_values("Ensemble").reset_index(drop=True)
    print("\nBest fit per ensemble:")
    print(best_df.to_string(index=False))
    best_df.to_csv("best_sigma_fit.csv", index=False)
    print("\nBest fits guardats correctament a 'best_sigma_fit.csv'")
else:
    print("\nNo hi ha resultats vàlids per seleccionar el millor fit.")

# Plot the fitted Sigma_a2 parameter as a function of r_min for each ensemble
plt.figure(figsize=(8, 5))
for ensemble in ensembles:
    df_ens = df_resultats[df_resultats["Ensemble"] == ensemble].sort_values("r_min")
    if df_ens.empty:
        continue
    plt.errorbar(
        df_ens["r_min"],
        df_ens["Sigma_a2"],
        yerr=df_ens["Error_Sigma_a2"],
        fmt='o',
        capsize=4,
        label=ensemble,
    )

plt.xlabel("r_min")
plt.ylabel("Sigma_a2")
plt.title("Dependencia de Sigma_a2 amb r_min")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("sigma_fit_rmin_scan.png", dpi=200)
plt.show()
print("\nGràfica guardada a 'sigma_fit_rmin_scan.png'")