
# Read every fit-results CSV file and print chi^2, reduced chi^2, p-value,
# and AIC for every row in every file.

import glob
import os

import pandas as pd

base_dir = os.path.dirname(__file__)

model_files = {
    'Linear': sorted(glob.glob(os.path.join(base_dir, 'fit_results_linear_global_MC*.csv'))),
    'Logarithmic': sorted(glob.glob(os.path.join(base_dir, 'fit_results_log_global*.csv'))),
    'Soto-Tarrus': sorted(glob.glob(os.path.join(base_dir, 'fit_results_ST_global*.csv'))),
}

# Use the same k value as in the original comparison script.
model_k = {
    'Linear': 1,
    'Logarithmic': 2,
    'Soto-Tarrus': 3,
}

print('=== All fit statistics (all rows, all files) ===')

for model_name, filenames in model_files.items():
    if not filenames:
        print(f'No files found for {model_name}.')
        continue

    k = model_k[model_name]

    for filename in filenames:
        print(f'\n--- {model_name}: {os.path.basename(filename)} ---')

        try:
            df = pd.read_csv(filename)
        except Exception as exc:
            print(f'Could not load {filename}: {exc}')
            continue

        if df.empty:
            print('No data found in this file.')
            continue

        # AIC formula used in the original script.
        n = 12
        df['AIC'] = df['chi_sq'] + 2.0 * k + 2.0 * k * (k + 1) / (n - k - 1)

        for _, row in df.iterrows():
            ensemble = row.get('Ensemble', '')
            data_type = row.get('Data_Type', '')
            branch = row.get('Branch', '')
            chi2 = row.get('chi_sq', '')
            red_chi2 = row.get('red_chi_sq', '')
            p_value = row.get('p_value', '')
            aic = row.get('AIC', '')

            print(
                f"{ensemble} | {data_type} | {branch} | "
                f"chi2 = {chi2:.10g} | red_chi2 = {red_chi2:.10g} | "
                f"p_value = {p_value:.10g} | AIC = {aic:.10g}"
            )

print('\nDone.')
