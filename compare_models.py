
# Code to compare the goodness of fit for the three models (Linear, Quadratic, Logarithmic)
# using the chi-squared and AIC criteria.[cite: 2]
# This updated version evaluates all branches and selects the BEST fit based on the lowest AIC.

import pandas as pd

A_types = ['Ar0', 'Api12']
A_index = 0

models = {
    'Linear': ('fit_results_linear_global_' + A_types[A_index] + '.csv', 1), # k=1
    'Logarithmic': ('fit_results_log_global_' + A_types[A_index] + '.csv', 2), # k=2 (assuming s0, c_pipi)
    'Soto-Tarrus': ('fit_results_ST_global_' + A_types[A_index] + '.csv', 2) # k=2 
}


data_type_to_check = 'smeared' # Change to 'smeared' if needed

print("=== Goodness of Fit Comparison (Best Branches) ===")
for model_name, (filename, k) in models.items(): #[cite: 2]
    try:
        df = pd.read_csv(filename) #[cite: 2]
        
        # Filter for the specific data type
        df_filtered = df[df['Data_Type'] == data_type_to_check].copy() #[cite: 2]
        
        if df_filtered.empty:
            print(f"--- {model_name} Model ---")
            print(f"No data found for Data_Type: '{data_type_to_check}'\n")
            continue
            
        # Calculate AIC for all branches of this model
        # AIC Formula: Chi^2 + 2*k [cite: 2]
        df_filtered['AIC'] = df_filtered['chi_sq'] + 2 * k #[cite: 2]
        
        # --- SELECTION LOGIC ---
        # Find the index of the row with the minimum AIC score
        best_idx = df_filtered['AIC'].idxmin()
        best_row = df_filtered.loc[best_idx]
        
        chi2 = best_row['chi_sq'] #[cite: 2]
        red_chi2 = best_row['red_chi_sq'] #[cite: 2]
        p_value = best_row['p_value'] #[cite: 2]
        best_aic = best_row['AIC']
        
        print(f"--- {model_name} Model ---") #[cite: 2]
        print(f"Selected Best Branch (Row Index {best_idx})")
        print(f"Chi-Sq: {chi2:.2f} | Reduced Chi-Sq: {red_chi2:.2f} | P-Value: {p_value:.2f} | AIC: {best_aic:.2f} \n") #[cite: 2]
        
    except Exception as e: #[cite: 2]
        print(f"Could not load/parse {model_name} from {filename}: {e}\n") #[cite: 2]

print("Note: The model with the LOWEST AIC is preferred.") #[cite: 2]