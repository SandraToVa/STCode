import pandas as pd
import numpy as np
import ast

def process_trace_data(file_path, output_path="summary_statistics.csv"):
    # 1. Load the CSV file
    # Replace 'your_data.csv' with the actual path to your file
    df = pd.read_csv(file_path)
    
    # 2. Define a helper function to convert string lists to numpy arrays
    def parse_traces(val):
        if isinstance(val, str):
            try:
                # Converts the string "[val1, val2, ...]" into an actual Python list
                return np.array(ast.literal_eval(val))
            except (ValueError, SyntaxError):
                return np.array([])
        return np.array(val)

    print("Parsing arrays... this might take a moment depending on file size.")
    # Apply the conversion to both trace columns
    df['y0_traces_array'] = df['y0_traces'].apply(parse_traces)
    df['L2_traces_array'] = df['L2_traces'].apply(parse_traces)
    
    # 3. Calculate mean and uncertainty (standard deviation) for each individual cell
    # This processes the 1000 values inside each cell
    df['y0_cell_mean'] = df['y0_traces_array'].apply(np.mean)
    df['y0_cell_std'] = df['y0_traces_array'].apply(np.std)
    
    df['L2_cell_mean'] = df['L2_traces_array'].apply(np.mean)
    df['L2_cell_std'] = df['L2_traces_array'].apply(np.std)
    
    # 4. Group by Data_Type and Branch to get the final statistics
    # This aggregates the cell results by your categories
    summary = df.groupby(['Data_Type', 'Branch']).agg(
        y0_mean=('y0_cell_mean', 'mean'),
        y0_uncertainty=('y0_cell_std', 'mean'),
        L2_mean=('L2_cell_mean', 'mean'),
        L2_uncertainty=('L2_cell_std', 'mean'),
        row_count=('Data_Type', 'count') # Counts how many rows were in this group
    ).reset_index()
    
    # 5. Save the summary to a new CSV and display it
    summary.to_csv(output_path, index=False)
    print(f"\nProcessing complete. Summary saved to {output_path}\n")
    print(summary)
    
    return summary, df

# --- How to run it ---
# Call the function with your specific file name
summary_stats, full_processed_data = process_trace_data('fit_results_log_global_Ar0.csv')

