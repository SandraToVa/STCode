import numpy as np
import matplotlib.pyplot as plt
import os

#Data paths for different ensembles
data_path_D200 = '/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/sbdata_Francesco/D200'
data_path_N200 = '/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/sbdata_Francesco/N200'
data_path_N203 = '/Users/sandra/Documents/Doctorat/Projectes PhD/String tension/sbdata_Francesco/N203'

# List of ensemble names you have
ensembles = ['D200', 'N200', 'N203']
markers = {'D200': 'o', 'N200': 's', 'N203': '^'}
colors = {'V0': 'blue', 'V1': 'red', 'V2': 'green'}

for ens in ensembles:
    plt.figure(figsize=(10, 7))

    # Construct the full path to the file
    filename = f'potentials{ens}.dat'
    full_path = os.path.join(globals()[f'data_path_{ens}'], filename)
    
    if os.path.exists(full_path):
        # Load the data, ignoring the comment line at the top
        data = np.loadtxt(full_path, comments='%')
        
        # Calculate r/a from (r/a)^2 (which is in the first column)
        r = np.sqrt(data[:, 0])
        
        # Extract potentials and errors
        v0, v0_err = data[:, 1], data[:, 2]
        v1, v1_err = data[:, 3], data[:, 4]
        v2, v2_err = data[:, 5], data[:, 6]
        
        # Plot V0
        plt.errorbar(r, v0, yerr=v0_err, fmt=f'{markers[ens]}-', 
                     color=colors['V0'], label=f'$V_0$ ({ens})', 
                     capsize=3, alpha=0.8)
        
        # Plot V1
        plt.errorbar(r, v1, yerr=v1_err, fmt=f'{markers[ens]}--', 
                     color=colors['V1'], label=f'$V_1$ ({ens})', 
                     capsize=3, alpha=0.8)
        
        # Plot V2
        plt.errorbar(r, v2, yerr=v2_err, fmt=f'{markers[ens]}:', 
                     color=colors['V2'], label=f'$V_2$ ({ens})', 
                     capsize=3, alpha=0.8)
    else:
        print(f"Warning: The file was not found at {full_path}")

    # Formatting the plot
    plt.xlabel('$r/a$', fontsize=14)
    plt.ylabel('$a(V(r) - 2E_B)$', fontsize=14)
    plt.title(f'Static Potentials ($V_0, V_1, V_2$) for Ensemble {ens}', fontsize=16)

    # Place legend outside the plot if it gets too crowded
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(f'potentials_comparison_{ens}.png', dpi=300)
    print(f"Plot saved: potentials_comparison_{ens}.png")

    # Display the plot
    plt.show()