# String Tension Fits

As it is written in the $\LaTeX$ documentation, we want to check whether our string tension dependence found in [Soto, 2026](
https://doi.org/10.48550/arXiv.2606.05791) on the light quark mass works or not. 

All the datils can be found on my **Detailed Notes: Lattice QCD String Tension
Extrapolations**. In here I'll just explain who this folder is organized.

## Folders
- `\Data\`: In here we have the Brambilla lattice data from [Brambilla, 2023](http://arxiv.org/abs/2206.03156) `data_bram.csv` and the Francesco lattice data from [Bulava, 2024](http://arxiv.org/abs/2403.00754) `data_fran.csv`. 
    - `\sbdata_Francesco\`: In here we have the raw data from [Bulava, 2024](http://arxiv.org/abs/2403.00754). The data from `data_fran.csv` is not properly treated so using the $V_2$ potentials and the code `sigma_fit.py` we obtain the real values for $\sigma a^2$ for each <ins>Ensamble</ins>. This results in the `best_sigma_fit.csv`file.
    - The brabilla files `data_bram.csv` are separated in bare data or in smeared data. All the treatment is done separately for each. There is no correlation between the data.
- `\Dimensional fits\`: This folder is a **outdated** version of the current folder. In here instead of doing a dimensionles fit of $\sigma a^2$ vs $am_l$, we did the dimensionfull fit $\sigma$ vs $m_l$. This is a poor treatment because it ends up with hughe uncertanties due to including the value of $a$ and not working directly with the data given ($\sigma a^2$ and $am_l$).
- `\Bram_Linear\`, `\Bram_Log\`, and `\Bram_ST\`: In this folders we include the plots for the respective linear, logarithmic and Soto-Tarrús (ST) model fits for the Brambilla Data. This are obtained executing the corresponfing `plot_bram_linear.py`, `plot_bram_log.py`, and `plot_bram_ST.py`as well as their global versions. 
- Francesco data is also to be stored in equal `\Fran_\`folders when computed.
- `\All fitted\`: This folder contains all the scrpits as well as the plots corresponding to all the fits by using the alternative method where all the parameters of the fit are obtained using the lattice data. _This file is yet to be competed_.
- `\P-value\`: In here we store the p-value comparison plots between the fits obtained with `plot_goodness.py` and `compare_models.py` scripts.


## Files

### Brambilla Data

Brambilla data has two choices ($A_{r_0}$ or $A_{\pi/12}$), two <ins>Data_Types</ins> (bare or smeared) and three <ins>Ensambles</ins> (M_i, M_ii, M_iii). Different Data_Types and $A$ are always fitted separatedly.

Fitting files have one simple purpose: read the lattice data from the `\Data\` folder and fit the data using the corresponding model. This model can be either **linear**, **logarithmic** or **ST model**. 

- **Linear**: We have 4 kinds of files that fit with a linear model with only one free parameter $y_0$. The slope $\lambda'$ in this case is the one obtained in [Soto, 2026](
https://doi.org/10.48550/arXiv.2606.05791) and has two <ins>Branches</ins>.

    - `fit_bram_linear.py`: Fits Ensamble by Ensamble with the two Branches. Uses a rather simple model to obtain the statistics of the fit. Generates `fit_results_linear_Api12.csv` and `fit_results_linear_Ar0.csv`. **File not used**.

    - `fit_bram_linear_global.py`: Fits all the Enambles from the same Data_Type and choice with the two Branches. Uses a rather simple model to obtain the statistics of the fit. Generates `fit_results_linear_global_Api12.csv` and `fit_results_linear_global_Ar0.csv`. **File not used**.

    - `fit_bram_linear_MC.py`: Fits Ensamble by Ensamble with the two Branches. Uses a Monte Carlo procedure to obtain the uncertainties. Generates `fit_results_linear_MC_Api12.csv` and `fit_results_linear_MC_Ar0.csv`. **File used**.

    - `fit_bram_linear_global_MC.py`: Fits all the Enambles from the same Data_Type and choice with the two Branches. Uses a Monte Carlo procedure to obtain the uncertainties. Generates `fit_results_linear_global_MC_Api12.csv` and `fit_results_linear_global_MC_Ar0.csv`. **File used**.

    The difference between the MC files and the regula procedure is negligible. We obtain the same result.


- **Logarithmic**: We have 2 kinds of files that fit with a log model with two free parameters $y_0$ and $\lambda''$. The other two parameters $\lambda'$ and $c_{\pi\pi}$ are the ones from [Soto, 2026](
https://doi.org/10.48550/arXiv.2606.05791) and we have four <ins>Branches</ins>.

    - `fit_bram_log.py`: Fits Ensamble by Ensamble with the four Branches. Uses a Bootstrap Monte Carlo model to obtain the statistics of the fit. Generates `fit_results_log_Api12.csv` and `fit_results_log_Ar0.csv`.

    - `fit_bram_log_global.py`: Fits all the Enambles from the same Data_Type and choice with the two Branches. Uses a Bootrstap Monte Carlo model to obtain the statistics of the fit. Generates `fit_results_log_global_Api12.csv` and `fit_results_log_global_Ar0.csv`.

    If instead of only fitting one or two parameters with the linear and logarithmic fit repectively we fit all the parameters of each fit we obtain the `\All fitted\`results using the script `fit_bram_lin_and_log.py` and `fit_bram_lin_and_log_global.py`. Which generates the `fit_results_full_#.csv` files.

- **ST**: We have 2 kind of files that fit with a ST model with three free parameters $y_0$, $c_l$ and $\mu$. There are no parameters extracted from any other reference there for has no <ins>Branches</ins>. This moel is extracted from [Soto, 2021](https://link.aps.org/doi/10.1103/PhysRevD.104.074027).

    - `fit_bram_ST.py`: Fits Ensamble by Ensamble. Uses an effective variance minimization teachnique to obtain the statistics of the fit. Generates `fit_results_ST_Api12.csv` and `fit_results_ST_Ar0.csv`.

    - `fit_bram_ST_global.py`: Fits all the Enambles from the same Data_Type and choice with the two Branches. Uses an effective variance minimization teachnique to obtain the statistics of the fit. Generates `fit_results_ST_global_Api12.csv` and `fit_results_ST_global_Ar0.csv`.

    If we consider that there is a term proportional to $\Lambda_\text{QCD}$ in the light quark mass, we have an extra parameter to fit. The results are in `\All fitted\All fitted ST\`. The fitting codes above have an option to turn on and off the $\Lambda_\text{QCD}$ dependence. Generates `fit_results_ST_full_Api12.csv`, `fit_results_ST_full_Ar0.csv`, `fit_results_ST_full_global_Api12.csv` and `fit_results_ST_full_global_Ar0.csv`.


Plotting files directly read the `fit_results_.csv`files generated and plot the fits in svg files. We have `plot_bram_#.py` and `plot_bram_#_global.py` files for each model.

### Francesco Data

Francesco data has only 3 data points. However, insteda of having the results in terms of $am_l$ we have it in terms of $\mu_l$. Therefore one firts need the expression $\mu_l=\mu_l(am_l)$.

Fitting files have one simple purpose: read the lattice data from the `\Data\` folder and fit the data using the corresponding model. This model can be either **linear**, **logarithmic** or **ST model**. 

- **Linear**: 