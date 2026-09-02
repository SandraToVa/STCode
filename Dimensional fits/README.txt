
##########################
SETS OF DATA:
##########################
- Francesco: 1 data set with 3 points "data_fran.cs"
- Brambilla: 2 data sets one at A=A_r0 and the other at A_pi12 "data_bram_Ar0.csv" and "data_bram_Api12.csv"
	Inside each data set the data is divided by:
	- Bare / Smeared
	- Ensamble M_i, M_ii, M_iii

##########################
FITS:
##########################
--------------------------------------------
#Francesco fits are done to the hole set
--------------------------------------------
- Lineal fit -> Only 1 parameter sigma_0 is fitted
I create the scrip "fit_fran_lineal.py"
That generate "fit_results_lineal_fran.csv"

- Log fit (uncompleted) -> Only 1 parameter sigma_0 is fitted
I create the scrip "fit_fran_log.py"
That generate "fit_results_log_fran.csv"

- Log fit v2 (complete) -> Fit of 2 parmeters sigma_0 and lamb_2prime
MISSING!

--------------------------------------------
#Brambilla fits can be done for each ensamble or to the hole data set
--------------------------------------------

## First I separate Ar0 and Api12 data sets and do a subseparation of bare and smeared and within all this separations I do the fit to separat Ensambles. So each Ensamble has one fit

- Lineal fit -> Only 1 parameter sigma_0 is fitted
I create the scrip "fit_bram_lineal.py"
That generate "fit_results_lineal_Ar0.csv" and "fit_results_lineal_Api12.csv"

- Log fit (uncompleted) -> Only 1 parameter sigma_0 is fitted
I create the scrip "fit_bram_log.py"
That generate "fit_results_log_Ar0.csv" and "fit_results_log_Api12.csv"

- Log fit v2 (complete) -> Fit of 2 parmeters sigma_0 and lamb_2prime
I create the scrip "fit_bram_log_v2.py"
That generate "fit_results_log_Ar0_v2.csv" and "fit_results_log_Api12_v2.csv"


## Another thing that can be done is separate the fits by Ar0 or Api12 and also bare / smeared but do the fit using the 3 enambles

- Lineal fit -> Only 1 parameter sigma_0 is fitted
I create the scrip "fit_bram_lineal_global.py"
That generate "fit_results_lineal_global_Ar0.csv" and "fit_results_global_lineal_Api12.csv"

- Log fit v2 (complete) -> Fit of 2 parmeters sigma_0 and lamb_2prime
I create the scrip "fit_bram_log_global.py"
That generate "fit_results_log_global_Ar0.csv" and "fit_results_global_log_Api12.csv"

##########################
PLOTS:
##########################
Regular treatment of the data get plots for each ensamble, data type and Amplitude used, with a grouped image. Also a heat map of the chi^2/dof compared.

Global plots only get plots depending on the data type and Amplitude. Also the heat map of the chi^2/dof.

