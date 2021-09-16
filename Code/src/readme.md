src folder
==========================

In `src` the most important files for all 'backend' calculations can be
found. We describe the files below

`classes.py`
----------------------------

Two main classes can be found in classes.py, we provide some details
below

#### - `DataImporter`

This class is used to import S&P500 option data. All data has been
pre-processed (using `__init__old`) and saved to a pickle for
efficiency. When the class is initialized we only read the processed
pickle. After this, there are a whole lot of functions which can be used
to import smiles and surfaces in different ways.

#### - `MatlabMonteCarlo`

This class is used to simulate a Monte Carlo smile (complete with greeks
for all strikes) using Matlab. When the class is initialized it starts
the Matlab Python engine after which simulations can be made using the
`simulateSABRSmile` function. The Matlab code in the `matlab_code`
folder is used in this function.

Both these classes thus are able to export 'smiles'. A smile in the
context of this code is a pandas dataframe with the columns for the
strike price, time to expiry, implied volatility, etc. A smile dataframe
can be used directly in many other functions (in hindsight, a smile
class would have been much better of course).

`helperfunctions.py`
--------------------------------------------

This file only consists of a range of commonly used functions, e.g. easy
calculation of Black-Scholes greeks, and derivatives.

`SABR.py`
----------------------

In this file all main functions used for SABR are found. For example, we
can first import an S&P500 smile using the `DataImporter` class (or
simulate an equivalent Pandas Dataframe using `MatlabMonteCarlo`) and
then calibrate the smile using the `calibrate_smile` function.
Equivalently, a complete surface can also be imported and calibrated
using `calibrate_surface`. There are also functions to calculate SABR
delta's and vega's (or functions to add the greeks to a complete
surface).

Most functions require an 'approximator' variable. This is an
initialized `ImpliedVolatilityApproximator` class from
`SABR_IV_approximators.py` below.

`SABR_IV_approximators.py`
--------------------------------------------------------

In this file, all implied vol approximators can be found. We have the
`ImpliedVolatilityApproximator` class with the functions `calc_call`
(calculates call price), `calc_iv` (calculates the IV) and `get_name`
(to get the name of the approximator).

We have children of the `ImpliedVolatilityApproximator` class for the
used approximators. Thus, here we have the Hagan, Antonov (2013) and
neural network approximators.

For testing purposes, we also have the Obloj approximator, Hagan normal
IV approximator and the exact Antonov (2013) approximator (with double
integrals).
