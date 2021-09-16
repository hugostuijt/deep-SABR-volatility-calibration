Deep calibration of SABR stochastic volatility
==================================================================================================

Description of code
--------------------------------------------

by Hugo Stuijt

In this main folder all scripts and notebooks used for analysis of the
model can be found. We also have notebooks for the creation of neural
network training data and the training of the network itself.

In the `src` subfolder all 'backend' functions and classes can be found,
there's a readme file in there with more details. If you want to know in
detail what happens in my code it's best to start here.

There's also a `matlab_code` folder, all Monte Carlo simulations are
implemented in Matlab and accessed via the Matlab Python engine (Matlab
was a lot faster than Python with the simulations).

One thing which might be important to note at this point is the
`SABR_IV_approximators.py` file from `src`. This is where all implied
volatility approximator classes are found (they are children from the
`ImpliedVolatilityApproximator` class). In many of the scripts in this
main folder I initiate them using e.g. `Hagan()`, an implied vol
estimate can be calculated using `Hagan().calc_iv(...)`

List of Notebooks
----------------------------------------

### Plots

All Notebooks for plots of the results

-   `Analyse simulations.ipynb`: all plots from the simulations
    (accuracy and calibration accuracy) are made here. Calculation of
    the results is done in the .py files and saved in dictionaries.
-   `plot bayesian calibration.ipynb`: using samples created in
    `Bayesian Calibration.ipynb`. The greeks are computed in here as
    well.
-   `plot hagan SPX params.ipynb`: calibrates the full sample SPX
    options with Hagan to see how the parameters move over time. Also
    calibrates using AntonovANN and AntonovExact for the full sample 30
    day results.
-   `plot heatmap.ipynb`: neural network accuracy heatmap
-   `plot interpolation error.ipynb`: figure in appendix
-   `plot neural network examples.ipynb`: two axis plot with errors
-   `plot spx full sample.ipynb`: results for full sample spx
    (calculations made in `plot hagan SPX params.ipynb`).

### Additional Notebooks

-   `ANN_training_pointwise.ipynb`: Network training
-   `Bayesian Calibration.ipynb`: Bayesian calibration on simulated
    smile, posterior samples are saved and used in
    `plot bayesian calibration.ipynb`.
-   `Bayesian Calibration.ipynb`: same but on an SPX smile
-   `Training_data_creation.ipynb`: fixed grid, and random training
    datasets, all created using pandas with `swifter` (multithreading).
    All batches are saved as pickles and then aggregated in a later
    stage to single pickles. Part of the random data creation is used in
    the neural network accuracy heatmap
-   `smile simulations.ipynb`: Monte Carlo simulation of the 10,000
    smiles. Smiles are saved while running of the code such that it can
    be stopped during runtime without problems.
-   `SPX surface evaluation.ipynb`: results and plots of the SPX full
    surface calibration (Augst 5th 2019)

List of Python files
----------------------------------------------

-   `analyze_calibration_accuracy.py`: calibrates the 10,000 simulated
    smiles and saves the results for all approximators
-   `analyze_simulations_accuracy.py`: very similar file, but does all
    calculations by treating the parameters as known.

These files are used in the numerical experiment. They are very similar
in that they compute the same results, but one of the treats the SABR
parameters as known and the other calibrates the smile first. These
scripts take very long to run for the `AntonovApprox` approximator such
that in practice they are split over multiple nights.

In both files, the results for each file are stored in dictionaries
(with dataframes).
