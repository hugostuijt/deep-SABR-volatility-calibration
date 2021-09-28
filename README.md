# Deep calibration of SABR stochastic volatility
Master Thesis Quantitative Finance
--------------------------------------------

My thesis attempts to increase the accuracy of the SABR stochastic volatility model (used to model the implied volatility curve) while maintaining fast calibration speeds. I approximate a very accurate, but numerically intense SABR implied volatility function using a deepfeed neural network. Calibration accuracy is increased significantly on simulated data (based on implied vol estimates, as well as skew delta and vega calculations). However, performance lacks on real market data (S&P500 options). 

The thesis PDF can be found in the main folder. In each folder within the `Code` folder a separate readme with description of all files can be found. 

--------------------------------------------

ABSTRACT

We present a calibration method for the SABR stochastic volatility model based on machine learning
techniques. The goal is to increase calibration accuracy with little decrease in calibration speed.
Using an artificial neural network, we approximate a very accurate SABR option pricing map from
model parameters to implied volatility. The neural network approximation is more accurate than
the traditional formulas from Hagan et al. (2002), and more efficient than the computations from the
original pricing map. In a numerical experiment, we find that the neural network can calibrate to an
implied volatility smile with higher accuracy than the traditional formulas while being significantly
faster than the original pricing map. However, we find that care must be taken when using the
neural network with market data. By applying the methods to S&P500 options, we observe higher
uncertainty in the SABR parameters calibrated by the neural network, creating instabilities in the
estimated vega hedges. Nevertheless, the results show the potential for neural networks to improve
SABR calibration speed and accuracy significantly.

--------------------------------------------

