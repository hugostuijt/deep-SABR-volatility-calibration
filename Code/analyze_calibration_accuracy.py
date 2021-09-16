import sys
sys.path.append('src')
from src import helperfunctions as hf
from src import SABR
from src.SABR_IV_approximators import AntonovANN, AntonovApprox, Hagan

import pickle
import time
import json
import numpy as np
import pandas as pd

# suppress all numpy warnings
import warnings
warnings.filterwarnings("ignore")

# import 10,000 smiles from pickle
smiles = hf.get_mc_simulations()

cols = ['i', 'T', 'alpha_real', 'alpha', 'alpha_error', 'rho_real', 'rho', 'rho_error', 'v_real', 'v', 'v_error',
        'time']


# loads a json with a true/false such that it can be stopped from outside python. This is handy for the
# AntonovApprox such that the computations can be split over multiple nights.
def load_stop():
    with open('stop.json') as json_file:
        return json.load(json_file)


def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, l):
    with open(path, 'wb') as f:
        pickle.dump(l, f)


def convert_dict(params_dict):
    return [params_dict['alpha'], params_dict['rho'], params_dict['v'], params_dict['T']]


# relative error
def error(real, estimate):
    return np.abs(estimate - real) / np.abs(real)


def accuracy_df(approximator):
    strikes = np.round(np.arange(.4, 1.65, .05), 2)

    # create result dataframes
    df = pd.DataFrame(columns=cols)

    delta = pd.DataFrame(columns=strikes)
    delta_real = pd.DataFrame(columns=strikes)
    delta_error = pd.DataFrame(columns=strikes)

    vega = pd.DataFrame(columns=strikes)
    vega_real = pd.DataFrame(columns=strikes)
    vega_error = pd.DataFrame(columns=strikes)

    fit_error = pd.DataFrame(columns=strikes)

    count = 0
    l = len(smiles)
    for smile_list in smiles:
        # Enable stopping outside of python
        if load_stop():
            print('Stopped')
            break

        params, smile = smile_list

        # Save delta and vega per smile in a single dataframe
        delta_real = delta_real.append(smile[['strike', 'delta']].set_index(['strike']).T.reset_index(drop=True))
        vega_real = vega_real.append(smile[['strike', 'vega']].set_index(['strike']).T.reset_index(drop=True))

        # real parameters
        alpha_real, rho_real, v_real, T = convert_dict(params)

        try:
            calibration, time = SABR.calibrate_smile(smile, approximator)
        except:
            count += 1
            print('Foutmelding bij calibreren: ' + str(count))
            continue

        alpha, rho, v = calibration.x  # calibrated parameters

        # Create fit for calibrated parameters
        smile['fit'] = SABR.get_fit(approximator, alpha, .5, rho, v, T, smile)
        smile['delta_fit'] = SABR.add_delta(approximator, alpha, .5, rho, v, T, smile)
        smile['vega_fit'] = SABR.add_vega(approximator, alpha, .5, rho, v, T, smile)

        # Calculate the error of the fit
        smile['error'] = error(smile['impl_volatility'], smile['fit'])
        smile['delta_error'] = error(smile['delta'], smile['delta_fit'])
        smile['vega_error'] = error(smile['vega'], smile['vega_fit'])

        # Add error and fits to output dataframe
        delta = delta.append(smile[['strike', 'delta_fit']].set_index(['strike']).T.reset_index(drop=True))
        delta_error = delta_error.append(
            smile[['strike', 'delta_error']].set_index(['strike']).T.reset_index(drop=True))
        vega = vega.append(smile[['strike', 'vega_fit']].set_index(['strike']).T.reset_index(drop=True))
        vega_error = vega_error.append(smile[['strike', 'vega_error']].set_index(['strike']).T.reset_index(drop=True))
        fit_error = fit_error.append(smile[['strike', 'error']].set_index(['strike']).T.reset_index(drop=True))

        output = pd.DataFrame.from_dict(
            {'i': [count], 'T': [T], 'alpha_real': [alpha_real], 'alpha': [alpha],
             'alpha_error': [error(alpha_real, alpha)], 'rho_real': [rho_real], 'rho': [rho],
             'rho_error': [error(rho_real, rho)], 'v_real': [v_real], 'v': [v],
             'v_error': [error(v_real, v)], 'time': [time]})
        df = df.append(output)

        count += 1
        print('{counter:.2f}'.format(counter=count / l), end='\r')

    output_dict = {'delta': delta, 'delta_real': delta_real, 'delta_error': delta_error, 'vega': vega,
                   'vega_real': vega_real, 'vega_error': vega_error, 'fit_error': fit_error, 'df': df}

    output_dict = {key: output_dict[key].reset_index(drop=True) for key in output_dict.keys()}

    return output_dict


start = time.time()
accuracy_hagan = accuracy_df(Hagan())
save_pickle(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\Thesis\Data\results\calibration_hagan.pkl", accuracy_hagan)
print(time.time() - start)
start = time.time()
accuracy_ANN = accuracy_df(AntonovANN())
save_pickle(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\Thesis\Data\results\calibration_ann.pkl", accuracy_ANN)
print(time.time() - start)
# The AntonovApprox() version takes very, very long to complete. It is split up over multiple nights.
accuracy_antonov = accuracy_df(AntonovApprox())
save_pickle(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\Thesis\Data\results\calibration_antonov.pkl", accuracy_antonov)
