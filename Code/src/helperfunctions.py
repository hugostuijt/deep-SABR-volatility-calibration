import numpy as np

from py_lets_be_rational.exceptions import AboveMaximumException, BelowIntrinsicException
from py_vollib.black import implied_volatility
from py_vollib.black.greeks import analytical as greeks
import pickle


def derivative(func, x0, dim, dx=1.0):
    """
    Computes the 1st derivative of a function using central differences. We use scipy.misc.derivative as a base for this
    function, but optimize it for n=1 and order=3. This way, we can remove the computation for the middle point because
    the weight is zero. This makes computations of SABR greeks with Monte Carlo faster.
    """
    order = 3
    weights = np.array([-1, 0, 1]) / 2.0

    val = 0.0 * np.ones(dim)
    ho = 1
    for k in range(order):
        if weights[k] == 0:
            continue
        val += weights[k] * func(x0 + (k - ho) * dx)
    return val / dx


def get_mc_simulations():
    """
    Loads pickle with the 10,000 simulated Monte Carlo smiles
    :return:
    """
    with open(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\Thesis\Data\mc_simulations_new.pkl", 'rb') as f:
        return pickle.load(f)


def time_to_expiry(startdate, enddate):
    """
    Computes the time to expiry using actual/365.
    """
    return (enddate - startdate).days / 365


def v_t_upper(T_t, T_s=30 / 365, v_s=7):
    """
    Computes the upper bound of v using the McGhee (2018) heuristic, see Equation (5)
    """
    return v_s * np.sqrt(T_s / T_t)


def v_t_lower(T_t, T_s=30 / 365, v_s=.05):
    """
    Computes the lower bound of v using the McGhee (2018) heuristic, see Equation (5)
    """
    return v_s * np.sqrt(T_s / T_t)


def get_contract_parameters(contract_params):
    if isinstance(contract_params, (int, float)):
        F = 1
        K = contract_params
    else:
        K, F = contract_params
    return K, F


def calc_black_scholes_iv(call_price, contract_params, T, r=0):
    """
    Computes the BS implied volatility using pyvollib (Jaeckel (2015))
    :param call_price: .
    :param T: time to expiry
    :param contract_params: tuple, (strike, forward)
    :param r: interest rate
    :return: float Black implied volatility
    """
    K, F = get_contract_parameters(contract_params)

    try:
        iv = implied_volatility.implied_volatility(call_price, F, K, r, T, 'c')
    except AboveMaximumException:
        iv = np.nan
    except BelowIntrinsicException:
        iv = np.nan

    return iv


def calc_black_scholes_delta(sigma_iv, T, contract_params, r=0):
    """
    Computes the BS delta using pyvollib (Jaeckel (2015))
    :param sigma_iv: implied volatility
    :param T: time to expiry
    :param contract_params: tuple, (strike, forward)
    :param r: interest rate
    :return: float Black delta
    """
    K, F = get_contract_parameters(contract_params)
    delta = greeks.delta('C', F, K, T, r, sigma_iv)

    return delta


def calc_black_scholes_vega(sigma_iv, T, contract_params, r=0):
    """
    Computes the BS vega using pyvollib (Jaeckel (2015))
    :param sigma_iv: implied volatility
    :param T: time to expiry
    :param contract_params: tuple, (strike, forward)
    :param r: interest rate
    :return: float Black vega
    """
    K, F = get_contract_parameters(contract_params)
    vega = greeks.vega('C', F, K, T, r, sigma_iv)

    return vega * 100  # multiply with 100. Pyvollib returns the vega for a 1 percentage point increase in vol


def make_smile_smooth(df):
    """
    Makes as (simulated) IV smile smooth by only keeping the convex part, and removing unrealistic values
    :param df: simulated smile (pandas dataframe)
    :return: dataframe with smoothened smile
    """

    # loop over each strike
    for i, r in df.iterrows():
        if (i == 0) | (i == len(df) - 1):
            continue

        k_step1 = df.loc[i, 'strike'] - df.loc[i - 1, 'strike']
        k_step2 = df.loc[i + 1, 'strike'] - df.loc[i, 'strike']
        deriv1 = (df.loc[i, 'impl_volatility'] - df.loc[i - 1, 'impl_volatility']) / k_step1
        deriv2 = (df.loc[i + 1, 'impl_volatility'] - df.loc[i, 'impl_volatility']) / k_step2

        # Check the derivative of the IV smile in subsequent steps
        if deriv2 < deriv1:
            k = r['strike']
            if k < 1:
                df = df.loc[df['strike'] > r['strike']].reset_index(drop=True)
            else:
                df = df.loc[df['strike'] < r['strike']].reset_index(drop=True)
            return make_smile_smooth(df)
    return df
