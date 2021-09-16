import pandas as pd
import numpy as np
import scipy.optimize as optimize
import helperfunctions as hf
import time
import matplotlib.pyplot as plt


def add_fit(approximator, alpha, beta, rho, v, market_smile, strike_price=False):
    """
    Adds the SABR implied volatilities for multiple given approximators and SABR parameters to a given smile.
    :param approximator: (list of) ImpliedVolatilityApproximator class(es) from SABR_IV_approximators.py
    :param alpha: SABR alpha
    :param beta: SABR beta
    :param rho: SABR correlation
    :param v: SABR vol of vol
    :param market_smile: pandas dataframe smile
    :param strike_price: Bool, if True use the actual strike price (not the relative strike).
    :return: market smile with extra columns with implied vols, col names are the approximator names.
    """
    if not strike_price:
        market_smile = market_smile.drop('strike_price', axis=1)
        market_smile = market_smile.rename(columns={'strike': 'strike_price'})
        market_smile['forward_price'] = 1
    T = market_smile['T'][0]  # hf.time_to_expiry(date.date(), exdate.date())

    if isinstance(approximator, list):
        for a in approximator:
            market_smile[a.get_name()] = get_fit(a, alpha, beta, rho, v, T, market_smile)
    else:
        market_smile[approximator.get_name()] = get_fit(approximator, alpha, beta, rho, v, T, market_smile)

    return market_smile


def get_fit(approximator, alpha, beta, rho, v, T, market_smile):
    """
    Calculates the SABR implied volatilities for a given approximator and SABR parameters for a given smile.
    :param approximator: ImpliedVolatilityApproximator class from SABR_IV_approximators.py
    :param alpha: SABR alpha
    :param beta: SABR beta
    :param rho: SABR correlation
    :param v: SABR vol of vol
    :param market_smile: pandas dataframe smile
    :return: numpy array with the implied volatilities
    """
    return market_smile.apply(
        lambda x: approximator.calc_iv(alpha, beta, rho, v, (x['strike_price'], x['forward_price']), T), axis=1)


def add_delta(approximator, alpha, beta, rho, v, T, market_smile):
    """
    Calculates the SABR delta for all strikes of a given smile. Returns as numpy array
    """
    return market_smile.apply(
        lambda x: delta(approximator, alpha, beta, rho, v, (x['strike_price'], x['forward_price']), T), axis=1)


def add_vega(approximator, alpha, beta, rho, v, T, market_smile):
    """
    Calculates the SABR vega for all strikes of a given smile. Returns as numpy array
    """
    return market_smile.apply(
        lambda x: vega(approximator, alpha, beta, rho, v, (x['strike_price'], x['forward_price']), T), axis=1)


def plot_SABR(approximator, alpha, beta, rho, v, T, strike_bounds=(0.4, 1.6), strike_step=.05):
    """
    Creates and plots a SABR implied volatility smile (as pandas dataframe) for a (list of) ImpliedVolatilityApproximator(s) for
    given SABR parameters. The bounds of the considered strikes are set via strike_bounds, and strike_step. We always
    consider F=1 in this function (although an extra parameter could have easily been added).

    This function is basically identical to get_fit but also creates a new smile and plots it.

    """
    strikes = np.arange(strike_bounds[0], strike_bounds[1] + strike_step, strike_step)
    strikes = np.round(strikes, 4)
    smile = pd.DataFrame(columns=['T', 'strike_price', 'forward_price'])
    smile['strike_price'] = strikes
    smile['T'] = T
    smile['forward_price'] = 1

    # Loop over all approximators if its a list
    if isinstance(approximator, list):
        approximator_names = [a.get_name() for a in approximator]
        for a in approximator:
            smile[a.get_name()] = get_fit(a, alpha, beta, rho, v, T, smile)
        smile.plot(x='strike_price', y=approximator_names)
    else:
        start = time.time()
        smile[approximator.get_name()] = get_fit(approximator, alpha, beta, rho, v, T, smile)
        print(time.time() - start)
        smile.plot(x='strike_price', y=approximator.get_name())

    return smile


def plot_fit(approximator, alpha, beta, rho, v, market_smile, strike_price=False):
    """
    For a given IV smile, adds the IV approximations for a list of approximators and SABR parameters and plots the fit.
    strike_price=True indicates that we use the real strike price (not relative).

    This again uses the get_fit function.

    """
    if not strike_price:
        market_smile = market_smile.drop('strike_price', axis=1)
        market_smile = market_smile.rename(columns={'strike': 'strike_price'})
        market_smile['forward_price'] = 1
    T = market_smile['T'][0]  # hf.time_to_expiry(date.date(), exdate.date())

    if isinstance(approximator, list):
        approximator_names = [a.get_name() for a in approximator]
        for a in approximator:
            market_smile[a.get_name()] = get_fit(a, alpha, beta, rho, v, T, market_smile)
        market_smile.plot(x='strike_price', y=['impl_volatility'] + approximator_names)
    else:
        market_smile[approximator.get_name()] = get_fit(approximator, alpha, beta, rho, v, T, market_smile)
        market_smile.plot(x='strike_price', y=['impl_volatility', approximator.get_name()])

    return market_smile


def plot_sim_fit(approximator, alpha, beta, rho, v, T, market_smile, strike_price=False, title=None):
    """
    Plots SABR fit of approximators in a simulated smile
    """
    if not strike_price:
        market_smile['forward_price'] = 1

    fig, ax = plt.subplots()
    plt.plot(market_smile['strike_price'], market_smile['impl_volatility'], label='Monte Carlo')

    if isinstance(approximator, list):
        for a in approximator:
            market_smile[a.get_name()] = get_fit(a, alpha, beta, rho, v, T, market_smile)
            plt.plot(market_smile['strike_price'], market_smile[a.get_name()], label=a.get_name())

    else:
        market_smile[approximator.get_name()] = get_fit(approximator, alpha, beta, rho, v, T, market_smile)
        market_smile.plot(x='strike_price', y=['impl_volatility', 'Hagan'])

    if title is not None:
        plt.title(title)
    plt.legend()
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.grid()

    return market_smile


def SABR_sum_squares(SABR_params, market_smile, approximator, T, beta):
    """
    Calculates the sum of squared errors from Equation (2). This is only used in a Nelder-Mead optimization, which is
    pretty slow (but reliable). Used in calibrate_smile function.
    """
    alpha, rho, v = SABR_params
    # create artificial bounds
    if alpha < 0 or v < 0 or abs(rho) > 1:
        return np.inf

    market_smile['approximation'] = market_smile.apply(
        lambda x: approximator.calc_iv(alpha, beta, rho, v, (x['strike_price'], x['forward_price']), T), axis=1)
    error = np.abs(market_smile['impl_volatility'] - market_smile['approximation'])
    output = np.sum(np.power(error, 2))
    if np.isnan(output):
        return np.inf
    return output


def residuals(SABR_params, market_smile, approximator, T, beta):
    """
    SABR residuals used for solving Equation (2) with Levenberg-Marquardt. Used in calibrate_smile function
    Returns a numpy array
    """
    alpha, rho, v = SABR_params
    # create artificial bounds
    if alpha < 0 or v < 0 or abs(rho) > 1:
        return np.array([np.inf] * len(market_smile))
    approximation = market_smile.apply(
        lambda x: approximator.calc_iv(alpha, beta, rho, v, (x['strike_price'], x['forward_price']), T), axis=1)
    error = market_smile['impl_volatility'] - approximation
    return error


def calibrate_smile(market_smile, approximator, starting_values=None, beta=.5, strike_price=False):
    """
    Main SABR calibration function.
    :param market_smile: dataframe with smile, can either be real market data, or simulated data.
    :param approximator: the implied volatility approximator instance
    :param starting_values: list with starting values, if None we use Le Floc'h and Kennedy (2014) starting values
    :param beta: SABR beta, we fix it at .5
    :param strike_price: bool, False means we use f=1 (simulated/standardised data).
    :return: the optimization result, and the calibration time.
    """
    if not strike_price:
        market_smile = market_smile.drop('strike_price', axis=1)
        market_smile = market_smile.rename(columns={'strike': 'strike_price'})
        market_smile['forward_price'] = 1

    T = market_smile['T'][0]

    if starting_values is None:
        starting_values = find_initial_values(market_smile, beta)

    # perform the actual calibrations
    try:
        start = time.time()
        result = optimize.least_squares(residuals, starting_values, method='lm',
                                        args=(market_smile, approximator, T, beta))
        stop = time.time() - start
    except Exception as e:
        print('Error in least squares: ' + str(e))
        print('Try default starting values')
        start = time.time()
        result = optimize.least_squares(residuals, [.1, -.1, .1], method='lm',
                                        args=(market_smile, approximator, T, beta))
        stop = time.time() - start

    return result, stop


def calibrate_surface(surface, approximator, beta=.5, strike_price=False):
    """
    SABR calibration to a complete surface (uses calibrate_smile for each maturity) and saves all parameters in
    a dataframe.
    :param surface: dataframe with surface, can either be real market data, or simulated data.
    :param approximator: the implied volatility approximator instance
    :param beta: SABR beta, we fix it at .5
    :param strike_price: bool, False means we use f=1 (simulated/standardised data).
    :return: dataframe with calibrated parameters for each maturity
    """
    # create dataframe for all calibrated parameters (for each maturity) and fill with nan
    params = surface[['date', 'exdate', 'days', 'T']].drop_duplicates().sort_values('days').reset_index(
        drop=True)
    params['alpha'], params['beta'], params['rho'], params['v'] = [np.nan,
                                                                   np.nan,
                                                                   np.nan,
                                                                   np.nan]
    params['time'] = np.nan

    for exdate in surface['exdate'].unique():
        smile = surface[(surface['exdate'] == exdate) & (~surface['impl_volatility'].isna())].reset_index(drop=True)
        try:
            res, time = calibrate_smile(smile, approximator, beta=beta, strike_price=strike_price)
        except Exception as e:
            print(e)
            continue
        # res = calibrate_smile(smile, AntonovApprox(), starting_values=None, beta=beta, strike_price=strike_price)
        # res2 = calibrate_smile(smile, approximator,starting_values=None, beta=beta, strike_price=strike_price)
        if not res.success:
            print('Not converged ===================')
        params = res.x
        # print('result: ' + str(params))
        params.loc[params['exdate'] == exdate, ['alpha', 'beta', 'rho', 'v', 'time']] = [params[0],
                                                                                         beta,
                                                                                         params[1],
                                                                                         params[2],
                                                                                         time]
    return params


def find_initial_values(smile, beta, default_output=None):
    """
    Calculates the initial parameters for SABR smile calibration following the methods of Le Floc'h and Kennedy (2014)
    :param smile: DataFrame with market smile
    :param beta: SABR beta parameters
    :param default_output: can be removed probably...
    :return: list with initial [alpha, rho, v] parameters.
    """
    if default_output is None:
        default_output = [.75, -.5, .75]
    fw = smile['forward_price'][0]

    offset = .1
    atm = smile.iloc[(smile['strike_price'] - fw).abs().argsort()[0]]
    below = smile[smile['strike_price'] < atm['strike_price']]
    if len(below) == 0:
        print('No strikes under ATM')
        return default_output
    below = below.iloc[(below['strike_price'] - (1 - offset) * fw).abs().argsort().reset_index(drop=True)[0]]

    up = smile[smile['strike_price'] > atm['strike_price']]
    if len(up) == 0:
        print('No strikes after ATM')
        return default_output
    up = up.iloc[(up['strike_price'] - (1 + offset) * fw).abs().argsort().reset_index(drop=True)[0]]
    z_min = np.log(below['strike_price'] / fw)  # z[0]
    z_0 = np.log(atm['strike_price'] / fw)  # z[1]
    z_plus = np.log(up['strike_price'] / fw)  # z[2]

    sigma_min = below['impl_volatility']  # sigma[0]
    sigma_0 = atm['impl_volatility']  # sigma[1]
    sigma_plus = up['impl_volatility']  # sigma[2]

    w_min = 1 / ((z_min - z_0) * (z_min - z_plus))
    w_0 = 1 / ((z_0 - z_min) * (z_0 - z_plus))
    w_plus = 1 / ((z_plus - z_min) * (z_plus - z_0))

    s = z_0 * z_plus * w_min * sigma_min + z_min * z_plus * w_0 * sigma_0 + z_min * z_0 * w_plus * sigma_plus
    s_ = -(z_0 + z_plus) * w_min * sigma_min - (z_min + z_plus) * w_0 * sigma_0 - (z_min + z_0) * w_plus * sigma_plus
    s__ = 2 * w_min * sigma_min + 2 * w_0 * sigma_0 + 2 * w_plus * sigma_plus

    alpha = s * fw ** (1 - beta)
    v_sq = 3 * s * s__ - 0.5 * (1 - beta) ** 2 * s ** 2 + 1.5 * (2 * s_ + (1 - beta) * s) ** 2

    if v_sq < 0:
        rho = .98 * np.sign(2 * s_ + (1 - beta) * s)
        v = (2 * s_ + (1 - beta) * s) / rho

        output = [alpha, rho, v]
        if np.nan in output or np.inf in np.abs(output):
            return default_output
        return output

    v = np.sqrt(v_sq)
    rho = (2 * s_ + (1 - beta) * s) / v
    if np.abs(rho) > 1:
        rho = .98 * np.sign(rho)
    output = [alpha, rho, v]

    if np.nan in output or np.inf in np.abs(output):
        return default_output
    return output


def delta(approximator, alpha, beta, rho, v, contract_params, T, dx=1e-4, print_output=False, r=0):
    """
    Computes the SABR delta for given approximator and parameters
    :param approximator: ImpliedVolatilityApproximator class instance
    :param alpha:
    :param beta:
    :param rho:
    :param v:
    :param contract_params: (strike, forward) tuple
    :param T: time to expiry
    :param dx: step size in central differences for the partial derivatives
    :param print_output: only used for debugging
    :param r: interest rate (always set at zero).
    :return: float delta value
    """
    K, F = hf.get_contract_parameters(contract_params)

    sigma_iv = approximator.calc_iv(alpha, beta, rho, v, (K, F), T)

    # computes the partial derivative to f using Appendix B.
    deriv_F = \
        hf.derivative(
            lambda x: approximator.calc_iv(alpha / np.sqrt((F + x) / F), beta, rho, v, (F * K / (F + x), F), T),
            0, 1, dx=dx)[0]

    if print_output:
        print('deriv_F : ' + str(deriv_F))

    # retrieve the black scholes greeks
    delta_bs = hf.calc_black_scholes_delta(sigma_iv, T, (K, F), r=r)
    vega_bs = hf.calc_black_scholes_vega(sigma_iv, T, (K, F), r=r)

    return delta_bs + vega_bs * deriv_F


def vega(approximator, alpha, beta, rho, v, contract_params, T, dx=1e-4, print_output=False):
    """
    Computes the SABR vega for given approximator and parameters
    :param approximator: ImpliedVolatilityApproximator class instance
    :param alpha:
    :param beta:
    :param rho:
    :param v:
    :param contract_params: (strike, forward) tuple
    :param T: time to expiry
    :param dx: step size in central differences for the partial derivatives
    :param print_output: only used for debugging
    :param r: interest rate (always set at zero).
    :return: float vega value
    """
    K, F = hf.get_contract_parameters(contract_params)

    sigma_iv = approximator.calc_iv(alpha, beta, rho, v, (K, F), T)

    # compute partial derivative to alpha
    deriv_alpha = hf.derivative(lambda x: approximator.calc_iv(x, beta, rho, v, (K, F), T), alpha, 1, dx=dx)[0]

    if print_output:
        print('deriv_alpha: ' + str(deriv_alpha))

    # Black-Scholes vega
    vega_bs = hf.calc_black_scholes_vega(sigma_iv, T, (K, F))

    return vega_bs * deriv_alpha
