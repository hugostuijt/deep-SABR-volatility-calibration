import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import matlab.engine
import os
import random

import helperfunctions as hf


def transform_OptionMetrics(df):
    """
    Basic transormations of standard options metrics data
    :param df: optionmetrics data in dataframe
    :return: transformed dataframe
    """
    df['strike_price'] /= 1000
    df['forward_price'] = np.nan
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    df['exdate'] = pd.to_datetime(df['exdate'].astype(str), format='%Y%m%d')

    df['days'] = (df['exdate'] - df['date']).dt.days
    df['T'] = df['days'] / 365

    return df


def thin_surface(surface, n_strikes, strike_bounds):
    """
    With input of complete surface dataframe, thins it according to the number of strikes and the bounds of the strikes,
    thus removes strikes for each maturity.
    :param surface: dataframe with complete surface
    :param n_strikes: maximum number of strikes for each maturity
    :param strike_bounds: tuple (minimum strike, maximum strike).
    :return:
    """
    days = surface['days'].unique()

    min_strike = strike_bounds[0]
    max_strike = strike_bounds[1]

    df = pd.DataFrame()

    for day in days:
        smile = surface[surface['days'] == day]
        smile = smile.loc[(smile['strike'] >= min_strike) & (smile['strike'] <= max_strike)].reset_index(drop=True)
        smile = smile.reset_index(drop=True)
        n = len(smile)  # number of strikes
        arr = np.arange(n)
        idx = np.round(np.linspace(0, len(arr) - 1, n_strikes)).astype(int)
        smile = smile.loc[idx]
        smile = smile.drop_duplicates()

        df = df.append(smile)
    return df


class DataImporter:
    DATA_PATH = Path(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\data_final.pkl")

    DEFAULT_COLS = ['date', 'exdate', 'days', 'T', 'cp_flag', 'best_bid', 'best_offer', 'strike', 'strike_price',
                    'impl_volatility', 'forward_price', 'spot']

    def __init__(self, data_path=DATA_PATH):
        self.data = pd.read_pickle(data_path)

    def __init__old(self, data_path=DATA_PATH):
        """" Only initially for some processing, after this the self.data dataframe is saved to a pickle and the
        normal init is used for efficiency
        """
        start = time.time()
        self.data = pd.read_csv(data_path)
        print('Data imported: ' + str(np.round(time.time() - start, 1)))

        # Add spot price and strike
        self.spot = pd.read_csv(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\thesis\data\SPX_spot.csv")
        self.spot['date'] = pd.to_datetime(self.spot['date'])
        self.data['spot'] = 0
        for date in self.data['date'].unique():
            spot = self.get_spot(date)
            self.data.loc[self.data['date'] == date, 'spot'] = spot
        self.data['strike'] = self.data['strike_price'] / self.data['spot']

        # add forward prices
        forward = pd.read_csv(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\Thesis\Data\forward.csv")
        forward['date'] = pd.to_datetime(forward['date'].astype(str), format='%Y%m%d')
        forward['expiration'] = pd.to_datetime(forward['expiration'].astype(str), format='%Y%m%d')
        forward = forward[['date', 'expiration', 'ForwardPrice']]
        forward = forward.rename(columns={'expiration': 'exdate'})

        self.data = transform_OptionMetrics(self.data)  # , self.yieldcurve, self.div_yield)

        self.data = self.data.merge(forward)
        self.data['forward_price'] = self.data['ForwardPrice']

        self.data.to_pickle(r"C:\Users\hugo\OneDrive\Documents\Quantitative Finance\data_final.pkl")

    def print_cols(self):
        print(self.data.columns)

    def print_dates(self):
        dates = self.data['date'].unique()
        dates = sorted(dates)
        for dt in dates:
            print(pd.to_datetime(dt).strftime('%Y-%m-%d'))

    def get_spot(self, date):
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")
        return float(self.spot.loc[self.spot['date'] == date, 'close'])

    def print_exdates(self, date, min_maturity=None, max_maturity=None):
        surface = self.surface_date(date)
        surface['T'] = (surface['exdate'] - surface['date']).dt.days / 365
        if max_maturity is not None:
            surface = surface.loc[surface['T'] <= max_maturity]
        if min_maturity is not None:
            surface = surface.loc[surface['T'] >= min_maturity]

        dates = sorted(surface['exdate'].unique())
        for dt in dates:
            print(pd.to_datetime(dt).strftime('%Y-%m-%d'))

    def get_surface(self, date, days=None):
        """
        Returns a dataframe with the SPX surface for a given date and given days to expiry
        :param date: trading date, can be datetime or integer with format yyyymmdd
        :param days: list with the days to expiry of interest
        :return: pandas dataframe with surface
        """
        if days is None:
            days = [7, 14, 21, 30, 60, 90, 180, 270, 360, 720]
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")
        surface = self.surface_date(date)
        surface_days = surface['days'].unique()
        keys = [min(surface_days, key=lambda x: abs(x - day)) for day in days]

        surface = pd.DataFrame()
        for day in keys:
            surface = surface.append(self.smile_days(date, day, n_strikes=25, strike_bounds=(.4, 1.6)))

        return surface

    def surface_date(self, date, n_strikes=None, strike_bounds=None, extra_cols=None):
        """
        Returns the dataframe with the complete SPX surface
        :param date: trading date, can be datetime or integer with format yyyymmdd
        :param n_strikes: maximum number of strikes considered
        :param strike_bounds: tuple with (minimum strike, maximum strike) with respect to spot
        :param extra_cols: extra columns wrt. the DEFAULT_COLS to be added
        :return: pandas dataframe with surface
        """
        if isinstance(extra_cols, list):
            cols = list(set(self.DEFAULT_COLS + extra_cols))
        if isinstance(extra_cols, str):
            cols = list(set(self.DEFAULT_COLS + [extra_cols]))
        if extra_cols is None:
            cols = self.DEFAULT_COLS
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")

        surface = self.data.loc[self.data['date'] == date].reset_index(drop=True)
        # Print line if nothing is found
        if len(surface) == 0:
            print('No data available for ' + str(date))
            return None

        if (n_strikes is not None) and (strike_bounds is not None):
            surface = thin_surface(surface, n_strikes, strike_bounds)

        return surface[cols]

    def smile_days_range(self, date, day_min, day_max, n_strikes=None, strike_bounds=None, extra_cols=None):
        """
        Returns dataframe with the SPX smile for given date where the number of days to expiry is between
        a minumum and maximum. The smile with the most strikes available is chosen if multiple expiries are available
        :param date: trading date, can be datetime or integer with format yyyymmdd
        :param day_min: minimum days to expiry
        :param day_max: maximum days to expiry
        :return: pandas dataframe with the smile
        """
        if isinstance(extra_cols, list):
            cols = list(set(self.DEFAULT_COLS + extra_cols))
        if isinstance(extra_cols, str):
            cols = list(set(self.DEFAULT_COLS + [extra_cols]))
        if extra_cols is None:
            cols = self.DEFAULT_COLS
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")

        surface = self.surface_date(date, extra_cols=cols)
        filtered = surface[(surface['days'] > day_min) & (surface['days'] < day_max)]

        if len(filtered) == 0:
            print('No data for this range')
            return filtered

        # Find the date with the most strikes available and select it
        count = pd.pivot_table(filtered, index=['date', 'days'], values='exdate', aggfunc='count').reset_index()
        day_select = count.loc[
            count['exdate'] == count['exdate'].max(), 'days']  # select day with maximum number of strikes
        day_select = day_select.iloc[0]

        smile = surface[surface['days'] == day_select].sort_values('strike_price')

        if strike_bounds is not None:
            min_strike = strike_bounds[0]
            max_strike = strike_bounds[1]
            smile = smile.loc[(smile['strike'] >= min_strike) & (smile['strike'] <= max_strike)]

        # Perform thinning on the dataframe
        if n_strikes is not None:
            smile = smile.reset_index(drop=True)
            n = len(smile)  # number of strikes
            arr = np.arange(n)
            idx = np.round(np.linspace(0, len(arr) - 1, n_strikes)).astype(int)
            smile = smile.loc[idx]
            smile = smile.drop_duplicates()

        return smile[cols].reset_index(drop=True)

    def smile_days(self, date, days, n_strikes=None, strike_bounds=None, extra_cols=None):
        """
        Returns dataframe with the SPX smile for given trading date with fixed number of days to maturity
        :param date: trading date, can be datetime or integer with format yyyymmdd
        :param days: number of days to maturity
        :return: pandas dataframe with smile
        """
        if isinstance(extra_cols, list):
            cols = list(set(self.DEFAULT_COLS + extra_cols))
        if isinstance(extra_cols, str):
            cols = list(set(self.DEFAULT_COLS + [extra_cols]))
        if extra_cols is None:
            cols = self.DEFAULT_COLS
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")

        surface = self.surface_date(date, extra_cols=cols)
        days_list = surface['days'].unique()
        day_select = min(days_list, key=lambda x: abs(x - days))

        smile = surface[surface['days'] == day_select].sort_values('strike_price')

        if strike_bounds is not None:
            min_strike = strike_bounds[0]
            max_strike = strike_bounds[1]
            smile = smile.loc[(smile['strike'] >= min_strike) & (smile['strike'] <= max_strike)]

        # Perform thinning on the dataframe
        if n_strikes is not None:
            smile = smile.reset_index(drop=True)
            n = len(smile)  # number of strikes
            arr = np.arange(n)
            idx = np.round(np.linspace(0, len(arr) - 1, n_strikes)).astype(int)
            smile = smile.loc[idx]
            smile = smile.drop_duplicates()

        return smile.reset_index(drop=True)

    def smile_date(self, date, exdate, n_strikes=None, strike_bounds=None, extra_cols=None):
        """
        Returns dataframe with the SPX smile for given trading date with a fixed expiration date
        :param date: trading date, can be datetime or integer with format yyyymmdd
        :param days: expiration date, can be datetime or integer with format yyyymmdd
        :return: pandas dataframe with smile
        """
        if isinstance(extra_cols, list):
            cols = list(set(self.DEFAULT_COLS + extra_cols))
        if isinstance(extra_cols, str):
            cols = list(set(self.DEFAULT_COLS + [extra_cols]))
        if extra_cols is None:
            cols = self.DEFAULT_COLS
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")
        if isinstance(exdate, int):
            exdate = datetime.strptime(str(exdate), "%Y%m%d")
        surface = self.surface_date(date)
        smile = surface[surface['exdate'] == exdate].sort_values('strike_price').reset_index(drop=True)
        if strike_bounds is not None:
            min_strike = strike_bounds[0]
            max_strike = strike_bounds[1]
            smile = smile.loc[(smile['strike'] >= min_strike) & (smile['strike'] <= max_strike)].reset_index(drop=True)

        # Perform thinning on the dataframe
        if n_strikes is not None:
            smile = smile.reset_index(drop=True)
            n = len(smile)  # number of strikes
            arr = np.arange(n)
            idx = np.round(np.linspace(0, len(arr) - 1, n_strikes)).astype(int)
            smile = smile.loc[idx]
            smile = smile.drop_duplicates()

        return smile[cols].reset_index(drop=True)


class MatlabMonteCarlo:

    def __init__(self):
        path = Path(os.path.abspath('')) / 'matlab_code'
        print(path)
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(str(path), nargout=0)

    def simulateSABRSmile(self, no_of_sim, no_of_steps, T, r, F_0, alpha, beta, rho, v, strike, seed=None, dx=1e-3,
                          smoothing=True):
        """
        Simulates a SABR IV smile using the Matlab Python engine. The Euler discretization scheme is used
        :param no_of_sim: number of simulations (float)
        :param no_of_steps: number of timesteps in each simulation (float)
        :param T: time to expiry (float)
        :param r: interest rate (always set to 0) (float)
        :param F_0: initial forward (float)
        :param alpha: SABR alpha (float)
        :param beta: SABR beta (float)
        :param rho: SABR rho (float)
        :param v: SABR volatility of volatility (float)
        :param strike: tuple for strike range, (minimum strike, maximum strike, size of step between strikes)
        :param seed: random seed
        :param dx: step size in central differences for greeks
        :param smoothing: smoothing of the smile (remove nan IVs and make smile convex).
        :return: pandas DataFrame with smile and the greeks.
        """
        if seed is None:
            seed = random.randint(2, 999999)

        # create numpy array for the strike range
        strike_min = strike[0]
        strike_max = strike[1]
        strike_step = strike[2]
        strikes = np.round(np.arange(strike_min, strike_max + strike_step, strike_step), 3)

        # perform the actual Monte Carlo simulation in Matlab
        iv, F, sigma = self.eng.simulateEulerIV(float(no_of_sim), float(no_of_steps), T, r, F_0, alpha, beta, rho, v,
                                                strike_min, strike_max, strike_step, seed, nargout=3)

        output = pd.DataFrame(
            columns=['T', 'r', 'strike', 'strike_price', 'forward_price', 'impl_volatility', 'sigma', 'deriv_alpha',
                     'deriv_F'])
        output['strike'] = strikes
        output['T'] = T
        output['r'] = r
        output['strike_price'] = strikes
        output['forward_price'] = F_0
        output['impl_volatility'] = iv[0]
        output['sigma'] = sigma[0]

        dim = len(strikes)  # used in calculation of derivative for each strike

        # Add the Monte Carlo partial derivative of implied vol to alpha
        f_deriv_alpha = lambda x: np.asarray(
            self.eng.simulateEulerIV(float(no_of_sim), float(no_of_steps), T, r, F_0, x, beta, rho, v, strike_min,
                                     strike_max, strike_step, seed)[0])
        deriv_alpha = hf.derivative(f_deriv_alpha, alpha, dim, dx=dx)
        output['deriv_alpha'] = deriv_alpha

        # Add the Monte Carlo partial derivative of implied vol to forward
        f_deriv_F = lambda x: np.asarray(
            self.eng.simulateEulerIV(float(no_of_sim), float(no_of_steps), T, r, x, alpha, beta, rho, v, strike_min,
                                     strike_max, strike_step, seed)[0])
        deriv_F = hf.derivative(f_deriv_F, F_0, dim, dx=dx)
        output['deriv_F'] = deriv_F

        # Compute the greeks
        output['delta_bs'] = output[['strike', 'impl_volatility']].apply(
            lambda x: hf.calc_black_scholes_delta(x[1], T, (x[0], F_0)), axis=1)
        output['vega_bs'] = output[['strike', 'impl_volatility']].apply(
            lambda x: hf.calc_black_scholes_vega(x[1], T, (x[0], F_0)), axis=1)

        output['vega'] = output['vega_bs'] * output['deriv_alpha']
        output['delta'] = output['delta_bs'] + output['vega_bs'] * output['deriv_F']

        # only keep non-zero implied vols and a convex smile
        if smoothing:
            output = hf.make_smile_smooth(output)
        return output
