# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module contains the code snippets found in Chapter 10 of "Advances in Financial Machine Learning" by
Marcos López de Prado. The code has been amended for readability, to conform to PEP8 rules, to keep the snippets as
manageable single-units of functionality, as well as to account for deprecation of functions originally used, but is
otherwise unaltered.
"""

import warnings

import pandas as pd
import numpy as np
from scipy.stats import norm

from mlfinlab.util import devadarsh
from mlfinlab.util.multiprocess import mp_pandas_obj


def get_signal(prob, num_classes, pred=None):
    """
    SNIPPET 10.1 - FROM PROBABILITIES TO BET SIZE
    Calculates the given size of the bet given the side and the probability (i.e. confidence) of the prediction. In this
    representation, the probability will always be between 1/num_classes and 1.0.

    :param prob: (pd.Series) The probability of the predicted bet side.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
        (i.e. without multiplying by the side).
    :return: (pd.Series) The bet size.
    """

    devadarsh.track('get_signal')

    # Get signals from predictions
    if prob.shape[0] == 0:
        return pd.Series(dtype='float64')

    # 1) Generate signals from multinomial classification (one-vs-rest)
    bet_sizes = (prob - 1/num_classes) / (prob * (1 - prob))**0.5

    # Allow for bet size to be returned with or without side
    if not isinstance(pred, type(None)):
        # signal = side * size
        bet_sizes = pred * (2 * norm.cdf(bet_sizes) - 1)
    else:
        # signal = size only
        bet_sizes = bet_sizes.apply(lambda s: 2 * norm.cdf(s) - 1)

    # Note 1: In the book, this function contains a conditional statement checking for a column named 'side',
    # then executes what is essentially the above line. This has been removed as it appears to be redundant
    # and simplifies the function

    # Note 2: In the book, this function includes the averaging and discretization steps, which are omitted here.
    # The functions for performing these are included in this file, and can be applied as options in the user-level
    # functions in bet_sizing.py

    return bet_sizes


def avg_active_signals(signals, num_threads=1):
    """
    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
    Function averages the bet sizes of all concurrently active bets. This function makes use of multiprocessing.

    :param signals: (pd.DataFrame) Contains at least the following columns:
        'signal' - the bet size
        't1' - the closing time of the bet
        And the index must be datetime format.
    :param num_threads: (int) Number of threads to use in multiprocessing, default value is 1.
    :return: (pd.Series) The averaged bet sizes.
    """

    devadarsh.track('avg_active_signals')

    # 1) Time points where signals change (either one start or one ends)
    t_pnts = set(signals['t1'].dropna().to_numpy())
    t_pnts = t_pnts.union(signals.index.to_numpy())
    t_pnts = list(t_pnts)
    t_pnts.sort()
    out = mp_pandas_obj(mp_avg_active_signals, ('molecule', t_pnts), num_threads, signals=signals)

    return out


def mp_avg_active_signals(signals, molecule):
    """
    Part of SNIPPET 10.2
    A function to be passed to the 'mp_pandas_obj' function to allow the bet sizes to be averaged using multiprocessing.

    At time loc, average signal among those still active.
    Signal is active if (a) it is issued before or at loc, and (b) loc is before the signal's end time,
    or end time is still unknown (NaT).

    :param signals: (pd.DataFrame) Contains at least the following columns: 'signal' (the bet size)
        and 't1' (the closing time of the bet).
    :param molecule: (list) Indivisible tasks to be passed to 'mp_pandas_obj', in this case a list of datetimes.
    :return: (pd.Series) The averaged bet size sub-series.
    """

    devadarsh.track('mp_avg_active_signals')

    out = pd.Series(dtype='float64')
    for loc in molecule:
        df0 = (signals.index.to_numpy() <= loc)&((loc < signals['t1'])|pd.isnull(signals['t1']))
        act = signals[df0].index
        if act.size > 0:
            # Average active signals if they exist
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            # Return zero if no signals are active at this time step
            out[loc] = 0

    return out


def discrete_signal(signal0, step_size):
    """
    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING
    Discretizes the bet size signal based on the step size given.

    :param signal0: (pd.Series) The signal to discretize.
    :param step_size: (float) Step size.
    :return: (pd.Series) The discretized signal.
    """

    devadarsh.track('discrete_signal')

    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1  # Cap
    signal1[signal1 < -1] = -1  # Floor

    return signal1


def bet_size_sigmoid(w_param, price_div):
    """
    Part of SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a sigmoid function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, forecast price - market price.
    :return: (float) The bet size.
    """

    devadarsh.track('bet_size_sigmoid')

    return price_div * ((w_param + price_div**2)**(-0.5))


def get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a sigmoid function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :return: (int) Target position.
    """

    devadarsh.track('get_target_pos_sigmoid')

    return int(bet_size_sigmoid(w_param, forecast_price-market_price) * max_pos)


def inv_price_sigmoid(forecast_price, w_param, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a sigmoid function for a bet size algorithm.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """

    devadarsh.track('inv_price_sigmoid')

    return forecast_price - m_bet_size * (w_param / (1 - m_bet_size**2))**0.5


def limit_price_sigmoid(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Part of SNIPPET 10.4
    Calculates the limit price.
    Based on a sigmoid function for a bet size algorithm.

    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :return: (float) Limit price.
    """

    devadarsh.track('limit_price_sigmoid')

    if target_pos == pos:
        # Return NaN if the current and target positions are the same to avoid divide-by-zero error.
        return np.nan

    sgn = np.sign(target_pos-pos)
    l_p = 0
    for j in range(abs(pos+sgn), abs(target_pos+1)):
        l_p += inv_price_sigmoid(forecast_price, w_param, j/float(max_pos))

    l_p = l_p / abs(target_pos-pos)

    return l_p


def get_w_sigmoid(price_div, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    Based on a sigmoid function for a bet size algorithm.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """

    devadarsh.track('get_w_sigmoid')

    return (price_div**2) * ((m_bet_size**(-2)) - 1)


def bet_size_power(w_param, price_div):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a power function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price, must be between -1 and 1, inclusive.
    :return: (float) The bet size.
    """

    devadarsh.track('bet_size_power')

    if not (-1 <= price_div <= 1):
        raise ValueError(f"Price divergence must be between -1 and 1, inclusive. Found price divergence value:"
                         f" {price_div}")
    if price_div == 0.0:
        return 0.0

    return np.sign(price_div) * abs(price_div)**w_param


def get_target_pos_power(w_param, forecast_price, market_price, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a power function for a bet size algorithm.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Target position.
    """

    devadarsh.track('get_target_pos_power')

    return int(bet_size_power(w_param, forecast_price-market_price) * max_pos)


def inv_price_power(forecast_price, w_param, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a power function for a bet size algorithm.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """

    devadarsh.track('inv_price_power')

    if m_bet_size == 0.0:
        return forecast_price

    return forecast_price - np.sign(m_bet_size) * abs(m_bet_size)**(1/w_param)


def limit_price_power(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. Based on a power function for a bet size algorithm.

    :param target_pos: (float) Target position.
    :param pos: (float) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Limit price.
    """

    devadarsh.track('limit_price_power')

    sgn = np.sign(target_pos-pos)
    l_p = 0
    for j in range(abs(pos+sgn), abs(target_pos+1)):
        l_p += inv_price_power(forecast_price, w_param, j/float(max_pos))

    l_p = l_p / abs(target_pos-pos)

    return l_p


def get_w_power(price_div, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'w' coefficient must be greater than or equal to zero.
    Based on a power function for a bet size algorithm.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """

    devadarsh.track('get_w_power')

    if not -1 <= price_div <= 1:
        raise ValueError("Price divergence argument 'x' must be between -1 and 1,"
                         " inclusive when using function 'power'.")

    w_calc = np.log(m_bet_size/np.sign(price_div)) / np.log(abs(price_div))
    if w_calc < 0:
        warnings.warn("'w' parameter evaluates to less than zero. Zero is returned.", UserWarning)

    return max(0, w_calc)


def bet_size(w_param, price_div, func):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) The bet size.
    """

    devadarsh.track('bet_size')

    return {'sigmoid': bet_size_sigmoid,
            'power': bet_size_power}[func](w_param, price_div)


def get_target_pos(w_param, forecast_price, market_price, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. The 'func' argument allows the user to choose between bet sizing functions.

    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (int) Target position.
    """

    devadarsh.track('get_target_pos')

    return {'sigmoid': get_target_pos_sigmoid,
            'power': get_target_pos_power}[func](w_param, forecast_price, market_price, max_pos)


def inv_price(forecast_price, w_param, m_bet_size, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """

    devadarsh.track('inv_price')

    return {'sigmoid': inv_price_sigmoid,
            'power': inv_price_power}[func](forecast_price, w_param, m_bet_size)


def limit_price(target_pos, pos, forecast_price, w_param, max_pos, func):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. The 'func' argument allows the user to choose between bet sizing functions.

    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) Limit price.
    """

    devadarsh.track('limit_price')

    return {'sigmoid': limit_price_sigmoid,
            'power': limit_price_power}[func](int(target_pos), int(pos), forecast_price, w_param, max_pos)


def get_w(price_div, m_bet_size, func):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'func' argument allows the user to choose between bet sizing functions.

    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """

    devadarsh.track('get_w')

    return {'sigmoid': get_w_sigmoid,
            'power': get_w_power}[func](price_div, m_bet_size)
