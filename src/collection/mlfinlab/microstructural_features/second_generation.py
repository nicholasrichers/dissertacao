# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Second generation models features: Kyle lambda, Amihud Lambda, Hasbrouck lambda (bar and trade based).
"""
# pylint: disable=invalid-name

from typing import List
import numpy as np
import pandas as pd

from mlfinlab.structural_breaks.sadf import get_betas
from mlfinlab.util import devadarsh


def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, aggressor_flags: pd.Series = None,
                              window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 286-288.

    Get Kyle lambda from bars data.

    :param close: (pd.Series) Close prices.
    :param volume: (pd.Series) Bar volume.
    :param aggressor_flags: (pd.Series) Series of indicators {-1, 1} if a bar was buy(1) or sell (-1). If None, sign
        of price differences is used.
    :param window: (int) Rolling window used for estimation.
    :return: (pd.Series) Kyle lambdas.
    """

    devadarsh.track('get_bar_based_kyle_lambda')

    lambda_df = pd.DataFrame(index=close.index, columns=['close_diff', 'agg_flags', 'signed_volume', 'kyle_lambda',
                                                         'kyle_lambda_t_value'])
    close_diff = close.diff()
    close_diff_sign = close_diff.apply(np.sign)
    close_diff_sign.replace(0, method='pad', inplace=True)  # Replace 0 values with previous
    if aggressor_flags is None:
        aggressor_flags = np.sign(close_diff)
    signed_volume = np.array(volume) * np.array(aggressor_flags)

    lambda_df['close_diff'] = close_diff
    lambda_df['agg_flags'] = aggressor_flags
    lambda_df['signed_volume'] = signed_volume

    results = []
    for i in range(window, lambda_df.shape[0]):
        # i+1 because we need to take current bar, but iloc does not include right side.
        subset = lambda_df.iloc[i - window + 1: i + 1]
        # Perform regression
        X = np.array(subset['signed_volume']).reshape(-1, 1)
        y = np.array(subset['close_diff'])
        coef, var = get_betas(X, y)
        t_value = coef[0] / np.sqrt(var[0]) if var[0] > 0 else np.array([0])
        results.append({'index': lambda_df.index[i], 'kyle_lambda': coef[0], 'kyle_lambda_t_value': t_value[0]})
    results_df = pd.DataFrame.from_records(results)
    results_df.set_index('index', inplace=True)

    return results_df


def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from bars data.

    :param close: (pd.Series) Close prices.
    :param dollar_volume: (pd.Series) Dollar volumes.
    :param window: (int) rolling window used for estimation.
    :return: (pd.Series) of Amihud lambda.
    """

    devadarsh.track('get_bar_based_amihud_lambda')

    returns_abs = np.log(close / close.shift(1)).abs()
    lambda_df = pd.DataFrame(index=close.index, columns=['return_abs', 'dollar_volume', 'amihud_lambda',
                                                         'amihud_lambda_t_value'])

    lambda_df['return_abs'] = returns_abs
    lambda_df['dollar_volume'] = dollar_volume

    results = []
    for i in range(window, lambda_df.shape[0]):
        # i+1 because we need to take current bar, but iloc does not include right side
        subset = lambda_df.iloc[i - window + 1: i + 1]
        # Perform regression
        X = np.array(subset['dollar_volume']).reshape(-1, 1)
        y = np.array(subset['return_abs'])
        coef, var = get_betas(X, y)
        t_value = coef[0] / np.sqrt(var[0]) if var[0] > 0 else np.array([0])
        results.append({'index': lambda_df.index[i], 'amihud_lambda': coef[0], 'amihud_lambda_t_value': t_value[0]})

    results_df = pd.DataFrame.from_records(results)
    results_df.set_index('index', inplace=True)

    return results_df


def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, aggressor_flags: pd.Series = None,
                                   window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from bars data.

    :param close: (pd.Series) Close prices.
    :param dollar_volume: (pd.Series) Dollar volumes.
    :param aggressor_flags: (pd.Series) Series of indicators {-1, 1} if a bar was buy(1) or sell (-1). If None, sign
                            of price differences is used.
    :param window: (int) Rolling window used for estimation.
    :return: (pd.Series) Hasbrouck lambdas series.
    """

    devadarsh.track('get_bar_based_hasbrouck_lambda')

    lambda_df = pd.DataFrame(index=close.index, columns=['log_ret', 'signed_dollar_volume_sqrt', 'hasbrouck_lambda',
                                                         'hasbrouck_lambda_t_value'])

    log_ret = np.log(close / close.shift(1))
    log_ret_sign = log_ret.apply(np.sign).replace(0, method='pad')
    if aggressor_flags is None:
        aggressor_flags = log_ret_sign
    signed_dollar_volume_sqrt = aggressor_flags * np.sqrt(dollar_volume)
    lambda_df['log_ret'] = log_ret
    lambda_df['signed_dollar_volume_sqrt'] = signed_dollar_volume_sqrt

    results = []
    for i in range(window, lambda_df.shape[0]):
        # i+1 because we need to take current bar, but iloc does not include right side.
        subset = lambda_df.iloc[i - window + 1: i + 1]
        # Perform regression
        X = np.array(subset['signed_dollar_volume_sqrt']).reshape(-1, 1)
        y = np.array(subset['log_ret'])
        coef, var = get_betas(X, y)
        t_value = coef[0] / np.sqrt(var[0]) if var[0] > 0 else np.array([0])
        results.append(
            {'index': lambda_df.index[i], 'hasbrouck_lambda': coef[0], 'hasbrouck_lambda_t_value': t_value[0]})
    results_df = pd.DataFrame.from_records(results)
    results_df.set_index('index', inplace=True)

    return results_df


def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.286-288.

    Get Kyle lambda from trades data.

    :param price_diff: (list) Price diffs.
    :param volume: (list) Trades sizes.
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define).
    :return: (list) Kyle lambda for a bar and t-value.
    """

    devadarsh.track('get_trades_based_kyle_lambda')

    signed_volume = np.array(volume) * np.array(aggressor_flags)
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    coef, var = get_betas(X, y)
    t_value = coef[0] / np.sqrt(var[0]) if var[0] > 0 else np.array([0])

    return [coef[0], t_value[0]]


def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from trades data.

    :param log_ret: (list) Log returns.
    :param dollar_volume: (list) Dollar volumes (price * size).
    :return: (float) Amihud lambda for a bar.
    """

    devadarsh.track('get_trades_based_amihud_lambda')

    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    coef, var = get_betas(X, y)
    t_value = coef[0] / np.sqrt(var[0]) if var[0] > 0 else np.array([0])

    return [coef[0], t_value[0]]


def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from trades data.

    :param log_ret: (list) Log returns.
    :param dollar_volume: (list) Dollar volumes (price * size).
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define).
    :return: (list) Hasbrouck lambda for a bar and t value.
    """

    devadarsh.track('get_trades_based_hasbrouck_lambda')

    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    coef, var = get_betas(X, y)
    t_value = coef[0] / np.sqrt(var[0]) if var[0] > 0 else np.array([0])

    return [coef[0], t_value[0]]
