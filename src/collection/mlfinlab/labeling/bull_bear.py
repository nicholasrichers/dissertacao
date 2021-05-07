# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Detection of bull and bear markets.
"""
import numpy as np
import pandas as pd


def pagan_sossounov(prices, window=8, censor=6, cycle=16, phase=4, threshold=0.2):
    """
    Pagan and Sossounov's labeling method. Sourced from `Pagan, Adrian R., and Kirill A. Sossounov. "A simple framework
    for analysing bull and bear markets." Journal of applied econometrics 18.1 (2003): 23-46.
    <https://onlinelibrary.wiley.com/doi/pdf/10.1002/jae.664>`__

    Returns a DataFrame with labels of 1 for Bull and -1 for Bear.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market.
    :param window: (int) Rolling window length to determine local extrema. Paper suggests 8 months for monthly obs.
    :param censor: (int) Number of months to eliminate for start and end. Paper suggests 6 months for monthly obs.
    :param cycle: (int) Minimum length for a complete cycle. Paper suggests 16 months for monthly obs.
    :param phase: (int) Minimum length for a phase. Paper suggests 4 months for monthly obs.
    :param threshold: (double) Minimum threshold for phase change. Paper suggests 0.2.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    return prices.apply(_apply_pagan_sossounov, axis=0, args=(window, censor, cycle, phase, threshold))


def _alternation(price):
    """
    Helper function to check peak and trough alternation.

    :param price: (pd.DataFrame) Close prices of all tickers in the market.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    # Unpack first row values
    first_row = price.iloc[0]
    best_val, curr_label, best_idx = first_row[0], first_row[1], first_row.name
    # Holder for filtered index values
    true_idx = []

    # Iterate through all rows
    for row in price.tail(-1).itertuples(name=None):
        # Changing labels
        if row[2] != curr_label:
            true_idx.append(best_idx)
            best_val = row[1]
            best_idx = row[0]
            curr_label *= -1
        # Constant labels
        elif (curr_label > 0 and row[1] > best_val) or (curr_label < 0 and row[1] < best_val):
            best_idx = row[0]
            best_val = row[1]

    return price.loc[true_idx]


def _apply_pagan_sossounov(price, window, censor, cycle, phase, threshold):
    """
    Helper function for Pagan and Sossounov labeling method.

    :param price: (pd.DataFrame) Close prices of all tickers in the market.
    :param window: (int) Rolling window length to determine local extrema. Paper suggests 8 months for monthly obs.
    :param censor: (int) Number of months to eliminate for start and end. Paper suggests 6 months for monthly obs.
    :param cycle: (int) Minimum length for a complete cycle. Paper suggests 16 months for monthly obs.
    :param phase: (int) Minimum length for a phase. Paper suggests 4 months for monthly obs.
    :param threshold: (double) Minimum threshold for phase change. Paper suggests 20%.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    # Set initial values
    df_idx = price.index
    adj_window = window * 2 + 1

    # Create adjacent dataframe
    df_price = pd.DataFrame(price)
    df_price['label'] = 0

    # Label local extrema
    df_price.loc[price.rolling(adj_window).max() == price, 'label'] = 1
    df_price.loc[price.rolling(adj_window).min() == price, 'label'] = -1

    # Remove censored positions
    df_price.iloc[:censor, 1] = 0
    df_price.iloc[-censor:, 1] = 0

    # Filter out unlabeled points
    df_price.index = np.arange((len(df_idx)))
    df_price['time'] = df_price.index
    df_price = df_price[df_price['label'] != 0]
    df_price = _alternation(df_price)

    # Filter out minimum cycle
    df_price['t_diff'] = df_price['time'].diff(2).fillna(cycle + 1)
    df_price = df_price[df_price['t_diff'] > cycle]
    df_price = _alternation(df_price)

    # Filter our minimum phase
    df_price['t_diff'] = df_price['time'].diff(1).fillna(phase + 1)
    df_price['p_diff'] = np.abs(df_price.iloc[:, 0].pct_change(periods=-1))
    df_price = df_price[(df_price['p_diff'] > threshold) | (df_price['t_diff'] > phase)].drop(columns=['t_diff', 'p_diff'])
    df_price = _alternation(df_price)

    # Return corresponding labels
    df_price.index = df_idx[df_price.index]
    price[:] = np.nan
    price.loc[df_price.index] = df_price['label']
    price.fillna(method='bfill', inplace=True)
    price.fillna(df_price.tail(1)['label'].values[0] * -1, inplace=True)

    return price


def lunde_timmermann(prices, bull_threshold=0.15, bear_threshold=0.15):
    """
    Lunde and Timmermann's labeling method. Sourced from `Lunde, Asger, and Allan Timmermann. "Duration dependence
    in stock prices: An analysis of bull and bear markets." Journal of Business & Economic Statistics 22.3 (2004): 253-273.
    <https://repec.cepr.org/repec/cpr/ceprdp/DP4104.pdf>`__

    Returns a DataFrame with labels of 1 for Bull and -1 for Bear.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market.
    :param bull_threshold: (double) Threshold to identify bull market. Paper suggests 0.15.
    :param bear_threshold: (double) Threshold to identify bear market. Paper suggests 0.15.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    return prices.apply(_apply_lunde_timmermann, axis=0, args=(bull_threshold, bear_threshold))


def _apply_lunde_timmermann(price, bull_threshold, bear_threshold):
    """
    Helper function for Lunde and Timmermann labeling method.

    :param price: (pd.DataFrame) Close prices of all tickers in the market.
    :param bull_threshold: (double) Threshold to identify bull market. Paper suggests 0.15.
    :param bear_threshold: (double) Threshold to identify bear market. Paper suggests 0.15.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    # Initialize variables
    res = np.zeros(price.shape)
    idx = 0
    state = 0
    high_price = low_price = price.head(1).item()

    # Iterate through rows
    for _, val in price.tail(-1).iteritems():
        idx += 1
        # Bull state
        if state == 1:
            if val > high_price:
                high_price = val
            elif val <= (1 - bear_threshold) * high_price:
                low_price = val
                state = -1
            res[idx] = state
        # Bear state
        elif state == -1:
            if val < low_price:
                low_price = val
            elif val >= (1 + bull_threshold) * low_price:
                high_price = val
                state = 1
            res[idx] = state
        # Edge case for identifying beginning state
        else:
            if val > high_price:
                high_price = val
            elif val < low_price:
                low_price = val
            elif val <= (1 - bear_threshold) * high_price:
                state = -1
                res[:idx + 1] = state
            elif val >= (1 + bull_threshold) * low_price:
                state = 1
                res[:idx + 1] = state
    price[:] = res

    return price
