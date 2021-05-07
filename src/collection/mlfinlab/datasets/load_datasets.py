# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
The module implementing various functions loading tick, dollar, stock data sets which can be used as
sandbox data.
"""

import os

import numpy as np
import pandas as pd

from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.util.volatility import get_daily_vol
from mlfinlab.filters.filters import cusum_filter
from mlfinlab.util import devadarsh


def load_stock_prices() -> pd.DataFrame:
    """
    Loads stock prices data sets consisting of
    EEM, EWG, TIP, EWJ, EFA, IEF, EWQ, EWU, XLB, XLE, XLF, LQD, XLK, XLU, EPP, FXI, VGK, VPL, SPY, TLT, BND, CSJ,
    DIA starting from 2008 till 2016.

    :return: (pd.DataFrame) The stock_prices data frame.
    """

    devadarsh.track('load_stock_prices')

    project_path = os.path.dirname(__file__)
    prices_df = pd.read_csv(os.path.join(project_path, 'data/stock_prices.csv'), index_col=0, parse_dates=[0])

    return prices_df


def load_tick_sample() -> pd.DataFrame:
    """
    Loads E-Mini S&P 500 futures tick data sample.

    :return: (pd.DataFrame) Frame with tick data sample.
    """

    devadarsh.track('load_tick_sample')

    project_path = os.path.dirname(__file__)
    tick_df = pd.read_csv(os.path.join(project_path, 'data/tick_data.csv'), index_col=0, parse_dates=[0])

    return tick_df


def load_dollar_bar_sample() -> pd.DataFrame:
    """
    Loads E-Mini S&P 500 futures dollar bars data sample.

    :return: (pd.DataFrame) Frame with dollar bar data sample.
    """

    devadarsh.track('load_dollar_bar_sample')

    project_path = os.path.dirname(__file__)
    bars_df = pd.read_csv(os.path.join(project_path, 'data/dollar_bar_sample.csv'), index_col=0, parse_dates=[0])

    return bars_df


def generate_multi_asset_data_set(start_date: pd.Timestamp = pd.Timestamp(2008, 1, 1),
                                  end_date: pd.Timestamp = pd.Timestamp(2020, 1, 1)) -> tuple:
    # pylint: disable=invalid-name
    """
    Generates multi-asset dataset from stock prices labelled by triple-barrier method.

    :param start_date: (pd.Timestamp) Dataset start date.
    :param end_date: (pd.Timestamp) Dataset end date.
    :return: (tuple) Tuple of dictionaries (asset: data) for X, y, cont contract used to label the dataset.
    """

    devadarsh.track('generate_multi_asset_data_set')
    prices_df = load_stock_prices()
    prices_df = prices_df.loc[start_date:end_date]
    tickers_subset = ['SPY', 'XLF', 'EEM', 'TLT', 'XLU']
    prices_df = prices_df[tickers_subset]
    X_dict = {}
    y_dict = {}
    close_prices = {}

    for asset in prices_df.columns:
        # Generate X, y
        daily_vol = get_daily_vol(close=prices_df[asset], lookback=10)
        cusum_events = cusum_filter(prices_df[asset], threshold=0.01)
        vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=prices_df[asset],
                                                 num_days=4)
        labeled_events = get_events(close=prices_df[asset],
                                    t_events=cusum_events,
                                    pt_sl=[1, 4],
                                    target=daily_vol,
                                    min_ret=5e-5,
                                    num_threads=1,
                                    vertical_barrier_times=vertical_barriers,
                                    verbose=False)
        labeled_events.dropna(inplace=True)
        labels = get_bins(labeled_events, prices_df[asset])
        labels['bin'] = np.sign(labels.ret)
        labels = labels[labels.bin.isin([-1, 1])]
        X = pd.DataFrame(index=prices_df[asset].index)

        for window in [5, 10, 20]:
            X['sma_{}'.format(window)] = prices_df[asset] / prices_df[asset].rolling(window=20).mean() - 1
        X.dropna(inplace=True)
        X = X.loc[labels.loc[X.index.min():X.index.max()].index]
        labels = labels.loc[X.index]
        labels['t1'] = labeled_events.loc[labels.index, 't1']

        # Save results
        X_dict[asset] = X.copy()
        y_dict[asset] = labels.copy()
        close_prices[asset] = prices_df[asset].copy()

    return X_dict, y_dict, close_prices
