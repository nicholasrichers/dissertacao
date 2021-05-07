# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module contains a class for ETF trick generation and futures roll function, described in Marcos Lopez de Prado's
book 'Advances in Financial Machine Learning' ETF trick class can generate ETF trick series either from .csv files
or from in-memory pandas DataFrames.
"""

import pandas as pd

from mlfinlab.util import devadarsh

class ETFTrick:
    """
    Contains logic of vectorised ETF trick implementation.
    """

    def __init__(self):
        """
        ETF Trick class constructor.
        """

        devadarsh.track('ETFTrick')

        self.prev_k = 1.0  # Init with $1 as initial value
        self.etf_trick_series = pd.Series(dtype=float)

        # We need to track allocations vector change on previous step
        # Previous allocation change is needed for delta component calculation
        self.prev_allocs_change = False
        self.prev_h = None  # To find current etf_trick value we need previous h value
        self.roll_flag = False

        self.prev_index_data = {k: None for k in ['open', 'close', 'alloc', 'costs', 'rates']}

    def get_etf_series(self, open_df: pd.DataFrame, close_df: pd.DataFrame, alloc_df: pd.DataFrame,
                       costs_df: pd.DataFrame = None, rates_df: pd.DataFrame = None) -> pd.Series:
        """
        Get ETF trick series.

        :param open_df: (pd.DataFrame) Open prices data frame corresponds to o(t) from the book.
        :param close_df: (pd.DataFrame) Close prices data frame or path to csv file, corresponds to p(t).
        :param alloc_df: (pd.DataFrame) Asset allocations data frame or path to csv file (in # of contracts),
            corresponds to w(t).
        :param costs_df: (pd.DataFrame) Rebalance, carry and dividend costs of holding/rebalancing the
            position, corresponds to d(t).
        :param rates_df: (pd.DataFrame) Dollar value of one point move of contract includes exchange rate,
            futures contracts multiplies). Corresponds to phi(t).
            For example, 1$ in VIX index, equals 1000$ in VIX futures contract value.
            If None then trivial (all values equal 1.0) is generated.
        :return: (pd.Series) Pandas Series with ETF trick values starting from 1.0.
        """

        alloc_df.fillna(0, inplace=True)
        start_index = 0

        if rates_df is None:
            rates_df = open_df.copy()
            # Set trivial(1.0) exchange rate if no data is provided
            rates_df[rates_df.columns] = 1.0

        if costs_df is None:
            costs_df = open_df.copy()
            # Set zero costs
            costs_df[costs_df.columns] = 0.0

        # The first step is trivial
        if self.etf_trick_series.shape[0] == 0:
            start_index = 1
            self.prev_index_data['open'] = open_df.iloc[0]
            self.prev_index_data['close'] = close_df.iloc[0]
            self.prev_index_data['alloc'] = alloc_df.iloc[0]
            self.prev_index_data['сosts'] = costs_df.iloc[0]
            self.prev_index_data['rates'] = rates_df.iloc[0]

            self.etf_trick_series.loc[open_df.index[0]] = 1
            self.roll_flag = True

        for i in range(start_index, close_df.shape[0]):
            current_index = close_df.index[i]
            current_open = open_df.loc[current_index]
            current_close = close_df.loc[current_index]

            # Get number of units in portfolio (h_i)
            if self.roll_flag is True:
                w_prev = self.prev_index_data['alloc']
                prev_rates = self.prev_index_data['rates']
                w_prev_abs = w_prev.abs().sum()
                num_of_holdings = w_prev * self.prev_k / (current_open * prev_rates * w_prev_abs)  # h_t from the book
                self.prev_h = num_of_holdings

                delta = current_close - current_open
            else:
                delta = current_close - self.prev_index_data['close']

            k = self.prev_k + (self.prev_h * rates_df.loc[current_index] * (delta + costs_df.loc[current_index])).sum()
            self.etf_trick_series.loc[current_index] = k

            self.roll_flag = bool(
                ~(self.prev_index_data['alloc'] == alloc_df.loc[current_index]).all())

            self.prev_index_data['open'] = open_df.loc[current_index]
            self.prev_index_data['close'] = close_df.loc[current_index]
            self.prev_index_data['alloc'] = alloc_df.loc[current_index]
            self.prev_index_data['сosts'] = costs_df.loc[current_index]
            self.prev_index_data['rates'] = rates_df.loc[current_index]

            self.prev_k = k
        return self.etf_trick_series


def get_futures_roll_series(data_df: pd.DataFrame, open_col: str, close_col: str, sec_col: str, current_sec_col: str,
                            roll_backward: bool = False, method: str = 'absolute') -> pd.Series:
    """
    Function for generating rolling futures series from data frame of multiple futures.

    :param data_df: (pd.DataFrame) Pandas DataFrame containing price info, security name and current active futures
        column.
    :param open_col: (str) Open prices column name.
    :param close_col: (str) Close prices column name.
    :param sec_col: (str) Security name column name.
    :param current_sec_col: (str) Current active security column name. When value in this column changes it means
        rolling.
    :param roll_backward: (bool) True for subtracting final gap value from all values.
    :param method: (str) What returns user wants to preserve, 'absolute' or 'relative'.
    :return (pd.Series) Futures roll adjustment factor series.
    """

    devadarsh.track('get_futures_roll_series')

    # Filter out security data which is not used as current security
    filtered_df = data_df[data_df[sec_col] == data_df[current_sec_col]]
    filtered_df.sort_index(inplace=True)

    # Generate roll dates series based on current_sec column value change
    roll_dates = filtered_df[current_sec_col].drop_duplicates(keep='first').index
    timestamps = list(filtered_df.index)  # List of timestamps
    prev_roll_dates_index = [timestamps.index(i) - 1 for i in roll_dates]  # Dates before rolling date index (int)

    # On roll dates, gap equals open - close or open/close
    if method == 'absolute':
        gaps = filtered_df[close_col] * 0  # roll gaps series
        gaps.loc[roll_dates[1:]] = filtered_df[open_col].loc[roll_dates[1:]] - filtered_df[close_col].iloc[
            prev_roll_dates_index[1:]].values
        gaps = gaps.cumsum()

        if roll_backward:
            gaps -= gaps.iloc[-1]  # Roll backward diff
    elif method == 'relative':
        gaps = filtered_df[close_col] * 0 + 1  # Roll gaps series
        gaps.loc[roll_dates[1:]] = filtered_df[open_col].loc[roll_dates[1:]] / filtered_df[close_col].iloc[
            prev_roll_dates_index[1:]].values
        gaps = gaps.cumprod()

        if roll_backward:
            gaps /= gaps.iloc[-1]  # Roll backward div
    else:
        raise ValueError('The method must be either "absolute" or "relative", please check spelling.')

    return gaps
