# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Volume classification methods (BVC and tick rule).
"""
# pylint: disable=invalid-name

import pandas as pd
from scipy.stats import norm
from scipy.stats import t

from mlfinlab.util import devadarsh


def get_bvc_buy_volume(close: pd.Series, volume: pd.Series, window: int = 20, distribution: str = 'norm',
                       df: float = 0.25) -> pd.Series:
    """
    Calculates the BVC buy volume.

    :param close: (pd.Series): Close prices.
    :param volume: (pd.Series): Bar volumes.
    :param window: (int): Window for std estimation uses in BVC calculation.
    :param distribution: (str): Distribution function used to estimate. Either 'norm' or 't_student'.
    :param df: (float) If `distribution` = 't_student', df is a number of degrees of freedom.
        Common used values are: 0.1, 0.25.
    :return: (pd.Series) BVC buy volume.
    """

    # .apply(norm.cdf) is used to omit Warning for norm.cdf(pd.Series with NaNs)
    devadarsh.track('get_bvc_buy_volume')

    if distribution == 'norm':
        buy_volume_frac = volume * (close.diff() / close.diff().rolling(window=window).std()).apply(norm.cdf)
    elif distribution == 't_student':
        buy_volume_frac = volume * (close.diff() / close.diff().rolling(window=window).std()).apply(
            lambda x: t.cdf(x, df=df))
    else:
        raise ValueError('Unknown value for `distribution`: use either `norm` or `t_student`')

    return buy_volume_frac
