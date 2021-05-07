# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/portfoliolab/blob/master/LICENSE.txt

"""
This module allows us to track how the library is used and measure statistics such as growth and lifetime.
"""

import os
from datetime import datetime as dt
from requests import get
import analytics
import getmac


# pylint: disable=missing-function-docstring
def validate_env_variable(env_variable_name):
    try:
        is_valid = bool(os.environ[env_variable_name])
    except KeyError:
        is_valid = False

    return is_valid


# pylint: disable=missing-function-docstring
def get_user():
    user = getmac.get_mac_address()
    if user is None:
        user = 'Alfred Borden'
    return user


# pylint: disable=missing-function-docstring
def validate_alum():
    try:
        is_circle = bool(validate_env_variable('IS_CIRCLECI'))
        is_alum = get_user()[-4:] == '62a0'
        is_dev = is_circle or is_alum
    except KeyError:
        is_dev = False

    return is_dev


# Identify
# pylint: disable=missing-function-docstring
def identify():
    if not IS_DEV:
        try:
            # pylint: disable=eval-used
            data = eval(get('https://ipinfo.io/').text)
            data['user'] = get_user()
            analytics.identify(USER, {'$name': USER,
                                      '$country_code': data['country'],
                                      '$region': data['region'],
                                      '$city': data['city'],
                                      '$location': data['loc'],
                                      })
        except ConnectionError:
            analytics.identify(USER,
                               {'$name': USER})


# pylint: disable=missing-function-docstring
def page(url):
    if not IS_DEV:
        analytics.page(USER, 'PortfolioLab', 'Import',
                       {"url": url,
                        'time': dt.now()})


# pylint: disable=missing-function-docstring
def track(func):
    if not IS_DEV:
        analytics.track(USER, func, {'time': dt.now()})


# Env Var
SEGMENT = 'U6xeFA5C0CzVod9DQDvzD2UamjRZ8MmB'

USER = get_user()
IS_DEV = validate_alum()

# Connect with DB
analytics.write_key = SEGMENT
