# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/portfoliolab/blob/master/LICENSE.txt

"""
This module allows us to track how the library is used and measure statistics such as growth and lifetime.
"""
# pylint: disable=missing-function-docstring, eval-used

import os
from datetime import datetime as dt
from requests import get

import analytics
import getmac


def validate_env_variable(env_variable_name):
    try:
        is_valid = bool(os.environ[env_variable_name])
    except KeyError:
        is_valid = False

    return is_valid


def get_user():
    user = getmac.get_mac_address()
    if user is None:
        user = 'Alfred Borden'
    return user


def validate_alum():
    try:
        is_circle = bool(validate_env_variable('IS_CIRCLECI'))
        is_travis = bool(validate_env_variable('IS_TRAVIS'))
        is_dev = is_circle or is_travis
    except KeyError:
        is_dev = False

    return is_dev


def identify():
    if not IS_DEV:
        try:
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


def page(url):
    if not IS_DEV:
        analytics.page(USER, 'MlFinLab', 'Import',
                       {"url": url,
                        'time': dt.now()})


def track(func):
    if not IS_DEV:
        analytics.track(USER, func, {'time': dt.now()})


# Env Var
SEGMENT = 'Tw8COFZpaxtMs04LUuv965YGd9QEM7cL'

USER = get_user()
IS_DEV = validate_alum()

# Connect with DB
analytics.write_key = SEGMENT
