#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:09:15 2020

@author: nicholasrichers
"""

##########
# File: pre_processing_macro.py
# Description:
#    Rotina de preprocessamento macro
##########


import sys
sys.path.append('../libraries/')


import pandas as pd
from adjust_datatype import PT_BR_string_to_numeric


#processa o df_macro
def pre_macro(df_macro):
    #object para datetime
    for col in df_macro.columns[1::]:
        df_macro[col] = PT_BR_string_to_numeric(df_macro, col)

    df_macro['Data'] = pd.to_datetime(df_macro['Data'], format='%d/%m/%Y')
    df_macro.drop(labels=['Ocupacao'], axis=1, inplace=True)
    return df_macro




