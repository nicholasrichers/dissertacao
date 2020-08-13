#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:08:34 2020

@author: nicholasrichers
"""

##########
# File: pre_processing_vendas.py
# Description:
#    Rotina de preprocessamento vendas
##########




import sys
sys.path.append('../libraries/')

import pandas as pd
import math
from adjust_datetime import week_of_month
from adjust_datatype import PT_BR_string_to_numeric



#processa o df_vendas
def pre_vendas(df_vendas):

	#COLS: Total, SM, ROUTE e Inditetos
    for col in df_vendas.columns[1:5]: 
        df_vendas[col] = PT_BR_string_to_numeric(df_vendas, col)
          
  	#cria a coluna outros para dar a soma de 'Total'
    df_vendas['OUTROS'] = df_vendas['Total'] - df_vendas[df_vendas.columns[2:5]].sum(axis=1)

  	#COLS: Week_445, Month_445, Day_gregoriano, Month_gregoriano, Year, Data.1
  	#Vamos também renomear a coluna Data.1 para Datetime, e trocar o tipo para datetime.
    df_vendas.rename(columns={'Data.1':'Datetime'}, inplace=True)
    df_vendas['Datetime'] = pd.to_datetime(df_vendas['Datetime'], format='%d/%m/%Y')

	  #usa a funcao week_of_month de adjust_datime
    df_vendas['Week_Month'] = df_vendas['Datetime'].map(week_of_month)

    years = math.ceil(df_vendas.shape[0]/53)
    df_vendas['Week_Year'] = ([i for i in range(1,53)]*years)[0:df_vendas.shape[0]]

	  #colunas cortadas
    drop_cols = ['Data', 'Month_445', 'Week_445', 'weekend', 'work_day']
    df_vendas.drop(labels=drop_cols, axis=1, inplace=True)


    return df_vendas