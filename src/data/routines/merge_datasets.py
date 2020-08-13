#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:09:57 2020

@author: nicholasrichers
"""

##########
# File: merge_datasets.py
# Description:
#    Rotina de preprocessamento para unir os datasets
##########



import pandas as pd




#une os datasets
def merge_df(df_vendas, df_macro):

	#ajeitar datas
	monthly_series = pd.date_range(start='2015-01-01',end ='2019-11-01', freq='MS')
	monthly_series_pd = pd.Series(index=monthly_series)
	df_macro_weekly = monthly_series_pd.asfreq('W-SAT').to_frame()


	for col in df_macro.columns[1::]:
	  monthly_series_pd = pd.Series(data = list(df_macro[col]), index=monthly_series)
	  df_macro_weekly[col] = monthly_series_pd.asfreq('W-SAT', method='ffill')
	df_macro_weekly.drop(labels=[0], axis=1, inplace=True)

	#ajeitar datas
	df_macro_weekly_short = df_macro_weekly[df_macro_weekly.index <= '2019-08-17'].reset_index(drop=True)

	df_merged = pd.concat([df_vendas, df_macro_weekly_short], axis=1)
	df_merged = df_merged.set_index('Datetime')


	return df_merged
