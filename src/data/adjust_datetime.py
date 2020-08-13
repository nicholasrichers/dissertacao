
##########
# File: adjust_datetime.py
# Description:
#    Coleção de funções para ajustar/converter datetime
##########


import pendulum
import pandas as pd


#Funcao para corrigir o numero da semana  do mes (1:5)
def week_of_month(data):
  dt = pendulum.parse(str(data))
  return dt.week_of_month



def next_holiday(df):
	holidays= df[df.holidays > 0 ].index
	last_holiday =  pd.to_datetime('2019-09-08', format='%Y/%m/%d') #1o feriado após a série histórica
	holidays_list = holidays.to_list()
	holidays_list.append(last_holiday)

	delta = []
	for ind in df.index:
	  
	  if (ind > holidays_list[0]): holidays_list.pop(0)

	  gap = (holidays_list[0] - ind).days
	  delta.append(int(round(gap/7,0)))

	return delta