
##########
# File: stats.py
# Description:
#    Coleção de funções estatisticas auxiliares
##########

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
import seaborn as sns
import pandas as pd


def seasonal_plot(df, ini, fim, frq):
  years = df.index.year.unique()
  groups = df[ini:fim].groupby(pd.Grouper(freq=frq))
  years = pd.DataFrame()
  pyplot.figure()
  i = 1
  n_groups = len(groups)
  for name, group in groups:
    pyplot.subplot((n_groups*100) + 10 + i)
    i += 1
    pyplot.plot(group)
  pyplot.tight_layout()
  pyplot.show()


def adf_test(df):
  res = []
  for col in df.columns:
    dftest = adfuller(df[col], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Nº Obs.'])

    for key,value in dftest[4].items():
        dfoutput['Critical Val.(%s)'%key] = value
    res.append(dfoutput.values)

  result = pd.DataFrame(res, index =df.columns,columns= dfoutput.index )
  return result