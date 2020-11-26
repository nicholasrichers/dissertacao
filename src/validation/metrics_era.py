import scipy
from scipy.stats import skew, kurtosis, sem, gmean
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np



def numerai_score(y_true, y_pred, df):

    #create y_true as df
    y_true = y_true.to_frame(name='target')
    y_true = y_true.join(df['era'])

    #create y_pred as df
    preds_df = pd.DataFrame(y_pred, index = y_true.index, columns=['preds'])
    preds_df = preds_df.join(df['era'])

    #print(y_true['era'].unique())

    era_scores = pd.Series(index=y_true['era'].unique())
    for era in y_true['era'].unique():
        era_df = y_true[y_true['era'] == era]
        era_preds = preds_df[preds_df['era'] == era]
        era_scores[era] = np.corrcoef(era_df['target'], 
                                      era_preds['preds'].rank(pct=True, method="first"))[0,1]


    return era_scores


def sharpe_ratio(y_true, y_pred, df):

  era_scores = numerai_score(y_true, y_pred, df)
  return np.mean(era_scores)/np.std(era_scores)

def validation_mean(y_true, y_pred, df):

  era_scores = numerai_score(y_true, y_pred, df)
  return np.mean(era_scores)


def annual_sharpe(x):
    return ((np.mean(x) -0.010415154) /np.std(x, ddof=1)) #* np.sqrt(12) 


def adj_sharpe(y_true, y_pred, df):
  x = numerai_score(y_true, y_pred, df)
  return annual_sharpe(x) * (1 + ((skew(x) / 6) * annual_sharpe(x)) - ((kurtosis(x) - 0) / 24) * (annual_sharpe(x) ** 2)) 

def ar1(x):
    a = np.corrcoef(x[:-1], x[1:])[0,1]
    return a

def autocorr_penalty(x):
    n = len(x)
    p = np.abs(ar1(x))
    return np.sqrt(1 + 2*np.sum([((n - i)/n)*p**i for i in range(1,n)]))

def smart_sharpe(y_true, y_pred, df):
  x = numerai_score(y_true, y_pred, df)
  #print(x)
  return (np.mean(x)/(np.std(x, ddof=1) * autocorr_penalty(x))) #* np.sqrt(12))




##############################################################################
##############################################################################
##############################################################################




def spearman_(target, pred):
    from scipy import stats
    return stats.spearmanr(target, pred)[0]

def eras_score(y_true, y_pred):

    #create y_true as df
    y_true = y_true.to_frame(name='target')
    y_true = y_true.join(df_training['era'])

    #create y_pred as df
    preds_df = pd.DataFrame(y_pred, index = y_true.index, columns=['preds'])
    preds_df = preds_df.join(df_training['era'])
    era_scores = pd.Series(index=y_true['era'].unique())

    for era in y_true['era'].unique():
        era_df = y_true[y_true['era'] == era]
        era_preds = preds_df[preds_df['era'] == era]
        era_scores[era] = np.corrcoef(era_df['target'], 
                                      era_preds['preds'].rank(pct=True, method="first"))[0,1]


    return era_scores


def sharpe_ratio_(y_true, y_pred):
  x = eras_score(y_true, y_pred)
  return np.mean(x)/np.std(x)



def smart_sharpe_(y_true, y_pred):
  x = eras_score(y_true, y_pred)
  return (np.mean(x)/(np.std(x, ddof=1) * autocorr_penalty(x)))



def adj_sharpe_(y_true, y_pred):
  x = eras_score(y_true, y_pred)
  return annual_sharpe(x) * (1 + ((skew(x) / 6) * annual_sharpe(x)) - ((kurtosis(x) - 0) / 24) * (annual_sharpe(x) ** 2)) 









