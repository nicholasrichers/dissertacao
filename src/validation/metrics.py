import scipy
from scipy.stats import skew, kurtosis, sem, gmean
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np

#to use in google colab
try:
  from src.validation import metrics_description
except:
  import metrics_description


#OK
def spearmanr(target, pred):
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]

#Pearson corr
def ar1(x):
    return np.corrcoef(x[:-1], x[1:])[0,1]

#https://forum.numer.ai/t/performance-stationarity/151 (VER O PAPER KEY QUANT)
def autocorr_penalty(x):
    n = len(x)
    p = np.abs(ar1(x))
    return np.sqrt(1 + 2*np.sum([((n - i)/n)*p**i for i in range(1,n)]))

# Sobre a anualizacao 12/sqrt(12) o std cresce com o aumento da amostra
#https://quant.stackexchange.com/questions/22397/sharpe-ratio-why-the-normalization-factor?noredirect=1&lq=1
#https://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio
def validation_sharpe(x):
    return np.mean(x)/np.std(x) * np.sqrt(12)



##ddof = delta degrees of freedom
## https://www.statsdirect.com/help/basics/degrees_of_freedom.htm
def smart_sharpe(x):
    return (np.mean(x)/(np.std(x, ddof=1) * autocorr_penalty(x)) * np.sqrt(12)) 


##approximated their average trading costs
def numerai_sharpe(x):
    return ((np.mean(x) - 0.010415154) / np.std(x, ddof=1)) * np.sqrt(12)

## usado embaixo
def annual_sharpe(x):
    return ((np.mean(x) -0.010415154) /np.std(x, ddof=1)) * np.sqrt(12) 




#Skewness and kurtosis: https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa
#https://rdrr.io/cran/PerformanceAnalytics/man/AdjustedSharpeRatio.html#:~:text=See%20Also%20Examples-,Description,negative%20skewness%20and%20excess%20kurtosis.
# fonte: https://forum.numer.ai/t/probabilistic-sharpe-ratio/446/5
#https://quantdare.com/probabilistic-sharpe-ratio/
def adj_sharpe(x):
    return annual_sharpe(x) * (1 + ((skew(x) / 6) * annual_sharpe(x)) - ((kurtosis(x) - 0) / 24) * (annual_sharpe(x) ** 2)) #(kurtosis(x) - 3)


#https://www.investopedia.com/ask/answers/033115/how-can-you-calculate-value-risk-var-excel.asp
def VaR(x):
    return -np.mean(x) - np.sqrt(np.var(x)) * np.percentile(x, 10)



#https://forum.numer.ai/t/performance-stationarity/151
#https://www.investopedia.com/terms/s/sortinoratio.asp
def sortino_ratio(x, target=0.010415154):
    xt = x - target
    return np.mean(xt) / (np.sum(np.minimum(0, xt)**2)/(len(xt)-1))**.5


#OK
def smart_sortino_ratio(x, target=0.010415154): ##approximated their average trading costs
    xt = x - target
    return np.mean(xt)/(((np.sum(np.minimum(0, xt)**2)/(len(xt)-1))**.5)*autocorr_penalty(x))




# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25).mean()


#OK
def feature_exposure(df, pred):
    #df = df[df.data_type == 'validation']
    feature_columns = [x for x in df.columns if x.startswith('feature_')]
    correlations = []
    for col in feature_columns:
        correlations.append(np.corrcoef(pred, df[col])[0, 1])
    corr_series = pd.Series(correlations, index=feature_columns)
    return np.std(correlations), max(np.abs(correlations)), corr_series






def submission_metrics(df_val, preds, model_name=''):


    new_df = df_val.copy()
    new_df['target'] = new_df['target_kazutsugi']
    new_df["pred"] = minmax_scale(preds) #caso seja classificacao (1..4)
    #new_df['pred'] = new_df['target_kazutsugi']
    era_scores = pd.Series(index=new_df['era'].unique())

    print("Qtde. eras:", len(new_df['era'].unique()))
    
    for era in new_df['era'].unique():
        era_df = new_df[new_df['era'] == era]
        era_scores[era] = spearmanr(era_df['pred'], era_df['target'])

    era_scores.sort_values(inplace=True)
    era_scores.sort_index(inplace=True)
    era_scores.plot(kind="bar")
    print("performance over time")
    plt.show()


    values = dict()
    values['Model_Name'] = model_name
    values['Max_Drawdown'] = np.min(era_scores)
    values['Validation_Mean'] = np.mean(era_scores)
    values['Median_corr'] = np.median(era_scores)
    values['Variance'] = np.var(era_scores)
    values['Std_Dev'] = np.std(era_scores)
    values['AR(1)'] = ar1(era_scores)
    values['Skewness'] = skew(era_scores)
    values['Exc_Kurtosis'] = kurtosis(era_scores)
    values['Std_Error_Mean'] = sem(era_scores)   # fonte: https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp
    values['Validation_Sharpe'] = validation_sharpe(era_scores)
    values['Smart_Sharpe'] = smart_sharpe(era_scores)
    values['Numerai_Sharpe'] = numerai_sharpe(era_scores)
    #values['Ann_Sharpe'] = annual_sharpe(era_scores)
    values['Adj_Sharpe'] = adj_sharpe(era_scores)
    values['VaR_10%'] = VaR(era_scores)
    values['Sortino_Ratio'] = sortino_ratio(era_scores)
    values['Smart_Sortino_Ratio'] = smart_sortino_ratio(era_scores)
    values['Payout'] = payout(era_scores)


    #by feature metrics
    values['Feat_exp_std'], values['Feat_exp_max'], feat_corrs  = feature_exposure(df_val, preds)


    metrics = metrics_description.get_metrics_dicts(values)
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics = df_metrics.set_index('Metrica')



    return era_scores, era_df, df_metrics, feat_corrs


########################################################################################################################
########################################################################################################################
########################################################################################################################














