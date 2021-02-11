import scipy
import math
from scipy.stats import skew, kurtosis, sem, gmean, norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
#from scipy import stats as scipy_stats

#to use in google colab
try:
  from src.validation import dsr
  from src.validation import metrics_description
  
except:
  #import dsr
  import metrics_description

import warnings
warnings.filterwarnings("ignore")

#to use in google colab
#try:
#  from src.validation import dsr
#except:
#  import dsr


TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = "target"#_{TOURNAMENT_NAME}"
PREDICTION_NAME = 'preds'#_{TOURNAMENT_NAME}"


#OK
def spearman(pred, target):
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]



#by era basis
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


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
    return np.mean(x)/np.std(x) #* np.sqrt(12)




def max_drawdown(x):
	#print("checking max drawdown...")
    rolling_max = (x + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (x + 1).cumprod()
    max_drawdown = -(rolling_max - daily_value).max()
    #print(f"max drawdown: {max_drawdown}")
    return max_drawdown


##ddof = delta degrees of freedom
## https://www.statsdirect.com/help/basics/degrees_of_freedom.htm
def smart_sharpe(x):
    return (np.mean(x)/(np.std(x, ddof=1) * autocorr_penalty(x))) #* np.sqrt(12)) 


##approximated their average trading costs
def numerai_sharpe(x):
    return ((np.mean(x) -0.010415154) /np.std(x, ddof=1)) #* np.sqrt(12)


## usado embaixo
def annual_sharpe(x):
    return ((np.mean(x) -0.010415154) /np.std(x, ddof=1)) #* np.sqrt(12) 




#Skewness and kurtosis: https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa
#https://rdrr.io/cran/PerformanceAnalytics/man/AdjustedSharpeRatio.html#:~:text=See%20Also%20Examples-,Description,negative%20skewness%20and%20excess%20kurtosis.
# fonte: https://forum.numer.ai/t/probabilistic-sharpe-ratio/446/5
#https://quantdare.com/probabilistic-sharpe-ratio/
def adj_sharpe(x):
    return annual_sharpe(x) * (1 + ((skew(x) / 6) * annual_sharpe(x)) - ((kurtosis(x) - 0) / 24) * (annual_sharpe(x) ** 2)) #(kurtosis(x) - 3)




def probabilistic_sharpe_ratio(x=None, sr_benchmark=0.0):
    n = len(x)
    sr = np.mean(x) / np.std(x, ddof=1)
    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew(x) * sr) + (((kurtosis(x) - 3) / 4) * sr ** 2)) / (n - 1))
    psr = scipy.stats.norm.cdf((sr - sr_benchmark) / sr_std)
    if math.isnan(psr): psr=1.0000
    return psr



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
    pred = pd.Series(pred, index=df.index)
    feature_columns = [x for x in df.columns if x.startswith('feature_')]


    # Check the feature exposure of your validation predictions
    #feature_exposures = df[feature_columns].apply(lambda d: correlation(pred, d), axis=0)
    max_per_era = df.groupby("era").apply(lambda d: d[feature_columns].corrwith(pred).abs().max())
    mean_per_era = df.groupby("era").apply(lambda d: np.sqrt(np.mean(np.square(d[feature_columns].corrwith(pred))))) 


    return max_per_era.std(), max_per_era.mean(), mean_per_era




#OK
def feature_exposure_old(df, pred):
    #df = df[df.data_type == 'validation']
    pred = pd.Series(pred, index=df.index)

    feature_columns = [x for x in df.columns if x.startswith('feature_')]
    correlations = []
    for col in feature_columns:
        correlations.append(np.corrcoef(pred.rank(pct=True, method="first"), df[col])[0, 1])
    corr_series = pd.Series(correlations, index=feature_columns)
    return np.std(correlations), max(np.abs(correlations)), corr_series



# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]





# to neutralize a column in a df by many other columns
def neutralize_old(df, columns, by, proportion=1.0):
    scores = df.loc[:, columns] #preds
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures,
         np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))

    exposures = np.hstack(
    	(exposures, 
    	np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))


    scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))

    return scores / scores.std()




def neutralize(df, columns, extra_neutralizers=None, proportion=1.0, normalize=True, era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        #print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std()

        computed.append(scores)

    #print(pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index))
    return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)







def get_feature_neutral_mean(ddf, preds):
    df = ddf.copy()
    df[PREDICTION_NAME] = preds
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME], feature_cols)[PREDICTION_NAME]

    scores = df.groupby("era").apply(lambda x: spearman(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)




def calculate_fnc(sub, targets, features):
    """    
    Args:
        sub (pd.Series)
        targets (pd.Series)
        features (pd.DataFrame)
    """
    
    # Normalize submission
    sub = (sub.rank(method="first").values - 0.5) / len(sub)

    # Neutralize submission to features
    f = features.values
    sub -= f.dot(np.linalg.pinv(f).dot(sub))
    sub /= sub.std()
    
    sub = pd.Series(np.squeeze(sub)) # Convert np.ndarray to pd.Series

    # FNC: Spearman rank-order correlation of neutralized submission to target
    fnc = np.corrcoef(sub.rank(pct=True, method="first"), targets)[0, 1]

    return fnc



# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)



def mmc_metrics(df, preds, model):

    validation_data = df.copy()

    # Load example preds to get MMC metrics
    file_path = 'https://raw.githubusercontent.com/nicholasrichers/dissertacao/master/reports/predicoes_validacao/'+model+'_preds_test.csv'
    example_preds = pd.read_csv(file_path).set_index("id")[model]
    validation_data.set_index("id", inplace=True)


    validation_example_preds = example_preds.loc[validation_data.index]
    validation_data["ExamplePreds"] = validation_example_preds
    validation_data[PREDICTION_NAME] = preds

    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in validation_data.groupby("era"):
        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                                   pd.Series(unif(x["ExamplePreds"])))

        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2)) #standard deviation of a uniform distribution is 0.29 (MM2 annoucement)

        corr_scores.append(spearman(unif(x[PREDICTION_NAME]), x[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    #corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

    #print(
    #    f"MMC Mean: {val_mmc_mean}\n"
    #    f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
    #    f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
    #)

    # Check correlation with example predictions
    corr_with_example_preds = np.corrcoef(validation_example_preds.rank(pct=True, method="first"),
                                          validation_data[PREDICTION_NAME].rank(pct=True, method="first"))[0, 1]

    #print(f"Corr with example preds: {corr_with_example_preds}")


    return val_mmc_mean, corr_plus_mmc_sharpe, corr_with_example_preds, validation_data["ExamplePreds"]


#############
def metrics_consolidated(df):
    
    df_cons = dict()
    for model in df.keys():
        
        #adicionando as colunas de valor
        df_cons[model] = df[model]["Valor"]
        
        
    #adicionando as colunas de descricao
    df_cons["Categoria"] = df[model]["Categoria"]
    df_cons["Range_Aceitavel"] = df[model]["Range_Aceitavel"]
    df_cons["Descricao"] = df[model]["Descricao"]
    df_cons = pd.DataFrame.from_dict(df_cons)
    return df_cons


def submission_metrics(df_val, preds, model_name, full=True, meta=''):

    features = [c for c in df_val if c.startswith("feature")]
    new_df = df_val.copy()
    #new_df['target'] = new_df['target']
    new_df[PREDICTION_NAME] = preds #caso seja classificacao (1..4)
    era_scores = pd.Series(index=new_df['era'].unique())
    era_scores_pearson = pd.Series(index=new_df['era'].unique())
    era_scores_diff = pd.Series(index=new_df['era'].unique())

        
    for era in new_df['era'].unique():
        era_df = new_df[new_df['era'] == era]
        era_scores[era] = spearman(era_df[PREDICTION_NAME], era_df['target'])
        era_scores_pearson[era] = np.corrcoef(era_df[PREDICTION_NAME], era_df['target'])[0,1]
        era_scores_diff[era] = era_scores[era] - era_scores_pearson[era]

    era_scores.sort_values(inplace=True)
    era_scores.sort_index(inplace=True)
    
    #print("Qtde. eras:", len(new_df['era'].unique()))
    #era_scores.plot(kind="bar")
    #print("performance over time")
    #plt.show()


    values = dict()
    values['Model_Name'] = model_name
    
    #performance
    values['Validation_Sharpe'] = validation_sharpe(era_scores)
    values['Validation_Mean'] = np.mean(era_scores)
    values['Feat_neutral_mean'] = get_feature_neutral_mean(df_val, preds)

    #risk
    values['Validation_SD'] = np.std(era_scores)
    #values['Feat_exp_std'], values['Feat_exp_max'], feat_corrs  = feature_exposure(df_val, preds)
    values['Max_Drawdown'] = max_drawdown(era_scores)

    #mmc
    #values['val_mmc_mean'], values['corr_plus_mmc_sharpe'], values['corr_with_example_preds'], _ = mmc_metrics(df_val, preds, 'ex_preds')


    if full:
        #print("Calculating all metrics")
        values['Feat_exp_std'], values['Feat_exp_max'], feat_corrs  = feature_exposure(df_val, preds)
        values['val_mmc_mean'], values['corr_plus_mmc_sharpe'], values['corr_with_example_preds'], _ = mmc_metrics(df_val, preds, 'ex_preds')
        values['FNC'] = calculate_fnc(pd.Series(pred, index=df_val.index), df_val.target, df_val[features])
        

        values['Median_corr'] = np.median(era_scores)
        values['Variance'] = np.var(era_scores)
        values['AR(1)'] = ar1(era_scores)
        values['Skewness'] = skew(era_scores)
        values['Exc_Kurtosis'] = kurtosis(era_scores)
        values['Std_Error_Mean'] = sem(era_scores)   # fonte: https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp
        values['Smart_Sharpe'] = smart_sharpe(era_scores)
        values['Numerai_Sharpe'] = numerai_sharpe(era_scores)
        values['Ann_Sharpe'] = annual_sharpe(era_scores)
        values['Adj_Sharpe'] = adj_sharpe(era_scores)
        values['Prob_Sharpe'] = probabilistic_sharpe_ratio(era_scores)
        values['VaR_10%'] = VaR(era_scores)
        values['Sortino_Ratio'] = sortino_ratio(era_scores)
        values['Smart_Sortino_Ratio'] = smart_sortino_ratio(era_scores)
        values['Payout'] = payout(era_scores)
        values['val_mmc_mean_FN'], values['corr_plus_mmc_sharpe_FN'], values['corr_with_ex_FN100'], _  = mmc_metrics(df_val, preds, 'ex_FN100')


    else:
        #print("Summary all metrics")
        values['Feat_exp_std'], values['Feat_exp_max'], feat_corrs  = 0,0,0 #feature_exposure(df_val, preds)
        values['val_mmc_mean'], values['corr_plus_mmc_sharpe'], values['corr_with_example_preds'], _ = 0,0,0,0 #mmc_metrics(df_val, preds, 'ex_preds')
        values['FNC'] = 0


        values['Median_corr'] = 0
        values['Variance'] = 0
        values['AR(1)'] = 0
        values['Skewness'] = 0
        values['Exc_Kurtosis'] = 0
        values['Std_Error_Mean'] = 0
        values['Smart_Sharpe'] = 0
        values['Numerai_Sharpe'] = 0
        values['Ann_Sharpe'] = 0
        values['Adj_Sharpe'] = 0
        values['Prob_Sharpe'] = 0
        values['VaR_10%'] = 0
        values['Sortino_Ratio'] = 0
        values['Smart_Sortino_Ratio'] =0
        values['Payout'] = 0
        values['val_mmc_mean_FN'], values['corr_plus_mmc_sharpe_FN'], values['corr_with_ex_FN100'], _  = 0,0,0,0


        
        



    metrics = metrics_description.get_metrics_dicts(values)

    #get DSR
    try:
        if meta=='': meta=model_name[4:]
        dict_dsr = dsr.dsr_summary(meta+'/era_scores_'+model_name+'.csv')
        

    except:
        dict_dsr = {"Metrica": 'Deflated_Sharpe', 
                 "Valor": 0, 
                 "Categoria": "Special", 
                 "Range_Aceitavel": "[0.5..1]", 
                 "Descricao": "Sharpe Descontado pelas tentativas" }


    metrics.append(dict_dsr)
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics = df_metrics.set_index('Metrica')
    scores = {'spearman': era_scores, 
              'pearson':  era_scores_pearson, 
              'diff':     era_scores_diff}



    return scores['spearman'], df_metrics, feat_corrs


########################################################################################################################
########################################################################################################################
########################################################################################################################



def sharpe_metrics(era_scores):
     

    values = dict()
    model_name = era_scores['hparam']
    era_scores = era_scores[:-1]
    
    cv_scores = era_scores.filter(like='train_', axis=0)
    val_scores = era_scores.filter(like='val_', axis=0)
    folds = ['_on_cv', '_on_test', '_full']
    scores = [cv_scores, val_scores, era_scores]
    
    for fold, data_score in zip(folds, scores):
    
        values['Mean_Corr'+fold] = np.mean(data_score)
        values['Sharpe_Ratio'+fold] = validation_sharpe(data_score)
        values['Adj_Sharpe'+fold] = adj_sharpe(data_score)
        values['Prob_Sharpe'+fold] = probabilistic_sharpe_ratio(data_score)

    
    values['hparam'] = model_name
    return values






