import scipy
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
  import dsr
  import metrics_description



#to use in google colab
#try:
#  from src.validation import dsr
#except:
#  import dsr


TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = "target"#_{TOURNAMENT_NAME}"
PREDICTION_NAME = "prediction"#_{TOURNAMENT_NAME}"


#OK
def spearman(pred, target):
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

    return scipy.stats.norm.cdf((sr - sr_benchmark) / sr_std)



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
def neutralize(df, columns, by, proportion=1.0):
    scores = df.loc[:, columns]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures,
         np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))

    scores = scores - proportion * exposures.dot(np.linalg.pinv(exposures).dot(scores))

    return scores / scores.std()


def get_feature_neutral_mean(df, preds):
    df["preds"] = preds
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, ["preds"], feature_cols)["preds"]

    scores = df.groupby("era").apply(lambda x: spearman(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)




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






def mmc_metrics(df, preds):

	validation_data = df.copy()

	 # Load example preds to get MMC metrics
	example_preds = pd.read_csv("../../data/interim/example_predictions.csv").set_index("id")["prediction"]
	validation_data.set_index("id", inplace=True)


	validation_example_preds = example_preds.loc[validation_data.index]
	validation_data["ExamplePreds"] = validation_example_preds
	validation_data["preds"] = preds

	# MMC over validation
	mmc_scores = []
	corr_scores = []
	for _, x in validation_data.groupby("era"):
	    series = neutralize_series(pd.Series(unif(x["preds"])),
	                               pd.Series(unif(x["ExamplePreds"])))

	    mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
	    corr_scores.append(spearman(unif(x["preds"]), x[TARGET_NAME]))

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
	                                      validation_data["preds"].rank(pct=True, method="first"))[0, 1]

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


def submission_metrics(df_val, preds, model_name='',  mmc=True, meta_model=''):


    new_df = df_val.copy()
    #new_df['target'] = new_df['target']
    new_df["pred"] = minmax_scale(preds) #caso seja classificacao (1..4)
    era_scores = pd.Series(index=new_df['era'].unique())

        
    for era in new_df['era'].unique():
        era_df = new_df[new_df['era'] == era]
        era_scores[era] = spearman(era_df['pred'], era_df['target'])

    era_scores.sort_values(inplace=True)
    era_scores.sort_index(inplace=True)
    
    #print("Qtde. eras:", len(new_df['era'].unique()))
    #era_scores.plot(kind="bar")
    #print("performance over time")
    #plt.show()


    values = dict()
    values['Model_Name'] = model_name
    values['Max_Drawdown'] = max_drawdown(era_scores)
    values['Validation_Mean'] = np.mean(era_scores)
    values['Median_corr'] = np.median(era_scores)
    values['Variance'] = np.var(era_scores)
    values['Validation_SD'] = np.std(era_scores)
    values['AR(1)'] = ar1(era_scores)
    values['Skewness'] = skew(era_scores)
    values['Exc_Kurtosis'] = kurtosis(era_scores)
    values['Std_Error_Mean'] = sem(era_scores)   # fonte: https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp
    values['Validation_Sharpe'] = validation_sharpe(era_scores)
    values['Smart_Sharpe'] = smart_sharpe(era_scores)
    values['Numerai_Sharpe'] = numerai_sharpe(era_scores)
    values['Ann_Sharpe'] = annual_sharpe(era_scores)
    values['Adj_Sharpe'] = adj_sharpe(era_scores)
    values['Prob_Sharpe'] = probabilistic_sharpe_ratio(era_scores)
    values['VaR_10%'] = VaR(era_scores)
    values['Sortino_Ratio'] = sortino_ratio(era_scores)
    values['Smart_Sortino_Ratio'] = smart_sortino_ratio(era_scores)
    values['Payout'] = payout(era_scores)


    #by feature metrics
    values['Feat_exp_std'], values['Feat_exp_max'], feat_corrs  = feature_exposure(df_val, preds)



    if model_name=="ex_preds":
        values['Feat_neutral_mean'] = get_feature_neutral_mean(df_val, preds)

    else:
        values['Feat_neutral_mean'] = get_feature_neutral_mean(df_val, preds)



    if mmc==True:
	    values['val_mmc_mean'], values['corr_plus_mmc_sharpe'], values['corr_with_example_preds'], example_predicts = mmc_metrics(df_val, preds)

    else:
	    values['val_mmc_mean'], values['corr_plus_mmc_sharpe'], values['corr_with_example_preds'], example_predicts = 0,validation_sharpe(era_scores),1 ,preds



    metrics = metrics_description.get_metrics_dicts(values)

    #get DSR
    try:
        
        dict_dsr = dsr.dsr_summary(meta_model+'/era_scores_'+model_name+'.csv')
        

    except:
        dict_dsr = {"Metrica": 'Deflated_sharpe_ratio', 
                 "Valor": 0, 
                 "Categoria": "Special", 
                 "Range_Aceitavel": "[0.5..1]", 
                 "Descricao": "Sharpe Descontado pelas tentativas" }


    metrics.append(dict_dsr)
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics = df_metrics.set_index('Metrica')



    return era_scores, df_metrics, feat_corrs, example_predicts


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






