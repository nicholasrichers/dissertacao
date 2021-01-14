

import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler   
import pandas as pd



from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge




def _neutralize(df, columns, by, ml_model, proportion=1.0): #['preds'], features,
    scores = df[columns] #preds
    exposures = df[by].values #features
    ml_model[0].fit(exposures, scores.values.reshape(1,-1)[0])
    neutr_preds = pd.DataFrame(ml_model[0].predict(exposures), index=df.index, columns=columns)
    #exposures.dot(np.linalg.pinv(exposures).dot(scores))    

    
    if ml_model[1] != None:
        ml_model[1].fit(exposures, scores.values.reshape(1,-1)[0])
        neutr_preds2 = pd.DataFrame(ml_model[1].predict(exposures), index=df.index, columns=columns)

    else: neutr_preds2 = 0# np.zeros(len(scores))


    scores = scores - ((proportion * neutr_preds) + ((1-proportion) * neutr_preds2))



    #scores = scores - proportion * neutr_preds
    return scores / scores.std()



def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)


def normalize_and_neutralize(df, columns, by, ml_model, proportion=1.0):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, ml_model, proportion)
    return df[columns]
   


def preds_neutralized(df, columns, by, ml_model, proportion=1.0):

    preds_neutr = df.groupby("era").apply( lambda x: normalize_and_neutralize(x, columns, by, ml_model, proportion))

    preds_neutr = MinMaxScaler().fit_transform(preds_neutr).reshape(1,-1)[0]

    return preds_neutr



fn_strategy_dict = {

'ex_preds':     {'strategy':'after', 'func': normalize_and_neutralize, 'model': [LinearRegression(fit_intercept=False), None], 'factor':0.0},
'ex_FN100':     {'strategy':'after', 'func': normalize_and_neutralize, 'model': [LinearRegression(fit_intercept=False), None], 'factor':1.0},

'lgbm_exp20':   {'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), None ], 'factor':0.0},
'lgbm_slider20':{'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), None ], 'factor':0.0},

'nrichers':     {'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), None ], 'factor':0.0},
'nick_richers': {'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), None ], 'factor':1.0},
'nr_rio':       {'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), None ], 'factor':0.4}, #R1

'nr__rio':      {'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), Ridge(alpha=0.5)                       ], 'factor':0.75},
'nr__sao_paulo':{'strategy':'after', 'func': normalize_and_neutralize, 'model': [SGDRegressor(), None                                   ], 'factor':0.9},
'nr__medellin': {'strategy':'after', 'func': normalize_and_neutralize, 'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 'factor':0.75},
}














