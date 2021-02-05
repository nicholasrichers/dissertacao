

import numpy as np
import pandas as pd
import scipy  

from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge






def _neutralize(df, columns, by, ml_model, proportion): #['preds'], features,
    scores = df[columns] #preds
    exposures = df[by].values #features
    ml_model[0].fit(exposures, scores.values.reshape(1,-1)[0])
    neutr_preds = pd.DataFrame(ml_model[0].predict(exposures), index=df.index, columns=columns)
    #exposures.dot(np.linalg.pinv(exposures).dot(scores))    

    
    if ml_model[1] != None:
        ml_model[1].fit(exposures, scores.values.reshape(1,-1)[0])
        neutr_preds2 = pd.DataFrame(ml_model[1].predict(exposures), index=df.index, columns=columns)
        #print(neutr_preds2)

    else: neutr_preds2 = 0# np.zeros(len(scores))


    scores = scores - ((proportion[0] * neutr_preds) + ((proportion[1]) * neutr_preds2))



    #scores = scores - proportion * neutr_preds
    return scores / scores.std()



def _normalize(df):
    X = (df.rank(method="first") - 0.5) / len(df)
    return scipy.stats.norm.ppf(X)


def normalize_and_neutralize(df, columns, by, ml_model, proportion):
    # Convert the scores to a normal distribution
    df[columns] = _normalize(df[columns])
    df[columns] = _neutralize(df, columns, by, ml_model, proportion)
    return df[columns]
   


def preds_neutralized_old(ddf, columns, by, ml_model, proportion):

    df = ddf.copy()
    preds_neutr = df.groupby("era").apply( lambda x: normalize_and_neutralize(x, columns, by, ml_model, proportion))

    preds_neutr = MinMaxScaler().fit_transform(preds_neutr).reshape(1,-1)[0]

    return preds_neutr




def preds_neutralized(ddf, columns, by, ml_model, proportion):

    df = ddf.copy()  
    preds_neutr = dict()
    for group_by in by:
        feat_by = [c for c in df if c.startswith('feature_'+group_by)]
        
        
        df[columns]=df.groupby("era").apply(
            lambda x:normalize_and_neutralize(x,columns,feat_by,ml_model, proportion))
        
        preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1,-1)[0]

    return preds_neutr_after


def preds_neutralized_groups(ddf, columns, by, ml_model, proportion):

    df = ddf.copy()  

    for group_by, p in by.items():
        feat_by = [c for c in df if c.startswith('feature_'+group_by)]
        
        df[columns]=df.groupby("era").apply(
            lambda x:normalize_and_neutralize(x,columns,feat_by,ml_model,[p[0], p[1]]))
        
        preds_neutr_after = MinMaxScaler().fit_transform(df[columns]).reshape(1,-1)[0]

    return preds_neutr_after





fn_strategy_dict = {

'ex_preds': {'strategy': 'after', 
             'func': preds_neutralized,  
             'columns': ['preds'], 
             'by': [''] ,      
             'model': [LinearRegression(fit_intercept=False), None], 
             'factor': [0.0, 0.0]
            },

'ex_preds1': {'strategy': 'after', 
             'func': preds_neutralized,  
             'columns': ['preds'], 
             'by': [''] ,      
             'model': [LinearRegression(fit_intercept=False), None], 
             'factor': [0.0, 0.0]
            },


'ex_FN100': {'strategy': 'after', 
             'func': preds_neutralized, 
             'columns': ['preds'], 
             'by': [''] ,
             'model': [LinearRegression(fit_intercept=False), None], 
             'factor': [0.0, 0.0]
            },



'lgbm_exp20': {'strategy': 'after', 
               'func': preds_neutralized,  
               'columns': ['preds'], 
               'by': [''] ,
               'model': [SGDRegressor(tol=0.001), None], 
               'factor': [0.0, 0.0]
              },


'lgbm_slider20': {'strategy': 'after', 
                  'func': preds_neutralized,  
                  'columns': ['preds'], 
                  'by': [''] ,
                  'model': [SGDRegressor(tol=0.001), None], 
                  'factor': [0.0, 0.0]
                 },






'nrichers': {'strategy':  None, 
             'func': None, 
             'columns': ['preds'], 
             'by': [''] , 
             'model': [None, None], 
             'factor': [0, 0]
            },





'nick_richers':{'strategy': 'after', 
                   'func': preds_neutralized_groups, 
                   'columns': ['preds'],

                    'by': {'constitution':[2.0,-0.5], 'strength':    [2.0,-0.5], 
                           'dexterity':   [2.0,-0.5], 'charisma':    [0.0,0], 
                           'wisdom':      [0.0,0], 'intelligence':[2.0,-0.5]},
            
                      
                   'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 
                   'factor': []
                  },



#R1
'nr_rio': {'strategy': 'after', 
           'func': preds_neutralized, 
           'columns': ['preds'], 
           'by': [''] , 
           'model': [SGDRegressor(tol=0.001), None ], 
           'factor': [0.4, 0.0]
          },
 





'nr__rio': {'strategy': 'after', 
            'func': preds_neutralized, 
            'columns': ['preds'], 
            'by': [''] ,
            'model': [SGDRegressor(tol=0.001), Ridge(alpha=0.5)], 
            'factor': [0.75, 0.25]
           },





'nr__sao_paulo':  {'strategy': 'after', 
                   'func': preds_neutralized, 
                   'columns': ['preds'], 
                   'by': [''] , 
                   'model': [SGDRegressor(tol=0.001), None], 
                   'factor': [0.9, 0.1]
                  },




'nr__medellin': {'strategy': 'after', 
                 'func': preds_neutralized, 
                 'columns': ['preds'], 
                 'by': [''] , 
                 'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 
                 'factor': [0.75, 0.25]
                 },




'nr__guadalajara':{'strategy': 'after', 
                   'func': preds_neutralized, 
                   'columns': ['preds'], 
                   'by': ['constitution', 'strength', 'dexterity', 'intelligence'], 
                   'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 
                   'factor': [0.75, 0.25]
                  },



'nr__san_francisco':{'strategy': 'after', 
                   'func': preds_neutralized_groups, 
                   'columns': ['preds'],

                    'by': {'constitution':[2.0,0.15], 'strength':   [2.0,0.25], 
                           'dexterity':   [2.0,0.0], 'charisma':   [-1.0,0.25], 
                           'wisdom':     [-1.0,0.5], 'intelligence':[2.0,0.0]},
            
                      
                   'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 
                   'factor': []
                  },


}












