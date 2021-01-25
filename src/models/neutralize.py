

import numpy as np
import pandas as pd
import scipy  

from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.functional import F



def exposures(x, y):
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=0)
    y = y - y.mean(dim=0)
    y = y / y.norm(dim=0)
    return torch.matmul(x.T, y)



def reduce_exposure(prediction, features, max_exp):
    # linear model of features that will be used to partially neutralize predictions
    lin = Linear(features.shape[1],  1, bias=False)
    lin.weight.data.fill_(0.)
    model = Sequential(lin)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-4)

    feats = torch.tensor(np.float32(features)-.5)
    pred = torch.tensor(np.float32(prediction))
    start_exp = exposures(feats, pred[:,None])

    # set target exposure for each feature to be <= current exposure
    # if current exposure is less than max_exp, or <= max_exp if  
    # current exposure is > max_exp
    targ_exp = torch.clamp(start_exp, -max_exp, max_exp)

    for i in range(100000):#100000
        optimizer.zero_grad()
        # calculate feature exposures of current linear neutralization
        exps = exposures(feats, pred[:,None]-model(feats))

        # loss is positive when any exposures exceed their target
        loss = (F.relu(F.relu(exps)-F.relu(targ_exp)) + F.relu(F.relu(-exps)-F.relu(-targ_exp))).sum()
        #print(loss)
        print(f'       loss: {loss:0.7f}', end='\r')

        if loss < 1e-7: #7
            #print('11111')
            neutralizer = [p.detach().numpy() for p in model.parameters()]
            neutralized_pred = pred[:,None]-model(feats)
            break
        loss.backward()
        optimizer.step()
    return neutralized_pred, neutralizer



def reduce_all_exposures(df, column, neutralizers=[],
                                     normalize=True,
                                     gaussianize=True,
                                     era_col="era",
                                     max_exp=0.10):
  
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, '\r') #print era
        df_era = df[df[era_col] == u]
        scores = df_era[column].values #preds
        exposure_values = df_era[neutralizers].values #features
        
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                if gaussianize:
                    x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2)[0]

        scores, neut = reduce_exposure(scores, exposure_values, max_exp)

        scores /= scores.std()

        computed.append(scores.detach().numpy())

    return pd.DataFrame(np.concatenate(computed), columns=column, index=df.index)


def neutralize_by_threshold(df, column, neutralizers=[],
                                     normalize=True,
                                     gaussianize=True,
                                     era_col="era",
                                     max_exp=0.10):
  

  data_rfe = reduce_all_exposures(df, column, neutralizers, 
                                  normalize, gaussianize, 
                                  era_col="era", 
                                  max_exp=0.10)
  
  df[column] = data_rfe[column]
  df[column]  -= df[column] .min()
  df[column]  /= df[column] .max()

  return df[column]







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
   


def preds_neutralized_old(df, columns, by, ml_model, proportion=1.0):

    preds_neutr = df.groupby("era").apply( lambda x: normalize_and_neutralize(x, columns, by, ml_model, proportion))

    preds_neutr = MinMaxScaler().fit_transform(preds_neutr).reshape(1,-1)[0]

    return preds_neutr

def preds_neutralized(df, columns, by, ml_model, proportion=1.0):

    for group_by in by:
      feat_by = [c for c in df if c.startswith('feature_'+group_by)]
      #print(feat_by)
      preds_neutr = df.groupby("era").apply( lambda x: normalize_and_neutralize(x, columns, feat_by, ml_model, proportion))
      df[columns] = MinMaxScaler().fit_transform(preds_neutr).reshape(1,-1)[0]
      preds_neutr_after = df[columns].values

    return preds_neutr_after





fn_strategy_dict = {

'ex_preds': {'strategy': 'after', 
             'func': preds_neutralized,  
             'columns': ['preds'], 
             'by': [''] ,      
             'model': [LinearRegression(fit_intercept=False), None], 
             'factor': 0.0
            },


'ex_FN100': {'strategy': 'after', 
             'func': preds_neutralized, 
             'columns': ['preds'], 
             'by': [''] ,
             'model': [LinearRegression(fit_intercept=False), None], 
             'factor':1.0
            },



'lgbm_exp20': {'strategy': 'after', 
               'func': preds_neutralized,  
               'columns': ['preds'], 
               'by': [''] ,
               'model': [SGDRegressor(tol=0.001), None], 
               'factor': 0.0
              },


'lgbm_slider20': {'strategy': 'after', 
                  'func': preds_neutralized,  
                  'columns': ['preds'], 
                  'by': [''] ,
                  'model': [SGDRegressor(tol=0.001), None], 
                  'factor':0.0
                 },






'nrichers': {'strategy':  None, 
             'func': None, 
             'columns': ['preds'], 
             'by': [''] , 
             'model': [None, None], 
             'factor': None
            },




'nick_richers': {'strategy': 'after', 
                 'func': preds_neutralized, 
                 'columns': ['preds'], 
                 'by': [''], 
                 'model': [SGDRegressor(tol=0.001), None],
                 'factor': 1.0
                },




#R1
'nr_rio': {'strategy': 'after', 
           'func': preds_neutralized, 
           'columns': ['preds'], 
           'by': [''] , 
           'model': [SGDRegressor(tol=0.001), None ], 
           'factor':0.4
          },
 





'nr__rio': {'strategy': 'after', 
            'func': preds_neutralized, 
            'columns': ['preds'], 
            'by': [''] ,
            'model': [SGDRegressor(tol=0.001), Ridge(alpha=0.5)], 
            'factor':0.75
           },





'nr__sao_paulo':  {'strategy': 'after', 
                   'func': preds_neutralized, 
                   'columns': ['preds'], 
                   'by': [''] , 
                   'model': [SGDRegressor(tol=0.001), None], 
                   'factor': 0.9
                  },




'nr__medellin': {'strategy': 'after', 
                 'func': preds_neutralized, 
                 'columns': ['preds'], 
                 'by': [''] , 
                 'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 
                 'factor': 0.75
                 },




'nr__guadalajara':{'strategy': 'after', 
                   'func': preds_neutralized, 
                   'columns': ['preds'], 
                   'by': ['intelligence', 'dexterity', 'strength', 'constitution'] , 
                   'model': [LinearRegression(fit_intercept=False), Ridge(alpha=0.5)], 
                   'factor': 0.75
                  },


}













