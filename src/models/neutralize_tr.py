


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
    #print('threshold..: ', max_exp)
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
                                     max_exp=0.07):
  
    #print('threshold.: ', max_exp)
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        if (u % 5==0):
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
                                     max_exp=0.07):
  
  print('threshold: ', max_exp)
  print('neutralizers {} & {} '.format(neutralizers[0],neutralizers[-1]))
  
  data_rfe = reduce_all_exposures(df, column, neutralizers, 
                                  normalize, gaussianize, 
                                  era_col="era", 
                                  max_exp=0.07)
  
  df[column] = data_rfe[column]
  df[column]  -= df[column] .min()
  df[column]  /= df[column] .max()

  return df[column]








def preds_neutralized_TR(ddf, l1_preds, features):
  path = '../../predictions/shanghai/'
    
  if tournament_data.data_type.unique()[-1:] == "live":
  

    df = ddf.copy()
    df['preds'] = l1_preds


    #precisa cortar era 212
    df_mini = df[df.era.isin(df.era.unique()[:306])==False]


    params = [df_mini,['preds'], features,True,True,"era",0.06]
    preds_live = neutralize_by_threshold(*params)


    #precisa colar
    preds_test = pd.read_csv(path+'predicoes_TR_test.csv')
    preds_sub =  preds_test.append(preds_live, ignore_index=True)


  #only val data (toy)
  else: preds_sub = pd.read_csv(path+'predicoes_TR_val.csv')
  return preds_sub










