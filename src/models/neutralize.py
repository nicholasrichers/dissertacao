

import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def _neutralize(df, columns, by, ml_model, proportion=1.0): #['preds'], features,
    scores = df[columns] #preds
    exposures = df[by].values #features
    ml_model.fit(exposures, scores.values.reshape(1,-1)[0])
    neutr_preds = pd.DataFrame(ml_model.predict(exposures), index=df.index, columns=columns)
    #exposures.dot(np.linalg.pinv(exposures).dot(scores))    
    
    scores = scores - proportion * neutr_preds
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



















