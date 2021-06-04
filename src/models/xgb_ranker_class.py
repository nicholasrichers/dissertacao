
import pandas as pd
from xgboost import XGBRanker
from sklearn.base import BaseEstimator, RegressorMixin




class MyXGBRanker(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[x.columns[1:]], y, group=group)

    def predict(self, x):
        return super().predict(x[x.columns[:]])


class XGBRanker_PIPE(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        x=pd.DataFrame(x)
        cdf = x.groupby(0).agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x.iloc[:,1:], y, group=group)

    def predict(self, x):
        x=pd.DataFrame(x)
        return super().predict(x.iloc[:,1:])







        