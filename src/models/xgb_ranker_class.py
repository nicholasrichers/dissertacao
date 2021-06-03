
import pandas as pd
from xgboost import XGBRanker
from sklearn.base import BaseEstimator, RegressorMixin

def fs():
  url = 'https://raw.githubusercontent.com/nicholasrichers/dissertacao/master/reports/feature_importance/linear/'
  importances_df = pd.read_csv(url+'sfi_vanilla'+'.csv')
  #importances_df = importances_df.reindex(df_training.columns)
  criteria = importances_df['mean'].quantile(.3)
  importances_df = importances_df[importances_df['mean']>criteria]
  features_custom = list(importances_df.index)
  return features_custom

class XGBRanker_FS123(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        features_custom = fs()
        x=pd.DataFrame(x, columns=['era']+features_custom)
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[features_custom], y, group=group)

    def predict(self, x):
        features_custom = fs()
        x=pd.DataFrame(x, columns=['era']+features_custom)
        return super().predict(x[features_custom])




class MyXGBRanker(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[x.columns[1:]], y, group=group)

    def predict(self, x):
        return super().predict(x[x.columns[:]])


class XGBRanker_FS(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        x=pd.DataFrame(x)
        cdf = x.groupby(0).agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x.iloc[:,1:], y, group=group)

    def predict(self, x):
        x=pd.DataFrame(x)
        return super().predict(x.iloc[:,1:])







        