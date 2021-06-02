
import pandas
from xgboost import XGBRanker
from sklearn.base import BaseEstimator, RegressorMixin


class XGBRanker_Custom(XGBRanker, BaseEstimator, RegressorMixin):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[features_custom], y, group=group)

    def predict(self, x):
        return super().predict(x[features_custom])