
import pandas
from xgboost import XGBRanker


class MyXGBRanker(XGBRanker):
    def fit(self, x, y):
        cdf = x.groupby('era').agg(['count'])
        group = cdf[cdf.columns[0]].values
        return super().fit(x[features], y, group=group)

    def predict(self, x):
        return super().predict(x[features])