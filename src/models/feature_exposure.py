


import csv
import numpy as np
import pandas as pd

TOURNAMENT_NAME = "kazutsugi"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"


def feature_exposure(df):
    df = df[df.data_type == 'validation']
    feature_columns = [x for x in df.columns if x.startswith('feature_')]
    pred = df[PREDICTION_NAME]
    correlations = []
    for col in feature_columns:
        correlations.append(np.corrcoef(pred, df[col])[0, 1])
    return np.std(correlations)


def max_feature_exposure(df):
    df = df[df.data_type == 'validation']
    feature_columns = [x for x in df.columns if x.startswith('feature_')]
    fe = {}
    for era in df.era.unique():
        era_df = df[df.era == era]
        pred = era_df[PREDICTION_NAME]
        correlations = []
        for col in feature_columns:
            correlations.append(np.corrcoef(pred, era_df[col])[0, 1])
        fe[era] = np.std(correlations)
    return max(fe.values())


def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if
              x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, index_col=0)

    return df


if __name__ == '__main__':
    tournament_data = read_csv(
        "numerai_tournament_data.csv")
    example_predictions = read_csv(
        "example_predictions_target_kazutsugi.csv")
    merged = pd.merge(tournament_data, example_predictions,
                      left_index=True, right_index=True)
    fe = feature_exposure(merged)
    max_fe = max_feature_exposure(merged)
    print(f"Feature exposure: {fe:.4f} "
          f"Max feature exposure: {max_fe:.4f}")





    