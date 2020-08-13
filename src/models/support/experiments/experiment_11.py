import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from datasets import get_data
from sklearn.impute import SimpleImputer
import numpy as np
from category_encoders import BinaryEncoder, CatBoostEncoder, OrdinalEncoder


def get_features_types():
  X, y = get_data('../data/trainDF.csv')
  dtypes = pd.DataFrame(X.dtypes.rename('type')).reset_index().astype('str')
  numeric = dtypes[(dtypes.type.isin(['int64', 'float64']))]['index'].values
  categorical = dtypes[~(dtypes['index'].isin(numeric)) & (dtypes['index'] != 'y')]['index'].values
  return categorical, numeric


def get_numeric_features():
  X, y = get_data('../data/trainDF.csv')
  dtypes = pd.DataFrame(X.dtypes.rename('type')).reset_index().astype('str')
  numeric = dtypes[(dtypes.type.isin(['int64', 'float64']))]['index'].values
  return numeric


#NUM_FEAT = get_numeric_features()
CAT_FEAT, NUM_FEAT = get_features_types()


def get_categorical_pipeline():

  # Create the transformers for categorical features
  cat_ct = ColumnTransformer([('categoricals', 'passthrough', CAT_FEAT)])

  # Create the pipeline to transform categorical features
  cat_pipeline = Pipeline([
          ('cat_ct', cat_ct),
          ('ohe', OneHotEncoder(handle_unknown='ignore'))
  ])

  return cat_pipeline

def get_numeric_pipeline():
  # Create the transformers for numeric features
  NUM_FEAT_PIPE11 = list(set(NUM_FEAT)-set(['Activities_Last_30_Days']))
  num_ct = ColumnTransformer([
      ('fill_diff', SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-1),NUM_FEAT_PIPE11)                          
  ])

  # Create the pipeline to transform numeric features
  num_pipeline = Pipeline([
          ('num_union', num_ct),
          ('scaler', RobustScaler()),
          ('minimax', MinMaxScaler())
  ])  
  return num_pipeline

def get_pipeline(cat_pipeline, num_pipeline):
  # Create the categorical and numeric pipelines
  #cat_pipeline = get_categorical_pipeline()
  #num_pipeline = get_numeric_pipeline()

  # Create the feature union of categorical and numeric attributes
  ft_union = FeatureUnion([
    ('cat_pipeline', cat_pipeline),
    ('num_pipeline', num_pipeline)
  ])

  pipeline = Pipeline([
    ('ft_union', ft_union)
  ])

  return pipeline



def get_full_pipeline():
  # Create the categorical and numeric pipelines
  cat_pipeline = get_categorical_pipeline()
  num_pipeline = get_numeric_pipeline()

  # Create the feature union of categorical and numeric attributes
  ft_union = FeatureUnion([
    ('cat_pipeline', cat_pipeline),
    ('num_pipeline', num_pipeline)
  ])

  pipeline = Pipeline([
    ('ft_union', ft_union)
  ])

  return pipeline



def save_pipeline(pipeline, file_path):
  #from sklearn.externals import joblib
  from joblib import dump
  dump(pipeline, file_path)




def load_pipeline(file_path):
  from joblib import load
  model = load(file_path)
  return model



def baseline_model_predictions(X, y, n_targeted):
  # Combine the targeted and random groups
  baseline_targets = X[X.Account_ICP_Tier.isin(['Tier S', 'Tier A'])].sample(n=n_targeted, random_state=20)
  baseline_ys = y.loc[baseline_targets.index]

  return baseline_ys



def random_model_predictions(X, y, n_targeted):
  # Combine the targeted and random groups
  random_targets = X.sample(n=n_targeted, random_state=42)
  random_ys = y.loc[random_targets.index]
  return random_ys



def get_model_profit(X, y, n_targeted, avg_revenue,avg_cost, generic_predictions=None, preds=None):

  if preds is None: preds = generic_predictions(X, y, n_targeted)
    
  outcomes = preds.apply(lambda x: avg_cost if x == 0 else avg_cost + avg_revenue)
  profit = sum(outcomes)
  return profit









