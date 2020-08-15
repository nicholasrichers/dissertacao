import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from category_encoders import BinaryEncoder, OrdinalEncoder
from datasets import get_data
from sklearn.impute import SimpleImputer
import numpy as np


########
def get_features_types():
  X, y = get_data('../data/trainDF.csv')
  dtypes = pd.DataFrame(X.dtypes.rename('type')).reset_index().astype('str')
  numeric = dtypes[(dtypes.type.isin(['int64', 'float64']))]['index'].values
  categorical = dtypes[~(dtypes['index'].isin(numeric)) & (dtypes['index'] != 'y')]['index'].values
  return categorical, numeric



CAT_FEAT, NUM_FEAT = get_features_types()


ORDINAL_FEATURES =  [
  'ZoomInfo_Revenue_Range',
  'Account_ICP_Tier',
  'Page_Count_Range',
  'Parent_Account_Status'
]

DEL_NUM_PIPE2 = [
  'Activities_Last_30_Days', 
  'Organic_Visits',
  'Annual_Revenue_converted', 
  'Page_Count'
]


MAP_ORDINAL = [

{'col': 'ZoomInfo_Revenue_Range', 
'mapping': {'Under $500,000': 0, '$500,000 - $1 mil.': 1, '$1 mil. - $5 mil.': 2, 
            '$5 mil. - $10 mil.': 3, '$10 mil. - $25 mil.': 4,
            '$25 mil. - $50 mil.' : 5, '$50 mil. - $100 mil.': 6, '$100 mil. - $250 mil.': 7,
            '$250 mil. - $500 mil.': 8,  '$500 mil. - $1 bil.' : 9,
            '$1 bil. - $5 bil.': 10, 'Over $5 bil.': 11, }},

    
{'col': 'Page_Count_Range', 
'mapping': {'<500': 0, 'Between 500 and 1K': 1,'Between 1K and 5K': 2, 'Between 5K and 10K': 3, 
            'Between 10K and 50K': 4, 'Between 50K and 100K': 5, '<100K': 6,
            'Between 100K and 250K': 7, 'Between 250K and 500K': 8, 'Between 500K and 1M': 9, '>1M': 10}},

    
{'col': 'Account_ICP_Tier', 
'mapping': {'Tier C': 0, 'Tier B': 1, 'Tier A': 2, 'Tier S': 3}},  

{'col': 'Parent_Account_Status', 
'mapping': {'Lost Customer': 0, 'Prospect': 1, 'Active Customer': 2}}

]


######

def get_categorical_pipeline():
  # Create the transformers for categorical features

    cat_features = [
    #('categoricals', 'passthrough', CAT_FEAT),
    ('binary', OrdinalEncoder(), 'ZoomInfo_Global_HQ_Country'),
    ('catboost', OrdinalEncoder(handle_unknown='value', handle_missing='value'), 'Adjusted_Industry'),
    ('ordinal',OrdinalEncoder(mapping=MAP_ORDINAL, handle_unknown='value'), ORDINAL_FEATURES),
    ]
    
    cat_ct = ColumnTransformer(cat_features)

    #Create the pipeline to transform categorical features
    cat_pipeline = Pipeline([
          ('cat_ct', cat_ct),
          #('ohe', OneHotEncoder(handle_unknown='ignore'))
      ])

    return cat_pipeline



def get_numeric_pipeline():
  # Create the transformers for numeric features
  NUM_FEAT_PIPE2 = list(set(NUM_FEAT)-set(DEL_NUM_PIPE2))
  num_ct = ColumnTransformer([
      ('fill_diff', SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-1), NUM_FEAT_PIPE2)                          
  ])

  # Create the pipeline to transform numeric features
  num_pipeline = Pipeline([
          ('num_union', num_ct),
          ('scaler', RobustScaler()),
          ('minimax', MinMaxScaler())
  ])  
  return num_pipeline


####################

def get_pipeline(cat_pipeline, num_pipeline):

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



############################

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









