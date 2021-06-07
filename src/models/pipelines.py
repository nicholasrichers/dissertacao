from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd


def get_criteria(model, importances_df):

    if model[:10]=='linear/mdi': criteria = 1/importances_df.shape[0]
    if model[:10]=='linear/mda': criteria = 0
    if model[:10]=='linear/sfi': criteria = importances_df['mean'].quantile(.3)
    if model[:8]=='shap/ebm': criteria = importances_df['mean'].quantile(.3)
        
    importances_df = importances_df[importances_df['mean']>criteria]
    return list(importances_df.index)

def get_features(model_path, features_sort, ranker=False):

    url = 'https://raw.githubusercontent.com/nicholasrichers/dissertacao/master/reports/feature_importance/'
    importances_df = pd.read_csv(url+model_path+'.csv')
    #importances_df = importances_df.reindex(features_sort)

    features_selected = get_criteria(model_path, importances_df)
    if ranker ==True: features_selected = ['era']+features_selected

    return features_selected


def feature_selection_pipeline(model_path, features_sort, ranker=False):
  # Create the transformers for numeric features 

  #model_path = 'linear/sfi_vanilla'
  features_selected = get_features(model_path, features_sort, ranker)


  fs_ct = ColumnTransformer([
      ('feature_selection', 'passthrough', features_selected)                          
  ])

  # Create the pipeline to transform numeric features
  fs_pipeline = Pipeline([
          ('fs_ct', fs_ct),
          #('scaler', RobustScaler()),
          #('minimax', MinMaxScaler())
  ])  
  
  return fs_pipeline



def get_full_pipeline():
  # Create the categorical and numeric pipelines
  #cat_pipeline = get_categorical_pipeline()
  fs_pipeline = feature_selection_pipeline()

  # Create the feature union of categorical and numeric attributes
  ft_union = FeatureUnion([
    #('cat_pipeline', cat_pipeline),
    ('fs_pipeline', fs_pipeline)
  ])

  pipeline = Pipeline([
    ('ft_union', ft_union)
  ])

  return pipeline