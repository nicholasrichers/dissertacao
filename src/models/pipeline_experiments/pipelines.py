

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion



def target_pipeline():
  # Create the transformers for numeric features
  target_pipe = ColumnTransformer([
      ('LabelEncoder_target', LabelEncoder(), 'target_kazutsugi')]
      #,remainder='passthrough'
  )

  # Create the pipeline to transform numeric features
  classification_pipeline = Pipeline([
          ('target', target_pipe),
    ])  

  return classification_pipeline



