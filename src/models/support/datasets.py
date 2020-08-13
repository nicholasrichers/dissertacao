import pandas as pd
def get_data(data_path):
  from sklearn.preprocessing import LabelBinarizer

  data = pd.read_csv(data_path, index_col=['Account_ID'])
  X = data.drop('y', axis=1)
  #X.drop(X.columns[0],axis=1,inplace=True)
  y = data.y.apply(lambda x: 1 if x == 'yes' else 0)



  return X, y