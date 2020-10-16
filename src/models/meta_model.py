
from sklearn.model_selection import  cross_val_predict, GroupKFold
import numpy as np

def create_preds_meta_full(models, df, splits=3):


  features = [c for c in df if c.startswith("feature")]
  eras = df.era.values
  X = df[features]
  y = df['target_nomi']

  #Group K-fold
  CV_out = GroupKFold(n_splits = splits)


  cv_mean = []
  stack_data = dict()
  for fold, (tr, ts) in enumerate(CV_out.split(X ,y, eras)):
      
      print("fitting fold:", fold)
      X_train, X_test = X.iloc[tr], X.iloc[ts]
      y_train, y_test = y.iloc[tr], y.iloc[ts]
      X_train_rank = X_train.copy()
      X_train_rank['era'] = df.iloc[tr].era



      predictions_cv, predictions_test = [], []
      for name, model_pipe in models.items():
          print("creating predictions to:", name)
          model = model_pipe.model

          #grp k-fold interno
          eras_in = df.iloc[tr].era
          CV_in = GroupKFold(n_splits = splits)
          grp_in = CV_in.split(X_train, y_train, eras_in.values)
          
          if name.startswith('xgb_ranker'):
            predictions_cv.append(cross_val_predict(model, X_train_rank, y_train, cv=grp_in).reshape(-1,1))
            model.fit(X_train_rank, y_train)

          else:
            predictions_cv.append(cross_val_predict(model, X_train, y_train, cv=grp_in).reshape(-1,1))
            model.fit(X_train, y_train)

          ptest = model.predict(X_test)
          predictions_test.append(ptest.reshape(-1,1))
      
      
      #out loop
      predictions_cv = np.concatenate(predictions_cv, axis=1)
      predictions_test = np.concatenate(predictions_test, axis=1)
      
      stack_data['fold_'+str(fold)] = {'Xtrain': predictions_cv,
                                       'Xtest': predictions_test,
                                       'ytrain': y_train,
                                       'ytest': y_test}

      
      #stacker = Ridge()
      #stacker.fit(predictions_cv, y_train)
      #error = spearman(y_test, stacker.predict(predictions_test))
      #cv_mean.append(error)
      #print('RMSLE Fold %d - RMSLE %.4f' % (fold, error))
      
  #print('RMSLE CV5 %.4f' % np.mean(cv_mean))
  return stack_data
    

#stack_data, era_series = get_preds_meta_cv(models_meta, df_training, splits=3)
#stack_data




def create_preds_meta_light(models, df, splits=3):


  features = [c for c in df if c.startswith("feature")]
  eras = df.era.values
  X = df[features]
  y = df['target_nomi']

  #Group K-fold
  CV_out = GroupKFold(n_splits = splits)


  cv_mean = []
  stack_data = dict()
  for fold, (tr, ts) in enumerate(CV_out.split(X ,y, eras)):
      
      print("fitting fold:", fold)
      X_train, X_test = X.iloc[tr], X.iloc[ts]
      y_train, y_test = y.iloc[tr], y.iloc[ts]
      X_train_rank = X_train.copy()
      X_train_rank['era'] = df.iloc[tr].era



      predictions_cv, predictions_test = [], []
      for name, model_pipe in models.items():
          print("creating predictions to:", name)
          model = model_pipe.model

          #grp k-fold interno
          eras_in = df.iloc[tr].era
          CV_in = GroupKFold(n_splits = splits)
          grp_in = CV_in.split(X_train, y_train, eras_in.values)
          
          if name.startswith('xgb_ranker'):
            #predictions_cv.append(cross_val_predict(model, X_train_rank, y_train, cv=grp_out).reshape(-1,1))
            model.fit(X_train_rank, y_train)

          else:
            #predictions_cv.append(cross_val_predict(model, X_train, y_train, cv=grp_out).reshape(-1,1))
            model.fit(X_train, y_train)

          ptest = model.predict(X_test)
          predictions_test.append(ptest.reshape(-1,1))
      
      
      #out loop
      #predictions_cv = np.concatenate(predictions_cv, axis=1)
      predictions_test = np.concatenate(predictions_test, axis=1)
      
      stack_data['fold_'+str(fold)] = {#'Xtrain': predictions_cv,
                                       'Xtest': predictions_test,
                                       #'ytrain': y_train,
                                       'ytest': y_test}

      
      #stacker = Ridge()
      #stacker.fit(predictions_cv, y_train)
      #error = spearman(y_test, stacker.predict(predictions_test))
      #cv_mean.append(error)
      #print('RMSLE Fold %d - RMSLE %.4f' % (fold, error))
      
  #print('RMSLE CV5 %.4f' % np.mean(cv_mean))
  return stack_data

#block_cv_data = get_preds_meta_cv(models_meta, df_training, splits=3)
#block_cv_data



######################################################################
######################################################################
######################################################################
######################################################################


import pickle
import pandas as pd
def get_stacked_data(meta_model="Sao_Paulo", kind="full", local="colab"):

  if local=='colab':
    file_path = '/content/drive/My Drive/Numerai/'+meta_model+'/stacked_data_'+kind+'.csv'

  else:

    file_path = '../../Data/processed/meta_model/'+meta_model+'/stacked_data_'+kind+'.csv'

  stacked_data= pd.read_csv(file_path)



  #with open(file_path, 'rb') as handle: #load
  #    stacked_data = pickle.load(handle)  

  #print(stacked_data)
  return stacked_data



#only accpets light data
def mount_stacked_data_light(data, model_names, train):
	#Ainda precisa incluir o target kaz e nomear as features no padrao feature_<Model

  df = pd.DataFrame()
  for fold in data.keys():
    df_X = pd.DataFrame(data[fold]['Xtest'])
    df_y = pd.DataFrame(data[fold]['ytest'])
    target_name = data[fold]['ytest'].name
    
    #concat both dataframes with right index
    df_fold = pd.concat([df_X.set_index(df_y.index), df_y], axis=1)
    df_fold['ind'] = df_fold.index
    df=df.append(df_fold)

  df = df.sort_values('ind')
  df.drop(['ind'], axis=1, inplace=True)
  df.set_axis(model_names+[target_name], axis=1, inplace=True)
  df['era'] = train.era
  df['id'] = train.id

  return df, model_names



