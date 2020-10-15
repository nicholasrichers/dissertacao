
from sklearn.model_selection import  cross_val_predict, GroupKFold

def get_preds_meta_cv(models, df, splits=3):


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