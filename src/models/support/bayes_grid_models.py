#param drid models for BayessearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

#skopt
lr_param_grid = {
  'lr__C': Real(1e-3, 100, 'log-uniform'),
  'lr__fit_intercept': Categorical([True, False]), 
  'lr__max_iter': Integer(1e+2, 1e+5, 'log-uniform'),
  'lr__penalty':  Categorical(['l1', 'l2']),
  'lr__solver': Categorical(['liblinear', 'saga']),
  'lr__tol': Real(1e-5, 1e-3, 'log-uniform'),
  'lr__class_weight':  Categorical([None, 'balanced']),

}




#skopt
xgb_param_grid = {
        'xgb__colsample_bylevel' : Real(1e-1, 1, 'uniform'),
        'xgb__colsample_bytree' : Real(6e-1, 1, 'uniform'),
        'xgb__gamma' :  Real(5e-1, 6, 'log-uniform'),
        'xgb__learning_rate' : Real(10**-5, 10**0, "log-uniform"),
        'xgb__max_depth' : Integer(1, 25, 'uniform'),
        'xgb__min_child_weight' : Integer(1, 10, 'uniform'),
        'xgb__n_estimators' : Integer(50, 400, 'log-uniform'),
        'xgb__reg_alpha' : Real(1e-2, 1, 'log-uniform'),
        'xgb__reg_lambda' : Real(1e-2, 1, 'log-uniform'),
        'xgb__subsample' : Real(6e-1, 1, 'uniform'),
        'xgb__scale_pos_weight' : Integer(1, 10000, 'log-uniform')
        
}



#skopt
lgbm_param_grid = {
    'lgbm__boosting_type': Categorical(['gbdt', 'goss', 'dart']),
    'lgbm__num_leaves':  Integer(2, 10, 'uniform'),
    'lgbm__learning_rate': Real(10**-5, 10**0, "log-uniform"),
    'lgbm__subsample_for_bin': Integer(20000, 300000, 'log-uniform'),
    'lgbm__min_child_samples': Integer(20, 500, 'uniform'),
    'lgbm__reg_alpha': Real(10**-3, 10**0, "log-uniform"),
    'lgbm__reg_lambda': Real(10**-3, 10**0, "log-uniform"),
    'lgbm__colsample_bytree': Real(6e-1, 1, 'uniform'),
    'lgbm__subsample': Real(5e-1, 1, 'uniform'),
    'lgbm__is_unbalance': Categorical([True, False]),
    #'lgbm__class_weight': Categorical([None, 'balanced'])

}



#############################################################################
#############################################################################
#############################################################################
#############################################################################

## VER https://github.com/ledmaster/notebooks_tutoriais/blob/master/como_tunar_hipers.ipynb
## https://sigopt.com/blog

from skopt import forest_minimize
def tune_lgbm(params):
    print(params)
    lr = params[0]
    max_depth = params[1]
    min_child_samples = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    n_estimators = params[5]
    
    min_df = params[6]
    ngram_range = (1, params[7])
    
    title_vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
    title_bow_train = title_vec.fit_transform(title_train)
    title_bow_val = title_vec.transform(title_val)
    
    Xtrain_wtitle = hstack([Xtrain, title_bow_train])
    Xval_wtitle = hstack([Xval, title_bow_val])
    
    mdl = LGBMClassifier(learning_rate=lr, num_leaves=2 ** max_depth, max_depth=max_depth, 
                         min_child_samples=min_child_samples, subsample=subsample,
                         colsample_bytree=colsample_bytree, bagging_freq=1,n_estimators=n_estimators, random_state=0, 
                         class_weight="balanced", n_jobs=6)
    mdl.fit(Xtrain_wtitle, ytrain)
    
    p = mdl.predict_proba(Xval_wtitle)[:, 1]
    
    print(roc_auc_score(yval, p))
    
    return -average_precision_score(yval, p)


space = [(1e-3, 1e-1, 'log-uniform'), # lr
          (1, 10), # max_depth
          (1, 20), # min_child_samples
          (0.05, 1.), # subsample
          (0.05, 1.), # colsample_bytree
          (100,1000), # n_estimators
          (1,5), # min_df
          (1,5)] # ngram_range

#res = forest_minimize(tune_lgbm, space, random_state=160745, n_random_starts=20, n_calls=50, verbose=1)
