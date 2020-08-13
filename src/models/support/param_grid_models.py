#param drid models for randomsearchCV

import numpy as np
xgb_param_grid = {
        'xgb__colsample_bylevel' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        'xgb__colsample_bytree' :[0.6, 0.7, 0.8, 1.0],
        'xgb__gamma' : list(np.linspace(0.05, 1, 6)),
        'xgb__learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        'xgb__max_depth' : list(range(3, 30, 3)),
        'xgb__min_child_weight' : list(range(1, 11, 2)),
        'xgb__n_estimators' : list(range(50, 400, 50)),
        'xgb__reg_alpha' : list(np.logspace(-1, 1, num=10)/10),
        'xgb__reg_lambda' : list(np.logspace(-1, 1, num=10)/10),
        'xgb__subsample' : [0.6, 0.7, 0.8, 1.0],
        'xgb__scale_pos_weight' : [1, 10, 25, 50, 75, 99, 100, 1000, 3000,5000, 10000]
}


lgbm_param_grid = {
    'lgbm__boosting_type': ['gbdt', 'goss', 'dart'],
    'lgbm__num_leaves': list(range(20, 150)),
    'lgbm__learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'lgbm__subsample_for_bin': list(range(20000, 300000, 20000)),
    'lgbm__min_child_samples': list(range(20, 500, 5)),
    'lgbm__reg_alpha': list(np.linspace(0, 1)),
    'lgbm__reg_lambda': list(np.linspace(0, 1)),
    'lgbm__colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'lgbm__subsample': list(np.linspace(0.5, 1, 100)),
    'lgbm__is_unbalance': [True, False],
    'lgbm__class_weight': [None, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}, 'balanced']
}


lr_param_grid = {
  'lr__C': np.logspace(-3, 2, 6),
  #'lr__dual': False, 
  #'lr__fit_intercept': True,
  'lr__intercept_scaling': [1], 
  #'lr__l1_ratio': None, 
  'lr__max_iter': [100],
  #'lr__multi_class': 'warn',
  #'lr__n_jobs': None, 
  'lr__penalty':  ['l1', 'l2'],
  #'lr__random_state': None, 
  'lr__solver': ['liblinear'], #good for small datasets
  'lr__tol': [0.0001],
  #'lr__verbose': 0,
  #'lr__warm_start': False,
  'lr__class_weight': [None, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}, 'balanced']
}





sgd_param_grid = {
  'sgd__alpha' : np.logspace(-4, 3, 8), 
  #'sgd__average' : [False], 
  #'sgd__class_weight' : [None],
  #'sgd__early_stopping' : [False], 
  #'sgd__epsilon' : [0.1], 
  #'sgd__eta0' : [0.0], 
  #'sgd__fit_intercept' : [True],
  #'sgd__l1_ratio' : [0.15], 
  #'sgd__learning_rate' : ['optimal'], 
  'sgd__loss' : ['log','hinge'],
  'sgd__max_iter' : [1000], 
  #'sgd__n_iter_no_change' : [5], 
  'sgd__n_jobs' : [-1], 
  'sgd__penalty' : ['l2', 'l1', 'elasticnet'],
  #'sgd__power_t' : [0.5], 
  #'sgd__random_state' : [None], 
  #'sgd__shuffle' : [True], 
  'sgd__tol' : [0.001],
  #'sgd__validation_fraction' : [0.1], 
  #'sgd__verbose' : [0], 
  #'sgd__warm_start' : [False],
  'sgd__class_weight': [None, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}, 'balanced']
}




knn_param_grid =  {
  #'knn__algorithm' : ['auto'], 
  'knn__leaf_size' : list(range(2, 40, 2)), 
  'knn__metric' : ['euclidean', 'manhattan'],
  #'knn__metric_params' : [None], 
  #'knn__n_jobs' : [None], 
  'knn__n_neighbors' : list(range(2, 18, 2)), 
  'knn__p' : [2,3],
  'knn__weights' : ['uniform', 'distance']
}




rf_param_grid =  {
    'rf__bootstrap' : [True, False],
    #'rf__ccp_alpha' : 0.0, 
    'rf__class_weight': [None, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}, 'balanced'],
    #'rf__criterion' : 'gini', 
    'rf__max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], 
    'rf__max_features' : ['auto', 'sqrt'],
    #'rf__max_leaf_nodes' : None,
    #'rf__max_samples' : None,
    #'rf__min_impurity_decrease' : 0.0, 
    #'rf__min_impurity_split' : None,
    'rf__min_samples_leaf' : [1, 2, 4],
    'rf__min_samples_split' : [2, 5, 10],
    #'rf__min_weight_fraction_leaf' : 0.0, 
    'rf__n_estimators' : [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #'rf__n_jobs' : None, 
    #'rf__oob_score' : False, 
    #'rf__random_state' : None,
    #'rf__verbose' : 0, 
    #'rf__warm_start' : False
}






