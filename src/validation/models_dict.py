

models_sp = {
    'integration_test': 'ex_preds',
    'nrichers' : 'xgb_ranker_ts',
    'nick_richers' : 'lgbm_forest',
    #'nr_rio' : 'lgbm_slider20',
    'nr__rio' : 'nr__rio',
    'nr__sao_paulo' : 'nr__sao_paulo',
    #'nr__medellin' : 'nr__medellin',
}


sp_dict = {'models': models_sp, 'round': 234}




models_medellin = {
    'integration_test': 'ex_preds',
    #'nrichers' : 'xgb_ranker_ts',
    #'nick_richers' : 'lgbm_forest',
    'nr_rio' : 'lgbm_slider20',
    'nr__rio' : 'nr__rio',
    'nr__sao_paulo' : 'nr__sao_paulo',
    'nr__medellin' : 'nr__medellin',
}


medellin_dict = {'models': models_medellin, 'round': 242}