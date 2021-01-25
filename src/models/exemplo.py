

%%time
path = '../../reports/predicoes_validacao/raw/'
models_nr = ['ex_preds','ex_FN100']#, 'nr__sao_paulo', 'lgbm_exp20', 'lgbm_slider20']

preds_nr, feat_corrs_nr = dict(), dict()
era_scores_nr, df_metrics_nr = dict(), dict()



for model in models_nr[:]:
    
    #predicoes val1 & val2
    print("creating predictions to:", model)
    if (model=='ex_preds' or model=='ex_FN100'): pred_path = github_url
    else: pred_path = path
        
    df_validation['preds']=pd.read_csv(pred_path+model+'_preds_test.csv',index_col='id').values.reshape(1,-1)[0]

    #preds neutralized after
    preds_nr[model] = fn_strategy[model]['func'](df_validation,
                                                      fn_strategy[model]['columns'],
                                                      fn_strategy[model]['by'],
                                                      fn_strategy[model]['model'],
                                                      fn_strategy[model]['factor']).reshape(1,-1)[0]
    

    #salvando as metricas
    era_scores_nr[model], df_metrics_nr[model], feat_corrs_nr[model] = \
                        metrics.submission_metrics(df_validation, preds_nr[model], model, False)   


#dict to dataframe
df_preds_nr = pd.DataFrame.from_dict(preds_nr)
df_era_scores_nr = pd.DataFrame.from_dict(era_scores_nr)
df_feat_corrs_nr = pd.DataFrame.from_dict(feat_corrs_nr)
df_metrics_cons_nr = metrics.metrics_consolidated(df_metrics_nr)



############################################################################################################
############################################################################################################
############################################################################################################