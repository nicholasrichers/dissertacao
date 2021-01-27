

#Boa referencia
#https://forum.numer.ai/t/model-diagnostics-risk-metrics/900

def  get_metrics_dicts(values):


    #OK
    dict_Model_Name = {"Metrica": "Model_Name", 
                        "Valor":  values['Model_Name'] ,
                        "Categoria": "Name", 
                        "Range_Aceitavel": "NA", 
                        "Descricao": "Nome do Modelo" }

    #OK 
    dict_Max_Drawdown = {"Metrica": 'Max_Drawdown', 
                         "Valor": values['Max_Drawdown'], 
                         "Categoria": "Risk", 
                         "Range_Aceitavel": "[0.004%..2%]", 
                         "Descricao": "Perda máxima em uma era em relação a anterior" }

    #OK
    dict_Avg_corr = {"Metrica": 'Validation_Mean', 
                     "Valor": values['Validation_Mean'], 
                     "Categoria": "Performance", 
                     "Range_Aceitavel": "[3.6%..4.4%] >4.8%, overfitting", 
                     "Descricao": "Média spearman corr por era" }

    #OK
    dict_Median_corr = {"Metrica": 'Median_corr', 
                        "Valor": values['Median_corr'], 
                        "Categoria": "Estatistica", 
                        "Range_Aceitavel": "[3.6%..4.4%] Dentro do range de médias", 
                        "Descricao": "Mediana spearman corr por era" }

    #OK
    dict_Variance = {"Metrica": 'Variance', 
                     "Valor":  values['Variance'], 
                     "Categoria": "Estatistica", 
                     "Range_Aceitavel": "[0..] Próximo a zero", 
                     "Descricao": "Variancia spearman corr por era" }

    #OK
    dict_Std_Dev = {"Metrica": 'Validation_SD', 
                    "Valor": values['Validation_SD'], 
                    "Categoria": "Risk", 
                    "Range_Aceitavel": "[0..] Próximo a zero", 
                    "Descricao": "Std spearman corr por era" }

    #OK
    dict_AR1 = {"Metrica": 'AR(1)', 
                "Valor": values['AR(1)'], 
                "Categoria": "Estatistica", 
                "Range_Aceitavel": "[0..] Próximo a zero", 
                "Descricao": "AR(1) pearson corr eras[:-1] e eras[1:]" }

    #OK
    dict_Skewness = {"Metrica": 'Skewness', 
                     "Valor":   values['Skewness'], 
                     "Categoria": "Estatistica", 
                     "Range_Aceitavel": "[0..1]", 
                     "Descricao": "Skewness, central moment **3" }

    #OK
    dict_Exc_Kurtosis = {"Metrica": 'Exc_Kurtosis', 
                         "Valor": values['Exc_Kurtosis'], 
                         "Categoria": "Estatistica", 
                         "Range_Aceitavel": "[-1..0]", 
                         "Descricao": "Kurtosis central moment **4" }


    #OK
    dict_Std_Error_Mean = {"Metrica": 'Std_Error_Mean', 
                           "Valor": values['Std_Error_Mean'], 
                           "Categoria": "Estatistica", 
                           "Range_Aceitavel": "[0..] Menor que o std", 
                           "Descricao": "std/sqrt(n) measures how far the sample mean of the data is likely to be from the true population mean." }

    #OK
    dict_Sharpe = {"Metrica": 'Validation_Sharpe', 
                  "Valor": values['Validation_Sharpe'], 
                  "Categoria": "Performance", 
                  "Range_Aceitavel": "Post[>1 good, >1.2 very good] > 1.5 Overfit", 
                  "Descricao": "Média da correlacoa por era / std corr por era, anualizado" }




    #OK
    dict_Smart_Sharpe = {"Metrica": "Smart_Sharpe", 
                         "Valor": values['Smart_Sharpe'], 
                         "Categoria": "Financeira", 
                         "Range_Aceitavel": "Pouco menor que o shape", 
                         "Descricao": "Sharpe * auto_corr_penalty" }

    #OK
    dict_Numerai_Sharpe = {"Metrica": "Numerai_Sharpe", 
                          "Valor": values['Numerai_Sharpe'], 
                          "Categoria": "Financeira", 
                          "Range_Aceitavel": "Pouco menor que o sharpe", 
                          "Descricao": "sharpe menos transaction costs" }

      #OK
    dict_ann_Sharpe = {"Metrica": "Annual_Sharpe", 
                          "Valor": values['Ann_Sharpe'], 
                          "Categoria": "Financeira", 
                          "Range_Aceitavel": "Pouco menor que o sharpe", 
                          "Descricao": "sharpe menos transaction costs * sqrt 12" }



    #OK
    dict_Adj_Sharpe = {"Metrica": 'Adj_Sharpe', 
                       "Valor": values['Adj_Sharpe'], 
                       "Categoria": "Special", 
                       "Range_Aceitavel": "Prox ao sharpe", 
                       "Descricao": "Adjusted Sharpe Ratio adjusts for skewness and kurtosis by incorporating a penalty factor for negative skewness and excess kurtosis" }

 #OK
    dict_Prob_Sharpe = {"Metrica": 'Prob_Sharpe', 
                       "Valor": values['Prob_Sharpe'], 
                       "Categoria": "Special", 
                       "Range_Aceitavel": ">90%", 
                       "Descricao": "Prob so sharpe real ser maior que 0 (benchmark)" }



    #ok
    dict_VaR_10 = {"Metrica": 'VaR_10', 
                    "Valor": values['VaR_10%'], 
                    "Categoria": "Financeira", 
                    "Range_Aceitavel": "[-0.1..0]", 
                    "Descricao": "Identificar exposicao ao risco || fornece um intervalo de confiança sobre a probabilidade de exceder um certo limite de perda." }




    #OK    
    dict_Sortino_Ratio = {"Metrica": 'Sortino_Ratio', 
                          "Valor": values['Sortino_Ratio'], 
                          "Categoria": "Financeira", 
                          "Range_Aceitavel": "Prox ao sharpe [5..25]", 
                          "Descricao": "Sortino ratio as an alternative to Sharpe for doing hyperparameter selection, \
                                        because it makes sense to me to only penalize downside volatility/variance" }

                         

    #OK
    dict_Smart_Sortino_Ratio = {"Metrica": 'Smart_Sortino_Ratio', 
                                "Valor": values['Smart_Sortino_Ratio'], 
                                "Categoria": "Financeira", 
                                "Range_Aceitavel": "Pouco abaixo do sortino", 
                                "Descricao":  "Sortino Ratio penalizado" }

    #OK
    dict_Payout = {"Metrica": 'Payout', 
                   "Valor": values['Payout'], 
                   "Categoria": "Financeira", 
                   "Range_Aceitavel": "[3.6%..4.4%] ", 
                   "Descricao": "Retira as 25% piores e melhores eras corr" }

    #OK
    dict_Feat_exp_std = {"Metrica": 'Feat_exp_std', 
                       "Valor": values['Feat_exp_std'], 
                       "Categoria": "Estatistica", 
                       "Range_Aceitavel": "[8%..x%]", 
                       "Descricao": "Std das features exposures, example_preds 0.0765" }

    #OK
    dict_Feat_exp_max = {"Metrica": 'Feat_exp_max', 
                       "Valor": values['Feat_exp_max'], 
                       "Categoria": "Risk", 
                       "Range_Aceitavel": "[0.08..0.15]", 
                       "Descricao": "feature com maior exp" }

    #OK
    dict_Feat_neutral_mean = {"Metrica": 'Feat_neutral_mean', 
                       "Valor": values['Feat_neutral_mean'], 
                       "Categoria": "Performance", 
                       "Range_Aceitavel": "[0.00027..0.027]", 
                       "Descricao": "Score médio com feature neutral = 1.0" }



    #
    dict_val_mmc = {"Metrica": 'val_mmc_mean', 
                       "Valor": values['val_mmc_mean'], 
                       "Categoria": "MMC", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": ".." }




    #
    dict_corr_mmc = {"Metrica": 'corr_plus_mmc_sharpe', 
                       "Valor": values['corr_plus_mmc_sharpe'], 
                       "Categoria": "MMC", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": "" }



    #
    dict_corr_ex_preds = {"Metrica": 'corr_with_example_preds', 
                       "Valor": values['corr_with_example_preds'], 
                       "Categoria": "MMC", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": ".." }


    dict_val_mmc_FN = {"Metrica": 'val_mmc_mean_FN', 
                       "Valor": values['val_mmc_mean_FN'], 
                       "Categoria": "MMC_FN", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": ".." }




    #
    dict_corr_mmc_FN = {"Metrica": 'corr_plus_mmc_sharpe_FN', 
                       "Valor": values['corr_plus_mmc_sharpe_FN'], 
                       "Categoria": "MMC_FN", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": "" }



    #
    dict_corr_ex_FN100 = {"Metrica": 'corr_with_ex_FN100', 
                       "Valor": values['corr_with_ex_FN100'], 
                       "Categoria": "MMC_FN", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": ".." }


    
    metrics_dicts_list = [dict_Model_Name

                              #Performance
                             ,dict_Sharpe
                             ,dict_Avg_corr
                             ,dict_Feat_neutral_mean

                              #Risk
                             ,dict_Std_Dev
                             ,dict_Feat_exp_max
                             ,dict_Max_Drawdown

                             #MMC
                             ,dict_corr_mmc
                             ,dict_val_mmc #OK
                             ,dict_corr_ex_preds #OK

                             #MMC_FN
                             ,dict_corr_mmc_FN
                             ,dict_val_mmc_FN
                             ,dict_corr_ex_FN100


                             #Special
                             ,dict_Adj_Sharpe
                             ,dict_Prob_Sharpe
                             #dict_dsr por fora
                             



                              #Financeira
                             ,dict_Smart_Sharpe
                             ,dict_Numerai_Sharpe
                             ,dict_ann_Sharpe
                             ,dict_Payout
                             ,dict_VaR_10
                             ,dict_Sortino_Ratio
                             ,dict_Smart_Sortino_Ratio
                             

                             #Estatistica
                             ,dict_Median_corr
                             ,dict_Variance
                             ,dict_AR1
                             ,dict_Skewness
                             ,dict_Exc_Kurtosis
                             ,dict_Std_Error_Mean
                             ,dict_Feat_exp_std



                             ]


    return metrics_dicts_list







