


def  get_metrics_dicts(values):



    dict_Model_Name = {"Metrica": "Model_Name", 
                        "Valor":  values['Model_Name'] ,
                        "Categoria": "Submission", 
                        "Range_Aceitavel": "NA", 
                        "Descricao": "Nome do Modelo" }


    dict_Max_Drawdown = {"Metrica": 'Max_Drawdown', 
                         "Valor": values['Max_Drawdown'], 
                         "Categoria": "Financeira", 
                         "Range_Aceitavel": "[..]", 
                         "Descricao": "Perda máxima em uma era" }


    dict_Avg_corr = {"Metrica": 'Avg_corr', 
                     "Valor": values['Avg_corr'], 
                     "Categoria": "Submission", 
                     "Range_Aceitavel": "[..]", 
                     "Descricao": "..." }


    dict_Median_corr = {"Metrica": 'Median_corr', 
                        "Valor": values['Median_corr'], 
                        "Categoria": "Estatistica", 
                        "Range_Aceitavel": "[..]", 
                        "Descricao": "..." }


    dict_Variance = {"Metrica": 'Variance', 
                     "Valor":  values['Variance'], 
                     "Categoria": "Estatistica", 
                     "Range_Aceitavel": "[..]", 
                     "Descricao": "..." }


    dict_Std_Dev = {"Metrica": 'Std_Dev', 
                    "Valor": values['Std_Dev'], 
                    "Categoria": "Estatistica", 
                    "Range_Aceitavel": "[..]", 
                    "Descricao": "..." }


    dict_AR1 = {"Metrica": 'AR(1)', 
                "Valor": values['AR(1)'], 
                "Categoria": "Estatistica", 
                "Range_Aceitavel": "[..]", 
                "Descricao": "..." }


    dict_Skewness = {"Metrica": 'Skewness', 
                     "Valor":   values['Skewness'], 
                     "Categoria": "Estatistica", 
                     "Range_Aceitavel": "[..]", 
                     "Descricao": "..." }


    dict_Exc_Kurtosis = {"Metrica": 'Exc_Kurtosis', 
                         "Valor": values['Exc_Kurtosis'], 
                         "Categoria": "Estatistica", 
                         "Range_Aceitavel": "[..]", 
                         "Descricao": "..." }


    dict_Std_Error_Mean = {"Metrica": 'Std_Error_Mean', 
                           "Valor": values['Std_Error_Mean'], 
                           "Categoria": "Estatistica", 
                           "Range_Aceitavel": "[..]", 
                           "Descricao": "..." }


    dict_Sharpe = {"Metrica": 'Sharpe', 
                  "Valor": values['Sharpe'], 
                  "Categoria": "Submission", 
                  "Range_Aceitavel": "[..]", 
                  "Descricao": "..." }





    dict_Smart_Sharpe = {"Metrica": "Smart_Sharpe", 
                         "Valor": values['Smart_Sharpe'], 
                         "Categoria": "Financeira", 
                         "Range_Aceitavel": "[..]", 
                         "Descricao": "..." }


    dict_Numerai_Sharpe = {"Metrica": "Numerai_Sharpe", 
                          "Valor": values['Numerai_Sharpe'], 
                          "Categoria": "Financeira", 
                          "Range_Aceitavel": "[..]", 
                          "Descricao": "..." }


    dict_Ann_Sharpe = {"Metrica": 'Ann_Sharpe', 
                     "Valor": values['Ann_Sharpe'], 
                     "Categoria": "Financeira", 
                     "Range_Aceitavel": "[..]", 
                     "Descricao": "..." }


    dict_Adj_Sharpe = {"Metrica": 'Adj_Sharpe', 
                       "Valor": values['Adj_Sharpe'], 
                       "Categoria": "Financeira", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": "..." }


    dict_VaR_10 = {"Metrica": 'VaR_10', 
                    "Valor": values['VaR_10%'], 
                    "Categoria": "Financeira", 
                    "Range_Aceitavel": "[..]", 
                    "Descricao": "..." }


    dict_Sortino_Ratio = {"Metrica": 'Sortino_Ratio', 
                          "Valor": values['Sortino_Ratio'], 
                          "Categoria": "Financeira", 
                          "Range_Aceitavel": "[..]", 
                          "Descricao": "Sortino ratio as an alternative to Sharpe for doing hyperparameter selection, \
                                        because it makes sense to me to only penalize downside volatility/variance" }

                         


    dict_Smart_Sortino_Ratio = {"Metrica": 'Smart_Sortino_Ratio', 
                                "Valor": values['Smart_Sortino_Ratio'], 
                                "Categoria": "Financeira", 
                                "Range_Aceitavel": "[..]", 
                                "Descricao":  "Sortino Ratio penalizado AR(1)" }


    dict_Payout = {"Metrica": 'Payout', 
                   "Valor": values['Payout'], 
                   "Categoria": "Leaderboard", 
                   "Range_Aceitavel": "[..]", 
                   "Descricao": "..." }


    dict_Feat_exp_std = {"Metrica": 'Feat_exp_std (val)', 
                       "Valor": values['Feat_exp_std'], 
                       "Categoria": "Estatistica", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": "..." }


    dict_Feat_exp_max = {"Metrica": 'Feat_exp_max (val)', 
                       "Valor": values['Feat_exp_max'], 
                       "Categoria": "Estatistica", 
                       "Range_Aceitavel": "[..]", 
                       "Descricao": "..." }




    metrics_dicts_list = [dict_Model_Name
                             ,dict_Max_Drawdown
                             ,dict_Avg_corr
                             ,dict_Median_corr
                             ,dict_Variance
                             ,dict_Std_Dev
                             ,dict_AR1
                             ,dict_Skewness
                             ,dict_Exc_Kurtosis
                             ,dict_Std_Error_Mean
                             ,dict_Sharpe
                             ,dict_Smart_Sharpe
                             ,dict_Numerai_Sharpe
                             ,dict_Ann_Sharpe
                             ,dict_Adj_Sharpe
                             ,dict_VaR_10
                             ,dict_Sortino_Ratio
                             ,dict_Smart_Sortino_Ratio
                             ,dict_Payout
                             ,dict_Feat_exp_std
                             ,dict_Feat_exp_max
                             ]


    return metrics_dicts_list







