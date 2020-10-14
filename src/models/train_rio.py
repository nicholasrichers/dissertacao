%load_ext autoreload
%autoreload 2

#basics
import sys,os
sys.path.insert(1, os.path.dirname(os.getcwd()))

#utils
#import paths

#main libraries
import numpy as np
import pandas as pd

#data
from src.data import make_dataset



df_training,features,target = make_dataset.get_data(nrows=5000,
                                                    low_memory=False, 
                                                    dataset="training", 
                                                    feather=False) #AWS



print(memory_usage.memory())


#Group K-fold
CV = GroupKFold(n_splits = 3)
grp = list(CV.split(X = df_training[features], y = df_training[target],  groups = df_training.era.values))







#salvando o pipeline completo
file_path = '../../models/NR__Sao_Paulo.pkl'

model = list(filter(lambda x: x[1] == 'rf', results))[0][0]
model.save(file_path)