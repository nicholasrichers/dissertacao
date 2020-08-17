%%capture
#ALL TOGETHER
def pip():
  !git clone https://github.com/nicholasrichers/dissertacao.git
  

def load_library(NAME_LIB, FILE_PATH):
    from importlib.machinery import SourceFileLoader
    FILE_PATH = '/content/dissertacao/' + FILE_PATH
    somemodule = SourceFileLoader(NAME_LIB, FILE_PATH).load_module()
    
    
def get_libs():
    #folder notebooks
    #load_library('paths', 'notebooks/Baseline/paths.py')

    
    #folder src
    load_library('make_dataset', 'src/data/make_dataset.py')
    #load_library('pre_processing_macro', 'routines/pre_processing_macro.py')
    #load_library('merge_datasets', 'routines/merge_datasets.py')
    #load_library('pre_processing', 'routines/pre_processing.py')
    #load_library('seasonal_error', 'routines/seasonal_error.py')




def get_df():
  REPO_URL = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
  df = pd.read_csv(REPO_URL, nrows=500)
  #print(f'DF Shape: {df.shape}')
  return df 
    
def setup():
    pip()
    get_libs()
    df = get_df()
    print("===========================================")
    print("===========SETUP COMPLETE==================")
    print("===========================================")
    return df

import pandas as pd
X = setup()
