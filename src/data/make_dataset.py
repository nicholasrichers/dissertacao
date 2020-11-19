# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

def get_data(nrows=None, low_memory=False, dataset="training", feather=False):


    #DOWNLOAD DATAFRAME
    if feather==True:
    	df = pd.read_feather('../../Data/Interim/'+dataset+'_compressed.feather').iloc[:nrows,:]

    elif dataset == "validation":
        data_path = '../../Data/Interim/'+dataset+'_data.csv'
        df = pd.read_csv(data_path, nrows=nrows)

    else:
    	data_path = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_'+dataset+'_data.csv.xz'
    	df = pd.read_csv(data_path, nrows=nrows)



    #Low memory
    if low_memory == True:
        print("low memory activated")
        df = reduce_mem_usage(df, verbose=True)

    
    #COLUMN NAMES
    X = [c for c in df if c.startswith("feature")]
    y = "target_kazutsugi"

    if dataset != "tournament":
        df['era'] = df.loc[:, 'era'].str[3:].astype('int32')

    #PRINT MEMORY USAGE
    print(df.info())
    return df, X, y


def get_data_nomi(nrows=None, low_memory=False, dataset="nomi", feather=False):


    #DOWNLOAD DATAFRAME
    if feather==True:
        df = pd.read_feather('../../Data/Interim/'+dataset+'_compressed.feather').iloc[:nrows,:]


    elif dataset == "nomi_colab":
        #data_path = "/content/drive/My Drive/Numerai/numerai_training_validation_target_nomi.csv"
        data_path = "/content/drive/My Drive/Numerai/training_and_val3.csv"
        df = pd.read_csv(data_path, nrows=nrows)

    else:
        data_path = '../../Data/Interim/'+dataset+'_data.csv'
        df = pd.read_csv(data_path, nrows=nrows)



    #Low memory
    if low_memory == True:
        print("low memory activated")
        df = reduce_mem_usage(df, verbose=True)

    
    #COLUMN NAMES
    X = [c for c in df if c.startswith("feature")]
    y = "target_nomi"

    if dataset != "tournament":
        df['era'] = df.loc[:, 'era'].str[3:].astype('int32')

    #PRINT MEMORY USAGE
    print(df.info())

    #split train val APENAS NOMI
    df_training = df[df.data_type=='train']
    df_validation = df[df.data_type=='validation']


    return df_training, df_validation, X, y





def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    #df.to_csv('../../data/interim/numerai_training_data_low_memory.csv')

    return df

#training_data = reduce_mem_usage(pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz"))
#training_data.head()


########################################################################################
########################################################################################
########################################################################################

import pandas as pd
import numpy as np
from joblib import dump, load
import pyarrow.feather as feather


def create_dtype():
	#download Numerai training data and load as a pandas dataframe
	TRAINING_DATAPATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
	df = pd.read_csv(TRAINING_DATAPATH, nrows=100) ## se der erro tira o nrows

	#create a list of the feature columns
	features = [c for c in df if c.startswith("feature")]

	#create a list of the column names
	col_list = ["id", "era", "data_type"]
	col_list = col_list + features + ["target_kazutsugi"]

	#create a list of corresponding data types to match the column name list
	dtype_list_back = [np.float32] * 311
	dtype_list_front = [str, str, str]
	dtype_list = dtype_list_front + dtype_list_back

	#use Python's zip function to combine the column name list and the data type list
	dtype_zip = zip(col_list, dtype_list)

	#convert the combined list to a dictionary to conform to pandas convention
	dtype_dict = dict(dtype_zip)

	#save the dictionary as a joblib file for future use
	#dump(dtype_dict, 'dtype_dict.joblib')
	return dtype_dict


#####


def create_feather_df(dataset="training"):

	#load dictionary to import data in specific data types
	dtype_dict = create_dtype()


	#Get Data
	if dataset == "validation": FILE_URL  = '../../Data/Interim/'+dataset+'_data.csv'

	else: FILE_URL  = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_'+dataset+'_data.csv.xz'
	
	#file Name	
	FILE_NAME = '../../Data/Interim/' + dataset + '_compressed.feather'


	#download Numerai training data and load as a pandas dataframe
	df = pd.read_csv(FILE_URL, dtype=dtype_dict)

	#download Numerai tournament data and load as a pandas dataframe
	TOURNAMENT_DATAPATH = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
	#df_tournament = pd.read_csv(TOURNAMENT_DATAPATH, dtype=dtype_dict)

	#save Numerai training data as a compressed feather file
	feather.write_feather(df, FILE_NAME)

	return df






