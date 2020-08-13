
##########
# File: adjust_datatype.py
# Description:
#    Coleção de funções para ajustar/converter tipos de dados
##########

import numpy as np
import re
def urlify(s):

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '_', s)

    return s


def int_to_float(df):
    for col in df.select_dtypes(include=['int64']):
        df[col] = df[col].astype(float)
    return df


#Funcao que converte numericos com milhar ',' e decimal '.', para o tipo numerico PT-BR
def string_to_numeric(df, col):
    df[col] = df[col].replace(to_replace=r'[\-]', value=np.nan, regex=True)
    return df[col].replace(to_replace=r'(?<=\d)[\,]', value='', regex=True).astype(float)



def string_to_datetime(df, col):
    df[col] = df[col].replace(to_replace=r'[\-]', value=np.nan, regex=True)
    return df[col].astype('datetime64[ns]') 



def currency_to_numeric(df, col):
    df[col] = df[col].replace({'USD ':''}, regex=True)
    return string_to_numeric(df, col)


def pct_to_numeric(df, col):
    df[col] = df[col].replace(to_replace=r'[\-]', value=np.nan, regex=True)
    return df[col].replace({'%':''}, regex=True).astype(float).div(100)


def adjust_categorical(df, col):
    return df[col].replace('-', 'unknown')