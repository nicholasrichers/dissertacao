#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 03:32:15 2020

@author: nicholasrichers
"""

##########
# File: baseline_model.py
# Description:
#    Test Harness Modelos ensemble recursivos
##########



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 02:58:28 2020

@author: nicholasrichers
"""

##########
# File: baseline_model.py
# Description:
#    Test Harness Modelos ensemble recursivos
##########


# recursive multi-step forecast with linear algorithms
from math import sqrt
from numpy import split
from numpy import array
from numpy import log, std
from numpy import concatenate
from pandas import read_csv
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

from numpy import mean
def mean_absolute_percentage_error(y_true, y_pred): 
    return mean(abs((y_true - y_pred) / y_true)) * 100


def split_dataset(data):
	# split into standard periods
	train, test = data[2:-40], data[-40:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/8))
	test = array(split(test, len(test)/8))
	return train, test

def get_models_(models=dict()):
	# non-linear models
	models['knn'] = KNeighborsRegressor(n_neighbors=8)
	models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
	# # ensemble models
	n_trees = 100 #500
	#models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	models['bag'] = BaggingRegressor(n_estimators=n_trees)
	models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	#models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
    
	models['xgb'] = XGBRegressor(max_depth=8, n_estimators=n_trees,
                                 min_child_weight=300, colsample_bytree=0.8, 
                                 subsample=0.8, eta=0.3, 
                                 seed=42, silent=True)
    

	models['lgbm'] = LGBMRegressor(n_jobs=-1,        random_state=0, 
                                   n_estimators=n_trees, learning_rate=0.001, 
                                   num_leaves=2**6,  subsample=0.9, 
                                   subsample_freq=1, colsample_bytree=1.)
    



# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	actual, predicted = actual.reshape(-1,1), predicted.reshape(-1,1)
	scores = [sqrt(mean_squared_log_error(act, pred)) for act, pred in zip(actual, predicted)]
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s+= (( log(actual[row, col]) - log(predicted[row, col]))**2)
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores


# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f]' % (name, score))#, s_scores))
#-------



def get_models(models=dict()):
	# linear models
	#models['lr'] = LinearRegression()
	#models['lasso'] = Lasso()
	#models['ridge'] = Ridge()
	#models['en'] = ElasticNet()
	#models['huber'] = HuberRegressor()
	models['svmr'] = SVR()
	#models['knn'] = KNeighborsRegressor(n_neighbors=8)
	#models['lars'] = Lars()
	#models['llars'] = LassoLars()
	#models['sgd'] = SGDRegressor(max_iter=1000000, tol=1e-3)
	#models['pa'] = PassiveAggressiveRegressor(max_iter=1000000, tol=1e-3)
	#models['ranscac'] = RANSACRegressor()
	models['lgbm'] = LGBMRegressor(n_jobs=-1,        random_state=0, 
									n_estimators=1000, learning_rate=0.001, 
									num_leaves=2**6,  subsample=0.9, 
									subsample_freq=1, colsample_bytree=1.)


	print('Defined %d models' % len(models))
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# make a recursive multi-step forecast
def forecast(model, input_x, n_input):
	yhat_sequence = list()
	input_data = [x for x in input_x]
	for j in range(8):
		# prepare the input data
		X = array(input_data[-n_input:]).reshape(1, n_input)
		# make a one-step forecast
		yhat = model.predict(X)[0]
		# add to the result
		yhat_sequence.append(yhat)
		# add the prediction to the input
		input_data.append(yhat)
	return yhat_sequence

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:,0] for week in data] #week[:, 0]
	features = [week[:,1:] for week in data] #week[:, 0]
	# flatten into a single series
	series = array(series).flatten()
	features = array(features).reshape(series.shape[0], 17)
	return series, features




# convert history into inputs and outputs
def to_supervised(history, n_input, output_ix):
	# convert history to a univariate series
	data, features = to_series(history)
	X, y = list(), list()
	ix_start = 0
	# step over the entire history one time step at a time
	for i in range(len(data)):
		# define the end of the input sequence
		ix_end = ix_start + n_input
		ix_output = ix_end + output_ix
		# ensure we have enough data for this instance
		if ix_output < len(data):
			lags = data[ix_start:ix_end]
			feat = features[ix_end-1, :]
			X.append(concatenate((lags, feat), axis=0))
			y.append(data[ix_output])
		# move along one time step
		ix_start += 1
	return array(X), array(y)



# fit a model and make a forecast
def sklearn_predict(model, history, n_input):
	yhat_sequence = list()
	# fit a model for each forecast day
	for i in range(8):
		# prepare data
		train_x, train_y = to_supervised(history, n_input, i)
		# make pipeline
		pipeline = make_pipeline(model)
		# fit the model
		pipeline.fit(train_x, train_y)
		# forecast
		x_input = array(train_x[-1, :]).reshape(1,train_x.shape[1])
		yhat = pipeline.predict(x_input)[0]
		# store
		yhat_sequence.append(yhat)
	return yhat_sequence



# evaluate a single model
def evaluate_model(model, train, test, n_input):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = sklearn_predict(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores



import pandas as pd
def seasonal_plot(df, ini, fim, frq):
  groups = df[ini:fim].groupby(pd.Grouper(freq=frq))
  pyplot.figure()
  i = 1
  n_groups = len(groups)
  for name, group in groups:
    pyplot.subplot((n_groups*100) + 10 + i)
    i += 1
    pyplot.plot(group,  marker='o')
  #pyplot.legend()
  pyplot.tight_layout()
  #pyplot.show()




#--------
def seasonal_error_recursive(dataset):
    # split into train and test
    train, test = split_dataset(dataset.values)
    # prepare the models to evaluate
    models = get_models()
    n_input = 4
    # evaluate each model
    weeks = dataset.index[-38:]#.strftime('%y-%m-%d')# ["Wk" + str(i) for i in range(1,9)]
    results = dict()
    for name, model in models.items():
        # evaluate and get scores
        score, scores = evaluate_model(model, train, test, n_input)
        results[name] = scores[2:]
        # summarize scores
        summarize_scores(name, score, scores)
        # plot scores
        #pyplot.plot(weeks.strftime('%m-%d'), scores, marker='o', label=name)
    # show plot
    seasonal_plot(pd.DataFrame(results, index = weeks), weeks[0],weeks[-1],frq="QS" )
    pyplot.tight_layout()
    #pyplot.legend()
    pyplot.show()
        
    return results
    
#'''

if __name__ == '__main__':
    ###### Setup
    REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
    dataset = read_csv(REPO_URL + 'trainDF.csv', sep=',',
                       infer_datetime_format=True,
                       parse_dates=['Datetime'],
                       index_col=['Datetime'])
    
    #Xt = Transform_Dataset(dataset)
    #Xt.decompose()
    #Xt.compose(get_df(),compose_values.resid)
    #Xt.df.head(4)
    
    
    model_scores = seasonal_error(dataset)
#'''




#'''
