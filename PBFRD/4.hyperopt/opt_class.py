import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import joblib
import pickle
import sys
import csv
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import pyplot
import warnings

import optuna

from sklearn import datasets
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb
import psutil
import time

warnings.filterwarnings(action='ignore', category=UserWarning)

data = pd.read_csv("data.csv")
data = data.dropna(axis = 0)
data = data.to_numpy()

train_X = []
test_X = []
val_X = []
train_y = []
test_y = []
val_y = []

iternum = 25

for iteri in range(iternum):
	train, test = train_test_split(data, test_size=0.15)
	train, val = train_test_split(train, test_size=0.15)
	traini_X, traini_y = np.split(train, [-1], axis=1)
	testi_X, testi_y = np.split(test, [-1], axis=1)
	vali_X, vali_y = np.split(val, [-1], axis=1)

	mean = traini_X.mean(axis=0)
	std = traini_X.std(axis=0)
	

	for j in range(len(traini_X[0])):
		for i in range(len(traini_X)):
			traini_X[i][j] -= mean[j]
			if std[j] != 0:
				traini_X[i][j] /= std[j]
		for i in range(len(vali_X)):
			vali_X[i][j] -= mean[j]
			if std[j] != 0:
				vali_X[i][j] /= std[j]
		for i in range(len(testi_X)):
			testi_X[i][j] -= mean[j]
			if std[j] != 0:
				testi_X[i][j] /= std[j]
	traini_y[traini_y < 98] = 0
	traini_y[traini_y >= 98] = 1
	vali_y[vali_y < 98] = 0
	vali_y[vali_y >= 98] = 1
	testi_y[testi_y < 98] = 0
	testi_y[testi_y >= 98] = 1
	
	train_X.append(traini_X)
	test_X.append(testi_X)
	val_X.append(vali_X)
	train_y.append(traini_y)
	test_y.append(testi_y)
	val_y.append(vali_y)

train_X = np.array(train_X)
test_X = np.array(test_X)
val_X = np.array(val_X)
train_y = np.array(train_y)
test_y = np.array(test_y)
val_y = np.array(val_y)

print('train_X.shape, test_X.shape, val_X.shape', train_X.shape, test_X.shape, val_X.shape)
print('train_y.shape, test_y.shape, val_y.shape', train_y.shape, test_y.shape, val_y.shape)

def objective(trial):

	params = {
		#"objective": "multi:softprob",
		"booster": 'gbtree',
		'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'gpu_id': 0, # GPU 사용시
		#"tree_method": 'exact', 'gpu_id': -1,  # CPU 사용시
		"verbosity": 0,
		#'num_class':3,
		"max_depth": trial.suggest_int("max_depth", 4, 15),
		"learning_rate": trial.suggest_float('learning_rate', 0.0001, 0.99, step=0.0001),
		'n_estimators': trial.suggest_int("n_estimators", 1000, 10000, step=100),
		"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.01),
		"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0, step=0.01),
		"colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0, step=0.01),
		"reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1, step=0.01),
		"reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1, step=0.01),
		'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),	 
		'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
		"gamma": 0#trial.suggest_float("gamma", 0, 1.0, step=0.001),
		#'num_parallel_tree': trial.suggest_int("num_parallel_tree", 1, 5)
	}

	accuracy = 0
	for i in range(iternum):
		model = xgb.XGBClassifier(**params, random_state = 1234, use_label_encoder = False)

		model.fit(train_X[i], train_y[i], early_stopping_rounds=50, eval_set=[(val_X[i], val_y[i])], eval_metric='rmse', verbose=False)
		pre = model.predict(test_X[i]).flatten()
		accuracy += sum((test_y[i].flatten() == pre) == True) / len(pre)
	
	accuracy = accuracy / iternum
	return accuracy


if __name__ == "__main__":

	train_start = time.time()

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=1000, show_progress_bar=True)

	print("Number of finished trials: ", len(study.trials))
	print("Best trial:")


	trial = study.best_trial

	print("  Accuracy: {}".format(trial.value))
	print("  Best hyperparameters: ")
	for key, value in trial.params.items():
		print("	{}: {}".format(key, value))

 
	accuracy = 0
	for i in range(iternum):  
		clf = xgb.XGBClassifier(**study.best_params, random_state = 1234, use_label_encoder = False)
		clf.fit(train_X[i], train_y[i], early_stopping_rounds=50, eval_set=[(val_X[i], val_y[i])], eval_metric='rmse', verbose=False)

		pre = clf.predict(test_X[i]).flatten()
		accuracy += sum((test_y[i].flatten() == pre) == True) / len(pre)
	
	accuracy = accuracy / iternum
	print("Accuracy: {}".format(accuracy))
