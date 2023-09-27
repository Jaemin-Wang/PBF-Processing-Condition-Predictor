import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost
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
import copy

warnings.filterwarnings(action='ignore', category=UserWarning)

ysname = 'ys.bin'
elname = 'el.bin'
utsname = 'uts.bin'

def RRMSE(y_true, y_pred):
	sum  = 0
	sum2 = 0
	for i in range(len(y_pred)):
		sum = sum + (y_true[i]-y_pred[i]) ** 2
		sum2 = sum2 + y_true[i]
	return ((sum/len(y_pred)) ** 0.5)/(sum2/len(y_true))


f2 = open('eval1', 'w')

droplist = []

r2test = 0
r2test2 = 0
#while r2test < 0.85 or r2test2 < 0.8:
iternum = 100
sumtest = 0
train_true = np.zeros(9)
train_positive = np.zeros(9)
val_true = np.zeros(9)
val_positive = np.zeros(9)
test_true = np.zeros(9)
test_positive = np.zeros(9)
iteri = 95
if True:#for iteri in range(iternum):
	r2val = 0
	r2val2 = 0
	for r_state in range(118,119):
	#while r2val < 0.85 or r2val2 < 0.8:
		data = pd.read_csv("data.csv")
		data = data.drop(data.columns[droplist], axis = 1)
		data = data.dropna(axis = 0)
		data = data.to_numpy()
		#for i in range(len(data)):
		#	if data[i][4] > 98:
		#		data[i][4] = 1
		#	else:
		#		data[i][4] = 0
	
		train, test = train_test_split(data, test_size=0.15, random_state = iteri)
		train, val = train_test_split(train, test_size=0.15, random_state = iteri)
		train_X, train_y = np.split(train, [-1], axis=1)
		test_X, test_y = np.split(test, [-1], axis=1)
		val_X, val_y = np.split(val, [-1], axis=1)

		mean = train_X.mean(axis=0)
		std = train_X.std(axis=0)

		for j in range(len(train_X[0])):
			for i in range(len(train_X)):
				train_X[i][j] -= mean[j]
				if std[j] != 0:
					train_X[i][j] /= std[j]
			for i in range(len(val_X)):
				val_X[i][j] -= mean[j]
				if std[j] != 0:
					val_X[i][j] /= std[j]
			for i in range(len(test_X)):
				test_X[i][j] -= mean[j]
				if std[j] != 0:
					test_X[i][j] /= std[j]

		f = open('train_mean_std', 'w')
		wr = csv.writer(f)
		wr.writerow(mean)
		wr.writerow(std)
		f.close()

	
		#model = xgboost.XGBRegressor(n_estimators=4096, learning_rate=0.01)#, max_depth=6, min_child_weight=15, n_jobs=12)#reg_alpha=0, reg_lambda=1, n_estimators=1024, learning_rate=0.01, gamma=0, subsample=0.75, max_depth=5, min_child_weight=15)
		params = {'max_depth': 15, 'learning_rate': 0.0272, 'n_estimators': 8800, 'colsample_bytree': 1.0, 'colsample_bylevel': 0.63, 'colsample_bynode': 0.67, 'reg_lambda': 0.03, 'reg_alpha': 0.01, 'subsample': 0.75, 'min_child_weight': 2}
		model = xgboost.XGBRegressor(**params)
		model.fit(train_X, train_y, early_stopping_rounds=50, eval_set=[(val_X, val_y)], eval_metric='rmse', verbose=False)
		model.save_model(ysname)
		#model2 = xgboost.XGBRegressor(n_estimators=4096, learning_rate=0.01)#, max_depth=4, min_child_weight=20, n_jobs=12)#reg_alpha=0, reg_lambda=1, n_estimators=1024, learning_rate=0.01, gamma=0, subsample=0.75, max_depth=5, min_child_weight=15)
	
		
		#print(r_state)
		r2val = r2_score(val_y, model.predict(val_X))
		r2test = r2_score(test_y, model.predict(test_X))
		if True:#r2val > 0.9 and r2test > 0.9 and r2test2 > 0.9:
			#print("val: ",r2val)
			#print()

			#print("train: ",r2_score(train_y, model.predict(train_X)))
			r2test = r2_score(test_y, model.predict(test_X))
			#print("test: ",r2test)
	
	f3 = open('traintest.csv', 'w')
	f3.write("train\n")
	pre = model.predict(train_X)
	train_y = train_y.reshape(-1)
	for i in range(len(train_y)):
		f3.write(str(train_y[i]) +"," + str(pre[i]) + '\n')
	f3.write("val\n")
	pre = model.predict(val_X)
	val_y = val_y.reshape(-1)
	for i in range(len(val_y)):
		f3.write(str(val_y[i]) +"," + str(pre[i]) + '\n')
	f3.write("test\n")
	pre = model.predict(test_X)
	test_y = test_y.reshape(-1)
	for i in range(len(test_y)):
		f3.write(str(test_y[i]) + "," + str(pre[i]) + '\n')
	f3.close()
