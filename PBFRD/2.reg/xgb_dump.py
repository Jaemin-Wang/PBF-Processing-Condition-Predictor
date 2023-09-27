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
for iteri in range(iternum):
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
	pre[pre < 98] = 0
	pre[pre >= 98] = 1
	#print(sum((train_y == pre) == True) / len(pre))
	for i in range(len(train_y)):
		if train_y[i] >= 99.75:
			train_true[0] += 1
			if pre[i] == 1:
				train_positive[0] += 1
		elif train_y[i] >= 99.5:
			train_true[1] += 1
			if pre[i] == 1:
				train_positive[1] += 1
		elif train_y[i] >= 99.25:
			train_true[2] += 1
			if pre[i] == 1:
				train_positive[2] += 1
		elif train_y[i] >= 99:
			train_true[3] += 1
			if pre[i] == 1:
				train_positive[3] += 1
		elif train_y[i] >= 98.75:
			train_true[4] += 1
			if pre[i] == 1:
				train_positive[4] += 1
		elif train_y[i] >= 98.5:
			train_true[5] += 1
			if pre[i] == 1:
				train_positive[5] += 1
		elif train_y[i] >= 98.25:
			train_true[6] += 1
			if pre[i] == 1:
				train_positive[6] += 1
		elif train_y[i] >= 98:
			train_true[7] += 1
			if pre[i] == 1:
				train_positive[7] += 1
		else:
			if pre[i] == 1:
				train_positive[8] += 1
		f3.write(str(train_y[i]) +"," + str(pre[i]) + '\n')
	f3.write("val\n")
	pre = model.predict(val_X)
	val_y = val_y.reshape(-1)
	pre[pre < 98] = 0
	pre[pre >= 98] = 1
	#print(sum((val_y == pre) == True) / len(pre))
	for i in range(len(val_y)):
		if val_y[i] >= 99.75:
			val_true[0] += 1
			if pre[i] == 1:
				val_positive[0] += 1
		elif val_y[i] >= 99.5:
			val_true[1] += 1
			if pre[i] == 1:
				val_positive[1] += 1
		elif val_y[i] >= 99.25:
			val_true[2] += 1
			if pre[i] == 1:
				val_positive[2] += 1
		elif val_y[i] >= 99:
			val_true[3] += 1
			if pre[i] == 1:
				val_positive[3] += 1
		elif val_y[i] >= 98.75:
			val_true[4] += 1
			if pre[i] == 1:
				val_positive[4] += 1
		elif val_y[i] >= 98.5:
			val_true[5] += 1
			if pre[i] == 1:
				val_positive[5] += 1
		elif val_y[i] >= 98.25:
			val_true[6] += 1
			if pre[i] == 1:
				val_positive[6] += 1
		elif val_y[i] >= 98:
			val_true[7] += 1
			if pre[i] == 1:
				val_positive[7] += 1
		else:
			if pre[i] == 1:
				val_positive[8] += 1
		f3.write(str(val_y[i]) +"," + str(pre[i]) + '\n')
	f3.write("test\n")
	pre = model.predict(test_X)
	test_y = test_y.reshape(-1)
	intest_y = copy.deepcopy(test_y)
	intest_y[intest_y < 98] = 0
	intest_y[intest_y >= 98] = 1
	pre[pre < 98] = 0
	pre[pre >= 98] = 1
	#print(sum((test_y == pre) == True) / len(pre))
	sumtest += sum((intest_y == pre) == True) / len(pre)
	for i in range(len(test_y)):
		if test_y[i] >= 99.75:
			test_true[0] += 1
			if pre[i] == 1:
				test_positive[0] += 1
		elif test_y[i] >= 99.5:
			test_true[1] += 1
			if pre[i] == 1:
				test_positive[1] += 1
		elif test_y[i] >= 99.25:
			test_true[2] += 1
			if pre[i] == 1:
				test_positive[2] += 1
		elif test_y[i] >= 99:
			test_true[3] += 1
			if pre[i] == 1:
				test_positive[3] += 1
		elif test_y[i] >= 98.75:
			test_true[4] += 1
			if pre[i] == 1:
				test_positive[4] += 1
		elif test_y[i] >= 98.5:
			test_true[5] += 1
			if pre[i] == 1:
				test_positive[5] += 1
		elif test_y[i] >= 98.25:
			test_true[6] += 1
			if pre[i] == 1:
				test_positive[6] += 1
		elif test_y[i] >= 98:
			test_true[7] += 1
			if pre[i] == 1:
				test_positive[7] += 1
		else:
			if pre[i] == 1:
				test_positive[8] += 1
		f3.write(str(test_y[i]) + "," + str(pre[i]) + '\n')
	f3.close()

#"""
train_recall = []
test_recall = []
val_recall = []
for i in range(8):
	train_recall.append(train_positive[i] / train_true[i])
	print(train_recall[i])
for i in range(8):
	val_recall.append(val_positive[i] / val_true[i])
	print(val_recall[i])
for i in range(8):
	test_recall.append(test_positive[i] / test_true[i])
	print(test_recall[i])
print((train_positive.sum() - train_positive[8]) / train_true.sum())
print((val_positive.sum() - val_positive[8]) / val_true.sum())
print((test_positive.sum() - test_positive[8]) / test_true.sum())
#"""

#"""
print((train_positive.sum() - train_positive[8]) / train_positive.sum())
print((val_positive.sum() - val_positive[8]) / val_positive.sum())
print((test_positive.sum() - test_positive[8]) / test_positive.sum())
print(sumtest / iternum)
#"""
