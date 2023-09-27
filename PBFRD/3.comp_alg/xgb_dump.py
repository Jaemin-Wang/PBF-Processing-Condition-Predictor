import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.linear_model import LassoCV , ElasticNetCV , RidgeCV, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import sklearn
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

warnings.filterwarnings(action='ignore', category=UserWarning)

ysname = 'ys.bin'

droplist = []

models = []
models.append(RandomForestRegressor())
models.append(DecisionTreeRegressor())
models.append(KNeighborsRegressor())
models.append(MLPRegressor())
models.append(LinearRegression())
models.append(Ridge())
models.append(Lasso())
models.append(ElasticNet())
models.append(SGDRegressor())
models.append(LassoCV())
models.append(ElasticNetCV())
models.append(RidgeCV())
models.append(BayesianRidge())
models.append(PLSRegression())
models.append(KernelRidge())
models.append(GaussianProcessRegressor())
models.append(SVR())
models.append(xgboost.XGBRegressor())

modelnames = []
modelnames.append("RandomForestRegressor")
modelnames.append("DecisionTreeRegressor")
modelnames.append("KNeighborsRegressor")
modelnames.append("MLPRegressor")
modelnames.append("LinearRegression")
modelnames.append("Ridge")
modelnames.append("Lasso")
modelnames.append("ElasticNet")
modelnames.append("SGDRegressor")
modelnames.append("LassoCV")
modelnames.append("ElasticNetCV")
modelnames.append("RidgeCV")
modelnames.append("BayesianRidge")
modelnames.append("PLSRegression")
modelnames.append("KernelRidge")
modelnames.append("GaussianProcessRegressor")
modelnames.append("SVR")
modelnames.append("XGBRegressor")

multstd = 1
modeliter = 0
for model in models:
	iternum = 100
	sumtest = 0
	for iteri in range(iternum):
		for r_state in range(118,119):
			data = pd.read_csv("data.csv")
			data = data.drop(data.columns[droplist], axis = 1)
			data = data.dropna(axis = 0)
			data = data.to_numpy()
		
			train, test = train_test_split(data, test_size=0.15, random_state = iteri)
			train, val = train_test_split(train, test_size=0.15, random_state = iteri)
			train_X, train_y = np.split(train, [-1], axis=1)
			test_X, test_y = np.split(test, [-1], axis=1)
			val_X, val_y = np.split(val, [-1], axis=1)
	
			mean = train_X.mean(axis=0)
			std = train_X.std(axis=0)
			impstd = 98
			
			for i in range(len(train_y)):
				train_y[i][0] -= impstd
				train_y[i][0] *= multstd
				train_y[i][0] = 1 / (1 + math.exp(-train_y[i][0]))
			for i in range(len(val_y)):
				val_y[i][0] -= impstd
				val_y[i][0] *= multstd
				val_y[i][0] = 1 / (1 + math.exp(-val_y[i][0]))
			for i in range(len(test_y)):
				test_y[i][0] -= impstd
				test_y[i][0] *= multstd
				test_y[i][0] = 1 / (1 + math.exp(-test_y[i][0]))
			
	
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
		
			#model = xgboost.XGBRegressor(n_estimators=4096, learning_rate=0.01)
			#model.fit(train_X, train_y, early_stopping_rounds=50, eval_set=[(val_X, val_y)], eval_metric='rmse', verbose=False)
			try:
				model.fit(train_X, train_y, eval_set=[(val_X, val_y)], eval_metric='rmse', verbose=False)
			except:
				model.fit(train_X, train_y)
			#model.save_model(ysname)
		
		
		train_y = train_y.reshape(-1)
		val_y = val_y.reshape(-1)
		test_y = test_y.reshape(-1)
		f3 = open('traintest.csv', 'w')
		f3.write("train\n")
		pre = model.predict(train_X).flatten()
		train_y = np.log(-(train_y/(train_y - 1))) / multstd + impstd
		pre = pre.astype('float64')
		pre[pre <= 0] = 1e-10
		pre[pre >= 1] = 1-1e-10
		pre = np.log(-(pre/(pre - 1))) / multstd + impstd
		pre[pre > 100] = 100
		pre[pre < 50] = 50
		train_y[train_y < 98] = 0
		train_y[train_y >= 98] = 1
		pre[pre < 98] = 0
		pre[pre >= 98] = 1
		for i in range(len(train_y)):
			f3.write(str(train_y[i]) +"," + str(pre[i]) + '\n')
		f3.write("val\n")
		pre = model.predict(val_X).flatten()
		val_y = np.log(-(val_y/(val_y - 1))) / multstd + impstd
		pre = pre.astype('float64')
		pre[pre <= 0] = 1e-10
		pre[pre >= 1] = 1-1e-10
		pre = np.log(-(pre/(pre - 1))) / multstd + impstd
		pre[pre > 100] = 100
		pre[pre < 50] = 50
		val_y[val_y < 98] = 0
		val_y[val_y >= 98] = 1
		pre[pre < 98] = 0
		pre[pre >= 98] = 1
		for i in range(len(val_y)):
			f3.write(str(val_y[i]) +"," + str(pre[i]) + '\n')
		pre = model.predict(test_X).flatten()
		test_y = np.log(-(test_y/(test_y - 1))) / multstd + impstd
		pre = pre.astype('float64')
		pre[pre <= 0] = 1e-10
		pre[pre >= 1] = 1-1e-10
		pre = np.log(-(pre/(pre - 1))) / multstd + impstd
		pre[pre > 100] = 100
		pre[pre < 50] = 50
		test_y[test_y < 98] = 0
		test_y[test_y >= 98] = 1
		pre[pre < 98] = 0
		pre[pre >= 98] = 1
		sumtest += sum((test_y == pre) == True) / len(pre)
		for i in range(len(test_y)):
			f3.write(str(test_y[i]) + "," + str(pre[i]) + '\n')
		f3.close()
	
	print(modelnames[modeliter])
	print(sumtest / iternum)
	modeliter += 1
