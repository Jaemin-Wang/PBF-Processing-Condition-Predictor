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
import seaborn as sns
import matplotlib.pyplot as plt

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
for multstd in [1]:#0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7, 10]:
	iternum = 100
	sumtest = 0
	train_true = np.zeros(9)
	train_positive = np.zeros(9)
	val_true = np.zeros(9)
	val_positive = np.zeros(9)
	test_true = np.zeros(9)
	test_positive = np.zeros(9)
	for iteri in range(95,iternum):
		print(iteri)
		r2val = 0
		r2val2 = 0
		for r_state in range(118,119):
			data = pd.read_csv("data.csv")
			data = data.drop(data.columns[droplist], axis = 1)
			data = data.dropna(axis = 0)
			#data = data.loc[data['layer thickness (mm)'] == 0.03]
		
			train, test = train_test_split(data, test_size=0.15, random_state = iteri)
			train, val = train_test_split(train, test_size=0.15, random_state = iteri)
			train_X, train_y = np.split(train, [-1], axis=1)
			test_X, test_y = np.split(test, [-1], axis=1)
			val_X, val_y = np.split(val, [-1], axis=1)
	
			mean = train_X.mean(axis=0)
			std = train_X.std(axis=0)
			impstd = 98
			
			for col in train_y:
				for i in train_y.index:
					tmpval = train_y._get_value(i,col)
					tmpval -= impstd
					tmpval *= multstd
					tmpval = 1 / (1 + math.exp(-tmpval))
					train_y._set_value(i, col, tmpval)
			for col in val_y:
				for i in val_y.index:
					tmpval = val_y._get_value(i,col)
					tmpval -= impstd
					tmpval *= multstd
					tmpval = 1 / (1 + math.exp(-tmpval))
					val_y._set_value(i, col, tmpval)
			for col in test_y:
				for i in test_y.index:
					tmpval = test_y._get_value(i,col)
					tmpval -= impstd
					tmpval *= multstd
					tmpval = 1 / (1 + math.exp(-tmpval))
					test_y._set_value(i, col, tmpval)
			
			train_X = (train_X - mean) / std
			val_X = (val_X - mean) / std
			test_X = (test_X - mean) / std
	
			f = open('train_mean_std', 'w')
			wr = csv.writer(f)
			wr.writerow(mean)
			wr.writerow(std)
			f.close()
	
		
			params = {'max_depth': 15, 'learning_rate': 0.0272, 'n_estimators': 8800, 'colsample_bytree': 1.0, 'colsample_bylevel': 0.63, 'colsample_bynode': 0.67, 'reg_lambda': 0.03, 'reg_alpha': 0.01, 'subsample': 0.75, 'min_child_weight': 2}
			model = xgboost.XGBRegressor()
			model.load_model('ys.bin')

			explainer = shap.TreeExplainer(model)
			shap_values = explainer.shap_values(train_X)
			shap_interaction_values = explainer.shap_interaction_values(train_X)
			#np.save('shap_int_val', shap_interaction_values)
			np.save('shap_val', shap_values)



		break
