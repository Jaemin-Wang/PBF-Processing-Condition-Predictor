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

droplist = [9]

r2test = 0
r2test2 = 0
#while r2test < 0.85 or r2test2 < 0.8:
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
		#while r2val < 0.85 or r2val2 < 0.8:
			data = pd.read_csv("data.csv")
			data = data.drop(data.columns[droplist], axis = 1)
			data = data.dropna(axis = 0)
			#data = data.to_numpy()
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
			impstd = 98
			#meany = train_y.mean(axis=0)
			#stdy = train_y.std(axis=0)
			
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
	
			#f = open('train_mean_std', 'w')
			#wr = csv.writer(f)
			#wr.writerow(mean)
			#wr.writerow(std)
			#f.close()
	
		
			model = xgboost.XGBRegressor(n_estimators=4096, learning_rate=0.01)#, max_depth=4, min_child_weight=6)#reg_alpha=0, reg_lambda=1, n_estimators=1024, learning_rate=0.01, gamma=0, subsample=0.75, max_depth=5, min_child_weight=15)
			params = {'max_depth': 15, 'learning_rate': 0.0272, 'n_estimators': 8800, 'colsample_bytree': 1.0, 'colsample_bylevel': 0.63, 'colsample_bynode': 0.67, 'reg_lambda': 0.03, 'reg_alpha': 0.01, 'subsample': 0.75, 'min_child_weight': 2}
			#model = xgboost.XGBRegressor(**params)
			model.fit(train_X, train_y, early_stopping_rounds=50, eval_set=[(val_X, val_y)], eval_metric='rmse', verbose=False)
			#model.save_model(ysname)
			#model = xgboost.XGBRegressor()
			#model.load_model('ys.bin')

			explainer = shap.TreeExplainer(model)
			#shap_values = explainer.shap_values(train_X)
			#shap_abs_values = np.abs(shap_values)
			#print(np.mean(shap_abs_values, axis = 0))
			#shap.summary_plot(shap_values, train_X, plot_type='bar')
			shap_interaction_values = explainer.shap_interaction_values(train_X)
			#sel_shap_int_val = shap_interaction_values[:,0,0].reshape(-1,1)
			#for i in range(1, shap_interaction_values.shape[1]):
			#	sel_shap_int_val = np.concatenate((sel_shap_int_val, shap_interaction_values[:,i,i].reshape(-1,1)), axis = 1)
			#train_X.columns = ['Reflectivity','Thermal Conductivity','Specific Heat Capacity','Density','Melting Point','Laser Power','Scan Speed','Layer Thickness','Hatch Spacing','Energy Density']
			#shap.summary_plot(sel_shap_int_val, train_X, show = False)#.iloc[:,5].to_frame())
			#shap.summary_plot(shap_interaction_values, train_X)
			#"""
			# Get absolute mean of matrices
			#mean_shap = np.abs(shap_interaction_values).mean(0)
			features = ['reflectivity (%)','thermal conductivity (W/mK)','specific heat capacity (J/gK)','density(g/cm3)','melting point (C)','laser power (watt)','scan speed (mm/s)','layer thickness (mm)','hatch spacing (mm)','energy density (J/mm3)']
			#df = pd.DataFrame(mean_shap,index=features,columns=features)

			# times off diagonal by 2
			#df.where(df.values == np.diagonal(df),df.values*2,inplace=True)

			# display 
			#plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
			#sns.set(font_scale=1.5)
			#sns.heatmap(df,cmap='coolwarm',annot=True,fmt='.3g',cbar=False)
			#plt.yticks(rotation=0) 
			#plt.show()
			#"""
			#sys.exit(1)

			#"""
			shap.dependence_plot(
				(features[8],features[6]),
				shap_interaction_values, train_X,
				display_features = train_X,
				show = False
			)
			plt.xlim(-2,2)
			plt.show()
			sys.exit(1)
			#"""
			#"""
			plt.xlabel('Normalized Laser Power')
			plt.ylabel('SHAP Interaction Value')
			fig = plt.gcf()
			ax = plt.gca()
			fig.axes[-1].set_ylabel('Normalized Thermal Conductivity')
			plt.tick_params(axis='x', direction='in')
			plt.tick_params(axis='y', direction='in')
			ax.spines['right'].set_visible(True)
			ax.spines['left'].set_visible(True)
			ax.spines['top'].set_visible(True)
			plt.rcParams["axes.spines.right"] = True
			plt.rcParams["axes.spines.left"] = True
			plt.rcParams["axes.spines.top"] = True
			#plt.xlim(-0.4,0.2)
			plt.tight_layout()
			#plt.axis([-2.5,3.5,-0.2,0.15])
			#plt.xticks(np.arange(-2.5, 3.6, 1))
			#plt.yticks(np.arange(-0.2, 0.16, 0.05))
			#plt.hlines(0, -2.5, 3.5, color = 'red', linestyle='--', linewidth=3)
			#plt.vlines(0, -0.45, 0.45, color = 'red', linestyle='--', linewidth=3)
			plt.show()
			#plt.savefig('dep_power_cond.png')
			#"""



		break
