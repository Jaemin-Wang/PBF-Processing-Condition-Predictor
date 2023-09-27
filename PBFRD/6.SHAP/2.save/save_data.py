import numpy as np
import pandas as pd
import shap
import warnings
import copy
import sys
from time import sleep
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import train_test_split

features = ['reflectivity (%)','thermal conductivity (W/mK)','specific heat capacity (J/gK)','density(g/cm3)','melting point (C)','laser power (watt)','scan speed (mm/s)','layer thickness (mm)','hatch spacing (mm)','energy density (J/mm3)']
features_n = ['Reflectivity','Thermal Conductivity','Specific Heat Capacity','Density','Melting Point','Laser Power','Scan Speed','Layer Thickness','Hatch Spacing','Energy Density']
features_u = ['%','W/mK','J/gK','g/cm$^3$','$^o$C','W','mm/s','mm','mm','J/mm$^3$']
features_s = ['reflect','cond','heat','density','melt','power','speed','thick','hatch','energy']

data = pd.read_csv("data.csv")
data = data.dropna(axis = 0)
		
train, test = train_test_split(data, test_size=0.15, random_state = 95)
train, val = train_test_split(train, test_size=0.15, random_state = 95)
train_X, train_y = np.split(train, [-1], axis=1)
test_X, test_y = np.split(test, [-1], axis=1)
val_X, val_y = np.split(val, [-1], axis=1)

shap_interaction_values = np.load('shap_int_val.npy')

xxx = 6 ###
zzz = 3 ###

print(features[xxx], features[zzz])

new_X = []
new_shap = []
new_X2 = []
new_shap2 = []
new_X3 = []
new_shap3 = []

topl = 6.5 ###
botl = 6.5 ###

xmin = 100 ###
xmax = 2100 ###

for i in range(len(shap_interaction_values)):
	infp = features[zzz]
	#if train_X[features[xxx]][train_X.index[i]] > xmax:
	#	continue
	#if train_X[features[xxx]][train_X.index[i]] < xmin:
	#	continue
	if train_X[infp][train_X.index[i]] >= topl:
		new_X.append(train_X[features[xxx]][train_X.index[i]])
		new_shap.append(shap_interaction_values[i][xxx][zzz]*2)
	#elif train_X[infp][train_X.index[i]] >= botl: 
	#	new_X2.append(train_X[features[xxx]][train_X.index[i]])
	#	new_shap2.append(shap_interaction_values[i][xxx][zzz]*2)
	else:
		new_X3.append(train_X[features[xxx]][train_X.index[i]])
		new_shap3.append(shap_interaction_values[i][xxx][zzz]*2)

new_X = np.array(new_X)
new_shap = np.array(new_shap)
smoothed = lowess(new_shap, new_X)
df_smoothed = pd.DataFrame({'lowess high ' + features[zzz] + ' and ' + features[xxx]: smoothed[:, 0], 'lowess shap': smoothed[:, 1]})
'''
new_X2 = np.array(new_X2)
new_shap2 = np.array(new_shap2)
smoothed = lowess(new_shap2, new_X2)
df_smoothed2 = pd.DataFrame({'lowess medium ' + features[zzz] + ' and ' + features[xxx]: smoothed[:, 0], 'lowess shap': smoothed[:, 1]})
'''
new_X3 = np.array(new_X3)
new_shap3 = np.array(new_shap3)
smoothed = lowess(new_shap3, new_X3)
df_smoothed3 = pd.DataFrame({'lowess low ' + features[zzz] + ' and ' + features[xxx]: smoothed[:, 0], 'lowess shap': smoothed[:, 1]})

new_df = pd.DataFrame(np.concatenate((new_X.reshape(-1,1), new_shap.reshape(-1,1)), axis=1), columns = ['high ' + features[zzz] + ' and ' + features[xxx], 'shap'])
#new_df2 = pd.DataFrame(np.concatenate((new_X2.reshape(-1,1), new_shap2.reshape(-1,1)), axis=1), columns = ['medium ' + features[zzz] + ' and ' + features[xxx], 'shap'])
new_df3 = pd.DataFrame(np.concatenate((new_X3.reshape(-1,1), new_shap3.reshape(-1,1)), axis=1), columns = ['low ' + features[zzz] + ' and ' + features[xxx], 'shap'])

# Create a DataFrame
conpd = pd.concat([new_df, df_smoothed], axis=1)
#conpd = pd.concat([conpd, new_df2, df_smoothed2], axis=1)
conpd = pd.concat([conpd, new_df3, df_smoothed3], axis=1)

conpd.to_csv(features_s[xxx] + '_' + features_s[zzz] + '.csv', index=False)
