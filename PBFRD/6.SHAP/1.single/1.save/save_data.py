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

xxx = 7

new_X = []
new_shap = []

xmin = 0.01
xmax = 0.1

for i in range(len(shap_interaction_values)):
	if train_X[features[xxx]][train_X.index[i]] > xmax:
		continue
	if train_X[features[xxx]][train_X.index[i]] < xmin:
		continue
	new_X.append(train_X[features[xxx]][train_X.index[i]])
	new_shap.append(shap_interaction_values[i][xxx][xxx])

new_X = np.array(new_X)
new_shap = np.array(new_shap)
new_df = pd.DataFrame(np.concatenate((new_X.reshape(-1,1), new_shap.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])

# Apply LOWESS smoothing
smoothed = lowess(new_shap, new_X)

x_smoothed = smoothed[:, 0]
y_smoothed = smoothed[:, 1]

# Create a DataFrame
df_smoothed = pd.DataFrame({'lowess ' + features[xxx]: x_smoothed, 'lowess shap': y_smoothed})

pd.concat([new_df, df_smoothed], axis=1).to_csv(features_s[xxx] + '.csv', index=False)
