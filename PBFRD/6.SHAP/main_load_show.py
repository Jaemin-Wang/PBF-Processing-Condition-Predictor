import numpy as np
import pandas as pd
import shap
import warnings
import copy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from time import sleep
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D

def update(handle, orig):
	handle.update_from(orig)
	handle.set_alpha(1)
	handle.set_sizes([100])

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

xxx = 9
zzz = 1

print(features[xxx], features[zzz])

"""
shap.dependence_plot(
	(features[xxx], features[zzz]),
	shap_interaction_values, train_X,
	display_features = train_X,
	show = False
)

#plt.xlim(0,10000)
plt.show()
sys.exit(1)
#"""

new_X = []
new_shap = []
new_X2 = []
new_shap2 = []
new_X3 = []
new_shap3 = []

topl = 75
botl = 75

xmin = 0
xmax = 500
xint = 100
ymin = -0.3
ymax = 0.3
yint = 0.1

for i in range(len(shap_interaction_values)):
	infp = features[zzz]
	if train_X[features[xxx]][train_X.index[i]] > xmax:
		continue
	if train_X[features[xxx]][train_X.index[i]] < xmin:
		continue
	if train_X[infp][train_X.index[i]] >= topl:
		new_X.append(train_X[features[xxx]][train_X.index[i]])
		new_shap.append(shap_interaction_values[i][xxx][xxx])
	elif train_X[infp][train_X.index[i]] >= botl: 
		new_X2.append(train_X[features[xxx]][train_X.index[i]])
		new_shap2.append(shap_interaction_values[i][xxx][xxx])
	else:
		new_X3.append(train_X[features[xxx]][train_X.index[i]])
		new_shap3.append(shap_interaction_values[i][xxx][xxx])

new_X = np.array(new_X)
new_shap = np.array(new_shap)
#new_X2 = np.array(new_X2)
#new_shap2 = np.array(new_shap2)
new_X3 = np.array(new_X3)
new_shap3 = np.array(new_shap3)
new_df = pd.DataFrame(np.concatenate((new_X.reshape(-1,1), new_shap.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])
#new_df2 = pd.DataFrame(np.concatenate((new_X2.reshape(-1,1), new_shap2.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])
new_df3 = pd.DataFrame(np.concatenate((new_X3.reshape(-1,1), new_shap3.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])

plt.rcParams.update({'axes.linewidth':2})
fig, ax = plt.subplots(figsize=(14,10))
plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9, wspace=0, hspace=0)
ax.tick_params(width=2, length=8, pad=12)
plt.rcParams.update({'font.size': 21})

sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df3, line_kws={'color': 'blue', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'blue', 's':30}, ax=ax, label = features_n[zzz] + " < " + str(botl) + " " + features_u[zzz])
#sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df2, line_kws={'color': 'black', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'black', 's':30}, ax=ax, label = str(botl) + " " + features_u[zzz] + " < " + features_n[zzz] + " ≤ " + str(topl) + " " + features_u[zzz] )
#sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df, line_kws={'color': 'red', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'red', 's':30}, ax=ax, label = str(topl) + " " + features_u[zzz] + " ≤ " + features_n[zzz])
sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df, line_kws={'color': 'red', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'red', 's':30}, ax=ax, label = features_n[zzz] + " ≥ " + str(botl) + " " + features_u[zzz])
ax.legend(handler_map={PathCollection : HandlerPathCollection(update_func = update), plt.Line2D : HandlerLine2D(update_func = update)}, loc = 'upper right')

plt.xlabel(features_n[xxx] + ' (' + features_u[xxx] + ')')
plt.ylabel('SHAP Value of ' + features_n[xxx])
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.axis([xmin,xmax,ymin,ymax])
plt.xticks(np.arange(xmin, xmax+0.01, xint))
plt.yticks(np.arange(ymin, ymax+0.01, yint))
plt.hlines(0, xmin, xmax, color = 'black', linestyle='--', linewidth=3)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
	item.set_fontsize(33)
for item in ([] + ax.get_xticklabels() + ax.get_yticklabels()):
	item.set_fontsize(27)
plt.show()
#plt.savefig('dep_' + features_s[xxx] + '_' + features_s[zzz] + '.png')
