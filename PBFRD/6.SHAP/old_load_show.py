import numpy as np
import pandas as pd
import shap
import warnings
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from time import sleep
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns

features = ['reflectivity (%)','thermal conductivity (W/mK)','specific heat capacity (J/gK)','density(g/cm3)','melting point (C)','laser power (watt)','scan speed (mm/s)','layer thickness (mm)','hatch spacing (mm)','energy density (J/mm3)']

data = pd.read_csv("data.csv")
data = data.dropna(axis = 0)
#data = data.loc[(data['layer thickness (mm)'] >= 0.02) & (data['layer thickness (mm)'] <= 0.04)]
#data = data.loc[data['layer thickness (mm)'] == 0.03]
		
train, test = train_test_split(data, test_size=0.15, random_state = 95)
train, val = train_test_split(train, test_size=0.15, random_state = 95)
train_X, train_y = np.split(train, [-1], axis=1)
test_X, test_y = np.split(test, [-1], axis=1)
val_X, val_y = np.split(val, [-1], axis=1)

mean = train_X.mean(axis=0)
std = train_X.std(axis=0)

shap_interaction_values = np.load('shap_int_val.npy')
shap_values = np.load('shap_val.npy')

"""
train_X = train_X.reset_index()
print(train_X[train_X[features[0]] == 65.3334][train_X[features[8]] == 0.15][train_X[features[6]] >= 1200])
tmpl = list(train_X[train_X[features[0]] == 65.3334][train_X[features[8]] == 0.15][train_X[features[6]] >= 1200].index)
for i in tmpl:
	print(train_X[features[8]][i], train_X[features[6]][i], train_X[features[9]][i], round(shap_interaction_values[i][8][6] * 2, 3), round(shap_values[i][9],3))
sys.exit(1)
"""

#"""
shap_abs = []
max_pos = []
for i in range(shap_interaction_values.shape[1]):
	shap_abs.append([])
	for j in range(i+1, shap_interaction_values.shape[2]):
		shap_num = 0
		for k in range(shap_interaction_values.shape[0]):
			shap_num += abs(shap_interaction_values[k][i][j])
		shap_abs[i].append(round(shap_num,2))# / shap_interaction_values.shape[0])
		max_pos.append([i,j,round(shap_num,2)])

	print(shap_abs[i])

max_pos = sorted(max_pos, key=lambda i:i[2])
max_pos=np.flip(np.array(max_pos))
print(max_pos)

sys.exit(1)
#"""

shap_interaction_values = shap_interaction_values.tolist()
i=0
while True:
	break
	infp = features[1]
	if train_X[infp][train_X.index[i]] >= (mean[infp] - 0 * std[infp]): #0.25
	#if train_X[infp][train_X.index[i]] <= (mean[infp] + 0 * std[infp]):
		train_X = train_X.drop(train_X.index[i])
		del shap_interaction_values[i]
	else:
		i += 1
	if len(train_X) <= i:
		break
shap_interaction_values = np.array(shap_interaction_values)
print(train_X.shape)

#train_X = (train_X - mean) / std
#val_X = (val_X - mean) / std
#test_X = (test_X - mean) / std

#9:1?
#5:4?
#8:0

"""
tmpdf = pd.DataFrame([], columns=train_X.columns)

for i in range(train_X.shape[0]):
	infp = features[6]
	infp2 = features[1]
	if train_X[infp][train_X.index[i]] < 0 and train_X[infp2][train_X.index[i]] < 0 and shap_interaction_values[i][1][6] < -0.0125: #graph val = shap int val * 2
		tmpdf.loc[train_X.index[i]] = train_X.iloc[i]# * std + mean

print(tmpdf.mean())
print(tmpdf)

sys.exit(1)
"""


"""

for xxx in range(5,10):
	for zzz in range(0,10):
		if xxx < 5 and zzz < 5:
			continue
		if xxx == zzz:
			continue
		mm=0
		mp=0
		pm=0
		pp=0
		mmn=0
		mpn=0
		pmn=0
		ppn=0
		for i in range(len(shap_interaction_values)):
			fval = train_X.iloc[i, zzz]
			xval = train_X.iloc[i, xxx]
			fval = fval * abs(xval)
			if xval < 0:#-0.25:
				if shap_interaction_values[i][xxx][zzz] < 0:
					mm += fval
					mmn += 1
				else:
					mp += fval
					mpn += 1
			if xval >= 0:#0.25:
				if shap_interaction_values[i][xxx][zzz] < 0:
					pm += fval
					pmn += 1
				else:
					pp += fval
					ppn += 1
		#if (int(mp*100) ^ int(mm *100)) >= 0 or (int(pp*100) ^ int(pm *100)) >= 0:
		#	continue
		#if (int((mp/mpn - mm/mmn)*1000) ^ int((pp/ppn - pm/pmn)*1000)) < 0:
		if mmn == 0 or mpn == 0 or pmn == 0 or ppn == 0:
			continue
		if ((int(mp*100) ^ int(mm*100)) < 0 and (int(pm*100) ^ int(mm*100)) < 0) or ((int(mp*100) ^ int(pp*100)) < 0 and (int(pm*100) ^ int(pp*100)) < 0):
			print(features[xxx], features[zzz])
			print(mp/mpn, end = '\t')
			print(pp/ppn, end = '\n')
			print(mm/mmn, end = '\t')
			print(pm/pmn, end = '\n')
"""

#sys.exit(1)
#sleep(1)

xxx = 6
zzz = 1

print(features[xxx], features[zzz])

new_X = []
new_shap = []
new_X2 = []
new_shap2 = []
new_X3 = []
new_shap3 = []
for i in range(len(shap_interaction_values)):
	infp = features[zzz]
	if train_X[infp][train_X.index[i]] >= 87:
		new_X.append(train_X[features[xxx]][train_X.index[i]])
		new_shap.append(shap_interaction_values[i][xxx][zzz])
	elif train_X[infp][train_X.index[i]] >= 20: 
		new_X2.append(train_X[features[xxx]][train_X.index[i]])
		new_shap2.append(shap_interaction_values[i][xxx][zzz])
	else:
		new_X3.append(train_X[features[xxx]][train_X.index[i]])
		new_shap3.append(shap_interaction_values[i][xxx][zzz])
new_X = np.array(new_X)
new_shap = np.array(new_shap)
new_X2 = np.array(new_X2)
new_shap2 = np.array(new_shap2)
new_X3 = np.array(new_X3)
new_shap3 = np.array(new_shap3)
#plt.scatter(new_X3, new_shap3, color='blue', alpha = 0.8, s = 15)
#plt.scatter(new_X2, new_shap2, color='black', alpha = 0.8, s = 15)
#plt.scatter(new_X, new_shap, color='red', alpha = 0.8, s = 15)
new_df = pd.DataFrame(np.concatenate((new_X.reshape(-1,1), new_shap.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])
new_df2 = pd.DataFrame(np.concatenate((new_X2.reshape(-1,1), new_shap2.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])
new_df3 = pd.DataFrame(np.concatenate((new_X3.reshape(-1,1), new_shap3.reshape(-1,1)), axis=1), columns = [features[xxx], 'shap'])

fig, ax = plt.subplots(figsize=(10,8))

sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df3, line_kws={'color': 'blue', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'blue', 's':15}, ax=ax, label = "Thermal Conductivity >= 87 W/mK")
sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df2, line_kws={'color': 'black', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'black', 's':15}, ax=ax, label = "87 W/mK > Thermal Conductivity >= 20 W/mK")
sns.regplot(x=features[xxx],y='shap', lowess=True, data=new_df, line_kws={'color': 'red', 'linewidth':5}, scatter_kws={'alpha':0.25, 'color': 'red', 's':15}, ax=ax, label = "20 W/mk > Thermal Conductivity")
ax.legend()
plt.xlim(0,3000)
plt.show()
sys.exit(1)
"""
shap.dependence_plot(
	(features[xxx], features[zzz]),
	shap_interaction_values, train_X,
	display_features = train_X,
	show = False
)
"""
plt.xlim(-2,4)
plt.show()
sys.exit(1)

print(mean)


viridis = cm.get_cmap('viridis', 12)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[-194:, :] = np.array([256/256, 0/256, 0/256, 1])#194
newcolors[:-194, :] = np.array([0/256, 0/256, 0/256, 1])#194
#newcolors[19:-194, :] = np.array([0/256, 256/256, 0/256, 1])
#newcolors[:19, :] = np.array([0/256, 0/256, 256/256, 1])
newcmp = ListedColormap(newcolors)

# Change the colormap of the artists
for fc in plt.gcf().get_children():
	for fcc in fc.get_children():
		if hasattr(fcc, "set_cmap"):
			fcc.set_cmap(newcmp)

plt.show()
sys.exit(1)

plt.xlabel('Normalized Hatch Spacing')
plt.ylabel('SHAP Interaction Value')
fig = plt.gcf()
ax = plt.gca()
fig.axes[-1].set_ylabel('Normalized Scan Speed')
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.tight_layout()
plt.axis([-2,4,-0.15,0.15])
plt.xticks(np.arange(-2, 4.1, 1))
plt.yticks(np.arange(-0.15, 0.16, 0.05))
plt.hlines(0, -2, 4, color = 'black', linestyle='--', linewidth=3)
plt.vlines(0, -0.15, 0.15, color = 'black', linestyle='--', linewidth=3)
plt.show()
#plt.savefig('dep_hatch_speed.png')

"""
plt.xlabel('Normalized Laser Power')
plt.ylabel('SHAP Interaction Value')
fig = plt.gcf()
ax = plt.gca()
fig.axes[-1].set_ylabel('Normalized Thermal Conductivity')
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.tight_layout()
plt.axis([-2.5,3.5,-0.2,0.15])
plt.xticks(np.arange(-2.5, 3.6, 1))
plt.yticks(np.arange(-0.2, 0.16, 0.05))
plt.hlines(0, -2.5, 3.5, color = 'black', linestyle='--', linewidth=3)
plt.vlines(0, -0.2, 0.15, color = 'black', linestyle='--', linewidth=3)
#plt.show()
plt.savefig('dep_power_cond.png')
"""
