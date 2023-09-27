import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import sys
import copy
import csv
import xgboost
from tensorflow.compat.v1.keras import backend as KK
from tensorflow.compat.v1 import ConfigProto
import tensorflow as tf
import os

model = xgboost.XGBRegressor()
model.load_model('ys.bin')


f = open('train_mean_std', 'r')
rdr = csv.reader(f)
rdr = list(rdr)
mean = rdr[0]
std = rdr[1]
mean = np.array(mean)
std = np.array(std)
mean = mean.astype(np.float)
std = std.astype(np.float)
f.close()

solution = []
if os.path.isfile("res.csv"):
	f = open("res.csv", "r")
	line = f.readline()
	for i in range(12):
		line = f.readline()
		line = line.split(",")[:-1]
		line = list(map(float, line))
		solution.append(line)
	f.close()
	additer = 12
else:
	additer = 24

solution=np.array(solution)
feature = ['reflectivity (%)','thermal conductivity (W/mK)','specific heat capacity (J/gK)','density(g/cm3)','melting point (C)','laser power (watt)','scan speed (mm/s)','layer thickness (mm)','hatch spacing (mm)']

for i in range(len(solution)):
	for j in range(len(solution[i])):
		solution[i][j] = (solution[i][j] - mean[j]) / std[j]
		print('{:10.5g}'.format(solution[i][j] * std[j] + mean[j]), end='\t')
	print()
print()

multstd = 1
impstd = 98

pre = model.predict(solution).flatten()
pre = pre.astype('float64')
pre[pre <= 0] = 1e-10
pre[pre >= 1] = 1-1e-10
print(pre)

f = open('res2.csv', 'w')
for j in range(len(feature)):
	f.write(feature[j])
	f.write(',')
f.write("relative density (%)")
f.write('\n')
for i in range(len(solution)):
	for j in range(len(solution[i])):
		print('{:10.5g}'.format(solution[i][j] * std[j] + mean[j]), end='\t')
		f.write('{:.5g}'.format(solution[i][j] * std[j] + mean[j]))
		f.write(',')
	print('{:10.5g}'.format(pre[i]), end='\t')
	f.write('{:.5g}'.format(pre[i]))
	
	f.write('\n')
	print()
f.close()
