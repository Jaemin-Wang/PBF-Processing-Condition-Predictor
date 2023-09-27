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

np.set_printoptions(precision=6, suppress=True)

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

f = open('ans.csv', 'r')
rdr = csv.reader(f)
rdr = list(rdr)
for line in range(len(rdr)):
	if line  == 0:
		fixa = np.array(rdr[line])
f.close()

model = xgboost.XGBRegressor()
model.load_model('ys.bin')

data = pd.read_csv("data.csv")
feature = ['reflectivity (%)','thermal conductivity (W/mK)','specific heat capacity (J/gK)','density(g/cm3)','melting point (C)','laser power (watt)','scan speed (mm/s)','layer thickness (mm)','hatch spacing (mm)','energy density (J/mm3)']
minl = []
maxl = []
for i in feature:
	if i == 'laser power (watt)':
		minl.append(data[i].min())
		maxl.append(400)
	elif i == 'scan speed (mm/s)':
		minl.append(data[i].min())
		maxl.append(4500)
	elif i == 'layer thickness (mm)':
		minl.append(0.025)
		maxl.append(0.12)
	elif i == 'hatch spacing (mm)':
		minl.append(0.08)
		maxl.append(data[i].max())
	elif i == 'energy density (J/mm3)':
		minl.append(float("-inf"))
		maxl.append(float("inf"))
	else:
		minl.append(data[i].min())
		maxl.append(data[i].max())

if len(fixa) != len(minl):
	print("Error: invalid input")
	sys.exit(1)

solution = []

additer = 24
#"""
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
#"""

for i in range(additer):
	tmp = []
	for j in range(len(minl)):
		if fixa[j] != 'n':
			tmp.append(float(fixa[j]))
		else:
			if j == 5:
				tmp.append(round(random.uniform(minl[j],maxl[j])))
			elif j == 6:
				tmp.append(round(random.uniform(minl[j],maxl[j])/10)*10)
			elif j == 7:
				tmp.append(round(random.uniform(minl[j],maxl[j]),3))
			elif j == 8:
				tmp.append(round(random.uniform(minl[j],maxl[j]),3))
			elif j == 9:
				tmp.append(round(tmp[5] / (tmp[6] * tmp[7] * tmp[8]),2))
			else:
				tmp.append(round(random.uniform(minl[j],maxl[j]),3))
	solution.append(tmp)
solution=np.array(solution)

for i in range(len(solution)):
	for j in range(len(solution[i])):
		solution[i][j] = (solution[i][j] - mean[j]) / std[j]
		print('{:10.5g}'.format(solution[i][j] * std[j] + mean[j]), end='\t')
	print()
print()

count = 0
maxc = float('inf')
rcount = 0
multstd = 1
impstd = 98

while True:
	
	pre = model.predict(solution).flatten()
	pre = pre.astype('float64')
	pre[pre <= 0] = 1e-10
	pre[pre >= 1] = 1-1e-10
	pre = np.log(-(pre/(pre - 1))) / multstd + impstd
	#pre[pre > 100] = 100
	#pre[pre < 50] = 50
	#pre[pre < 98] = 0
	#pre[pre >= 98] = 1

	if rcount % 200 == 199:
		for i in range(len(solution)):
			for j in range(len(solution[i])):
				print('{:10.5g}'.format(solution[i][j] * std[j] + mean[j]), end='\t')
			print(pre[i], end='\t')
			print()
		print()

	halfn = int(pre.shape[0]/2) 
	while True:
		inbred = False
		for i in range(int(pre.shape[0])):
			for l in range(int(pre.shape[0])):
				if i == l:
					continue
				solsum = 0
				sumbred = 0
				for k in range(len(solution[i])-1):
					solsum += abs(solution[i][k] - solution[l][k])
					if fixa[k] == 'n':
						if abs(solution[i][k] - solution[l][k]) < 0.01:
							sumbred += 2
						elif abs(solution[i][k] - solution[l][k]) < 0.03:
							sumbred += 1
				if solsum < 1 or sumbred >= 2:
					inbred = True
					halfn -= 1
					if pre[i] < pre[l]:
						maxi = i
					else:
						maxi = l
					break
			if inbred:
				break
		if inbred:
			solution = np.delete(solution, maxi, axis=0)
			pre = np.delete(pre, maxi, axis=0)
		else:
			break

	for j in range(halfn):
		maxp = float("inf")
		for i in range(int(pre.shape[0])):
			if maxp > pre[i]:
				maxi = i
				maxp = pre[i]
		solution = np.delete(solution, maxi, axis=0)
		pre = np.delete(pre, maxi, axis=0)
	maxp = float("inf")
	for i in range(int(pre.shape[0])):
		if maxp > pre[i]:
			maxi = i
			maxp = pre[i]

	rcount += 1
	if rcount % 20 == 0:
		print(count,maxc, sep=': ')
	if rcount % 200 == 0:
		for i in range(len(solution)):
			for j in range(len(solution[i])):
				print('{:10.5g}'.format(solution[i][j] * std[j] + mean[j]), end='\t')
			print(pre[i], end='\t')
			print()
		print()
				
	if maxp != maxc:
		maxc = maxp
		count = 0
	else:
		count += 1

	if count > 100000 or rcount % 100 == 0:

		f = open('res.csv', 'w')
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
			if pre[i] < 98:
				print('0', end='\t')
				f.write('0')
			else:
				print('1', end='\t')
				f.write('1')
			
			f.write('\n')
			print()
		f.close()
		if count > 100000:
			break

	additer = 24 - len(pre)
	for i in range(additer):
		tmp = []
		for j in range(len(minl)):
			if fixa[j] != 'n':
				tmp.append(float(fixa[j]))
			else:
				if j == 5:
					tmp.append(round(random.uniform(minl[j],maxl[j])))
				elif j == 6:
					tmp.append(round(random.uniform(minl[j],maxl[j])/10)*10)
				elif j == 7:
					tmp.append(round(random.uniform(minl[j],maxl[j]),3))
				elif j == 8:
					tmp.append(round(random.uniform(minl[j],maxl[j]),3))
				elif j == 9:
					tmp.append(round(tmp[5] / (tmp[6] * tmp[7] * tmp[8]),2))
				else:
					tmp.append(round(random.uniform(minl[j],maxl[j]),3))
		for j in range(len(minl)):
			tmp[j] = (tmp[j] - mean[j]) / std[j]
		solution = np.append(solution, np.array([tmp]), axis=0)
