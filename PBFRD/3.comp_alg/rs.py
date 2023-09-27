import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import sys
import copy
import csv
import xgboost
from calc import calp, calh
from tensorflow.compat.v1.keras import backend as KK
from tensorflow.compat.v1 import ConfigProto
import tensorflow as tf
import os

model_path = "/home/onwer/2.sdb2/1.work/2.hea/2.ml/4.dft/1.compare/1.newmodel/2.rev_pre"

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

KK.set_session(tf.compat.v1.Session(config=config))

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

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def relative_RMSE(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))/(K.mean(y_true))

def RRMSE(y_true, y_pred):
	sum  = 0
	sum2 = 0
	for i in range(len(y_pred)):
		sum = sum + (y_true[i]-y_pred[i]) ** 2
		sum2 = sum2 + y_true[i]
	return ((sum/len(y_pred)) ** 0.5)/(sum2/len(y_true))

f = open('ans.csv', 'r')
rdr = csv.reader(f)
rdr = list(rdr)
thermol = ['ys', 'el', 'uts']
for line in range(len(rdr)):
	if line  == 0:
		#output = rdr[line][0]
		ans = []
		for i in range(len(thermol)):
			ans.append(float(rdr[line][i]))
			if ans[i] == 0:
				ans[i] = 1e-5
	elif line == 1:
		fixa = np.array(rdr[line])
f.close()


ys_models_1 = []
ys_models_2 = []
ys_models_3 = []

file_list = os.listdir(model_path + '/ys_model/.')
for i in file_list:
	i = model_path + '/ys_model/' + i
	if 'project_model1' in i:
		ys_models_1.append(load_model(i))

	if 'project_model2' in i:
		ys_models_2.append(load_model(i))

	if 'project_model3' in i:
		ys_models_2.append(load_model(i))

	if 'project_model4' in i:
		ys_models_1.append(load_model(i))

	if 'project_model5' in i:
		ys_models_3.append(load_model(i))

bst = xgboost.XGBRegressor()
bst.load_model(model_path + '/ys_model/ys.bin')
ys_models_3.append(bst)

uts_models_1 = []
uts_models_2 = []
uts_models_3 = []

file_list = os.listdir(model_path + '/uts_model/.')
for i in file_list:
	i = model_path + '/uts_model/' + i
	if 'project_model1' in i:
		uts_models_1.append(load_model(i))

	if 'project_model2' in i:
		uts_models_2.append(load_model(i))

	if 'project_model3' in i:
		uts_models_2.append(load_model(i))

	if 'project_model4' in i:
		uts_models_1.append(load_model(i))

	if 'project_model5' in i:
		uts_models_3.append(load_model(i))

bst = xgboost.XGBRegressor()
bst.load_model(model_path + '/uts_model/ys.bin')
uts_models_3.append(bst)

el_models_1 = []
el_models_2 = []
el_models_3 = []

file_list = os.listdir(model_path + '/el_model/.')
for i in file_list:
	i = model_path + '/el_model/' + i
	if 'project_model1' in i:
		el_models_1.append(load_model(i))

	if 'project_model2' in i:
		el_models_2.append(load_model(i))

	if 'project_model3' in i:
		el_models_2.append(load_model(i))

	if 'project_model4' in i:
		el_models_1.append(load_model(i))

	if 'project_model5' in i:
		el_models_3.append(load_model(i))

bst = xgboost.XGBRegressor()
bst.load_model(model_path + '/el_model/ys.bin')
el_models_3.append(bst)

data = pd.read_csv("data.csv")
feature = ['C(at%)','Al(at%)','V(at%)','Cr(at%)','Mn(at%)','Fe(at%)','Co(at%)','Ni(at%)','Cu(at%)','Mo(at%)','Hom_Temp(K)','R(%)','Anneal_Temp(K)','Anneal_Time(h)','Hom_yes','Rol_yes','Ann_yes']
minl = []
maxl = []
for i in feature:
	tmpd = data.copy()
	if i in ['Hom_Temp(K)','R(%)','Anneal_Temp(K)','Anneal_Time(h)']:
		indx = tmpd[tmpd[i] == 0].index
		tmpd = tmpd.drop(indx)
	if i in ['V(at%)','Cr(at%)','Mn(at%)','Fe(at%)','Co(at%)','Ni(at%)','Cu(at%)']:
		minl.append(0)
		maxl.append(40)
	elif i == 'Mo(at%)':
		minl.append(0)
		maxl.append(10)
	elif i == 'Al(at%)':
		minl.append(0)
		maxl.append(15)
	elif i in ['C(at%)']:
		minl.append(0)
		maxl.append(5)
	elif i == 'Anneal_Time(h)':
		minl.append((1/12))
		maxl.append(2)
	else:
		minl.append(tmpd[i].min())
		maxl.append(tmpd[i].max())

if len(fixa) != len(minl):
	print("Error: invalid input")
	sys.exit(1)

solution = []
ori_solution = []

if os.path.isfile("res.csv"):
	f = open("res.csv", "r")
	line = f.readline()
	for i in range(6):
		line = f.readline()
		line = line.split(",")[:-6]
		line = list(map(float, line))
		ori_solution.append(line)
	f.close()
	additer = 6
else:
	additer = 12

for i in range(additer):
	ckover = True
	while ckover:
		tmp = []
		for j in range(len(minl)):
			if j == 10:
				if fixa[j] != 'n':
					tmp.append(float(fixa[j]))
				else:
					tmp.append(random.randrange(1173,int(maxl[j]) + 50, 50))
			elif j == 11:
				if fixa[j] != 'n':
					tmp.append(float(fixa[j]))
				else:
					tmp.append(random.randrange(int(minl[j]),int(maxl[j]) + 5, 5))
			elif j == 12:
				if fixa[j] != 'n':
					tmp.append(float(fixa[j]))
				else:
					tmp.append(random.randrange(1173,int(maxl[j]) + 50, 50))
			elif j == 13:
				if fixa[j] != 'n':
					tmp.append(float(fixa[j]))
				else:
					tmp.append((random.randrange(minl[j]*60,maxl[j]*60,5)/60))
			elif j == 14 or j == 15 or j == 16:
				if fixa[j] != 'n':
					tmp.append(float(fixa[j]))
				else:
					tmp.append(1)
			else:
				if fixa[j] != 'n':
					tmp.append(float(fixa[j]))
				else:
					tmp.append(round(random.uniform(minl[j],maxl[j]),2))
	
		randlist = random.choices(list(range(0,10)), k=random.randint(0,5))
		for j in randlist:
			tmp[j] = 0

		tmp2 = 0
		tmp3 = 0
		for j in range(10):
			if fixa[j] != 'n':
				if float(fixa[j]) < 0 or float(fixa[j]) > 99:
					print("Error: fixed value out of range")
					sys.exit(1)
				else:
					tmp[j] = float(fixa[j])
					tmp3 += tmp[j]
			else:
				tmp2 += tmp[j]
		tmp4 = 0
		maxv = float("-inf")
		for j in range(10):
			if fixa[j] == 'n':
				tmp[j] = tmp[j] / tmp2 * (100 - tmp3)
			else:
				tmp4 += tmp[j]
				continue
			if tmp[j] > 10:
				tmp[j] = round(tmp[j])
			elif tmp[j] > 1:
				tmp[j] = round(tmp[j], 1)
			else:
				tmp[j] = round(tmp[j], 2)
			tmp4 += tmp[j]
			if maxv < tmp[j]:
				maxj = j
				maxv = tmp[j]
		tmp[maxj] += 100 - tmp4
		ckover = False
		for j in range(len(tmp)):
			if fixa[j] == 'n':
				if tmp[j] < minl[j] or tmp[j] > maxl[j]:
					ckover = True
					break
	ori_solution.append(tmp)
ori_solution=np.array(ori_solution)

for i in range(len(ori_solution)):
	for j in range(len(ori_solution[i])):
		print('{:10.5g}'.format(ori_solution[i][j]), end='\t')
	print()
print()

properties = [[0.773,4,2.55,5,2.267,1050,442,0.1,479,0.8,140,7837,120000,12.011,6,3823,117,8.517,1086.5,11.3,607,121.776,18350,598,4300],\
	[1.432,3,1.61,4.2,2.6989,70,76,0.36,26,23.1,237,26.5,250,26.981,13,933.47,10.71,24.2,577.5,57.8,186,41.762,5000,284,2743],\
	[1.316,5,1.63,4.3,6.11,128,160,0.37,47,8.4,30.7,197,630,50.9415,23,2183,21.5,24.89,650.9,87,242,50.911,4560,444,3680],\
	[1.249,6,1.66,4.5,7.19,279,160,0.21,115,4.9,93.9,125,1060,51.9961,24,2180,21,23.35,652.9,83,155,65.21,5940,339.5,2755],\
	[1.35,7,1.55,4.1,7.44,198,120,0.35,76.4,21.7,7.81,1440,200,54.938,25,1519,12.91,26.32,717.3,68,42,-50,5150,221,2334],\
	[1.241,8,1.83,4.67,7.87,211,170,0.291,82,11.8,80.4,96.1,608,55.845,26,1811,13.81,25.1,762.5,62,100,14.785,5120,340,3134],\
	[1.251,9,1.88,5,8.8,209,180,0.32,75,13,100,62.4,1043,58.933,27,1768,16.06,24.81,760.4,55,167,63.898,4720,377,3200],\
	[1.246,10,1.91,5.22,8.88,200,180,0.31,76,13.4,90.9,69.3,638,58.693,28,1728,17.48,26.07,737.1,49,261.9,111.65,4900,379,3003],\
	[1.278,11,1.9,5.1,8.96,120,140,0.34,48,16.5,401,16.78,350,63.546,29,1357.77,13.26,24.44,745.5,47,202,119.235,3810,300.4,2835],\
	[1.363,6,2.16,4.53,10.28,329,230,0.31,126,4.8,138,53.4,2000,95.95,42,2896,37.48,24.06,684.3,87,218.1,72.1,5400,598,4912]]

properties = np.array(properties)
meanp = properties.mean(axis=0)
properties -= meanp
stdp = properties.std(axis=0)
properties /= stdp

solution = []
solution3 = []
homfrac = []

for i in range(int(len(ori_solution)/2)):
	tmp = ori_solution[i][0:10].tolist() + [ori_solution[i][12]]
	tmp2 = ori_solution[i][0:10].tolist() + [ori_solution[i][10]]
	prel = calp(tmp)
	homfrac.append(calh(tmp2)[-1])
	solution.append(np.array(ori_solution[i][10:14].tolist() + prel + ori_solution[i][14:].tolist()))
	solution3.append(np.array(ori_solution[i][10:14].tolist() + ori_solution[i][14:].tolist()))

solution2 = ori_solution[:,:10].tolist()
	
tmp_solution = ori_solution[:,:10].copy()
for j in range(10):
	for i in range(len(tmp_solution)):
		tmp_solution[i][j] = 0 if tmp_solution[i][j] == 0 else 1

for i in range(len(solution2)):
	for j in range(len(solution2[i])):
		solution2[i][j] = [solution2[i][j]]
		solution2[i][j].append(tmp_solution[i][j])
		for k in range(len(properties[j])):
			solution2[i][j].append(properties[j][k])

solution=np.array(solution)	
solution2=np.array(solution2)
solution3=np.array(solution3)
homfrac=np.array(homfrac)

count = 0
maxc = float('inf')
rcount = 0

while True:
	
	for i in range(len(solution)):
		if ori_solution[i][12] == 0 or ori_solution[i][13] == 0:
			ori_solution[i][12] = 0
			ori_solution[i][13] = 0
	
	solution = solution.tolist()
	solution3 = solution3.tolist()
	homfrac = homfrac.tolist()

	for i in range(int(len(ori_solution)/2),len(ori_solution)):
		cknone = False
		tmp = ori_solution[i][0:10].tolist() + [ori_solution[i][12]]
		tmp2 = ori_solution[i][0:10].tolist() + [ori_solution[i][10]]
		prel = calp(tmp)
		for j in range(len(prel)):
			if prel[j] == None:
				cknone = True
				break
		if cknone:
			ori_solution[i] = ori_solution[0]
			homfrac.append(homfrac[0])
			solution.append(solution[0])
			solution3.append(solution3[0])
		else:
			homfrac.append(calh(tmp2)[-1])
			solution.append(np.array(ori_solution[i][10:14].tolist() + prel + ori_solution[i][14:].tolist()))
			solution3.append(np.array(ori_solution[i][10:14].tolist() + ori_solution[i][14:].tolist()))

	solution2 = ori_solution[:,:10].tolist()
	
	tmp_solution = ori_solution[:,:10].copy()
	for j in range(10):
		for i in range(len(tmp_solution)):
			tmp_solution[i][j] = 0 if tmp_solution[i][j] == 0 else 1
	
	for i in range(len(solution2)):
		for j in range(len(solution2[i])):
			solution2[i][j] = [solution2[i][j]]
			solution2[i][j].append(tmp_solution[i][j])
			for k in range(len(properties[j])):
				solution2[i][j].append(properties[j][k])
	
	solution=np.array(solution)	
	solution2=np.array(solution2)
	solution3=np.array(solution3)
	homfrac=np.array(homfrac)

	prel = []
	solutiont = solution.copy()
	solutiont3 = solution3.copy()
	for j in range(len(solutiont[0])-3):
		for i in range(len(solutiont)):
			solutiont[i][j] -= mean[j+10]
			if std[j] != 0:
				solutiont[i][j] /= std[j+10]

	for j in range(4):
		for i in range(len(solutiont3)):
			solutiont3[i][j] -= mean[j+10]
			if std[j] != 0:
				solutiont3[i][j] /= std[j+10]
	
	pre = 0
	num = 0
	ys_list = []
	for i in range(len(ys_models_1)):
		pre += ys_models_1[i].predict([solution2, solutiont3]).flatten()
		ys_list.append(ys_models_1[i].predict([solution2, solutiont3]).flatten())
		num += 1
	for i in range(len(ys_models_2)):
		pre += ys_models_2[i].predict([solution2, solutiont]).flatten()
		ys_list.append(ys_models_2[i].predict([solution2, solutiont]).flatten())
		num += 1
	for i in range(len(ys_models_3)):
		pre += ys_models_3[i].predict(solutiont).flatten()
		ys_list.append(ys_models_3[i].predict(solutiont).flatten())
		num += 1
	ys_list = np.array(ys_list)
	ys_list = ys_list.T
	min_ys_list = ys_list.min(axis=1)
	pre /= num
	prel.append(pre)
	
	pre = 0
	num = 0
	el_list = []
	for i in range(len(el_models_1)):
		pre += el_models_1[i].predict([solution2, solutiont3]).flatten()
		el_list.append(el_models_1[i].predict([solution2, solutiont3]).flatten())
		num += 1
	for i in range(len(el_models_2)):
		pre += el_models_2[i].predict([solution2, solutiont]).flatten()
		el_list.append(el_models_2[i].predict([solution2, solutiont]).flatten())
		num += 1
	for i in range(len(el_models_3)):
		pre += el_models_3[i].predict(solutiont).flatten()
		el_list.append(el_models_3[i].predict(solutiont).flatten())
		num += 1
	el_list = np.array(el_list)
	el_list = el_list.T
	min_el_list = el_list.min(axis=1)
	pre /= num
	prel.append(pre)

	pre = 0
	num = 0
	uts_list = []
	for i in range(len(uts_models_1)):
		pre += uts_models_1[i].predict([solution2, solutiont3]).flatten()
		uts_list.append(uts_models_1[i].predict([solution2, solutiont3]).flatten())
		num += 1
	for i in range(len(uts_models_2)):
		pre += uts_models_2[i].predict([solution2, solutiont]).flatten()
		uts_list.append(uts_models_2[i].predict([solution2, solutiont]).flatten())
		num += 1
	for i in range(len(uts_models_3)):
		pre += uts_models_3[i].predict(solutiont).flatten()
		uts_list.append(uts_models_3[i].predict(solutiont).flatten())
		num += 1
	uts_list = np.array(uts_list)
	uts_list = uts_list.T
	min_uts_list = uts_list.min(axis=1)
	pre /= num
	prel.append(pre)
	prel = np.array(prel)
	prel = prel.T	

	if rcount % 200 == 199:
		for i in range(len(ori_solution)):
			for j in range(len(ori_solution[i])):
				print('{:10.5g}'.format(ori_solution[i][j]), end='\t')
			print(prel[i], end='\t')
			print()
		print()

	halfn = int(prel.shape[0]/2) 
	for j in range(halfn):
		maxp = 0
		for i in range(int(prel.shape[0])):
			if homfrac[i] < 1:
				maxi = i
				break
			inbred = False
			for l in range(halfn):
				solsum = 0
				for k in range(10):#len(ori_solution[i])):
					solsum += abs(ori_solution[i][k] - ori_solution[l][k])
					if ori_solution[i][k] > maxl[k] and fixa[k] == 'n':
						solsum = -1
						break
				if solsum == -1:
					maxi = i
					break
				if solsum < 2 * 10 and i != l:
					inbred = True
					if prel[i][1] < ans[1] and min_el_list[i] < (ans[1] / 2):
						maxi = i
						break
					if prel[i][2] < prel[i][0] + 100:
						maxi = i
						break
					sumi = 0
					sumi += abs((prel[i][0] + 3 * min_ys_list[i]) / 4 - ans[0]) / abs(ans[0])
					'''
					for k in range(prel.shape[1]):
						sumi += abs(prel[i][k] - ans[k]) / abs(ans[k])
					'''
					if prel[l][1] < ans[1] and min_el_list[l] < (ans[1] / 2):
						maxi = l
						break
					if prel[l][2] < prel[l][0] + 100:
						maxi = l
						break
					suml = 0
					suml += abs((prel[l][0] + 3 * min_ys_list[l]) / 4 - ans[0]) / abs(ans[0])
					'''
					for k in range(prel.shape[1]):
						suml += abs(prel[l][k] - ans[k]) / abs(ans[k])
					'''
					if sumi > suml:
						maxi = i
					else:
						maxi = l
					break
			if inbred:
				break
			
			if prel[i][1] < ans[1] and min_el_list[i] < (ans[1] / 2):
				maxi = i
				break
			if prel[i][2] < prel[i][0] + 100:
				maxi = i
				break
			suml = 0
			suml += abs((prel[i][0] + 3 * min_ys_list[i]) / 4 - ans[0]) / abs(ans[0])
			'''
			for k in range(prel.shape[1]):
				suml += abs(prel[i][k] - ans[k]) / abs(ans[k])
			'''
			if maxp < suml:
				maxi = i
				maxp = suml
		homfrac = np.delete(homfrac, maxi, axis=0)
		solution = np.delete(solution, maxi, axis=0)
		solution2 = np.delete(solution2, maxi, axis=0)
		solution3 = np.delete(solution3, maxi, axis=0)
		min_ys_list = np.delete(min_ys_list, maxi, axis=0)
		min_uts_list = np.delete(min_uts_list, maxi, axis=0)
		min_el_list = np.delete(min_el_list, maxi, axis=0)
		ori_solution = np.delete(ori_solution, maxi, axis=0)
		prel = np.delete(prel, maxi, axis=0)
	maxp = 0
	for i in range(int(prel.shape[0])):
		suml = 0
		suml += abs((prel[i][0] + 3 * min_ys_list[i]) / 4 - ans[0]) / abs(ans[0])
		'''
		for k in range(prel.shape[1]):
			suml += abs(prel[i][k] - ans[k]) / abs(ans[k])
		'''
		if maxp < suml:
			maxi = i
			maxp = suml

	rcount += 1
	if rcount % 20 == 0:
		print(maxc)
	if rcount % 200 == 0:
		for i in range(len(ori_solution)):
			for j in range(len(ori_solution[i])):
				print('{:10.5g}'.format(ori_solution[i][j]), end='\t')
			print(prel[i], end='\t')
			print(min_ys_list[i], end = '\t')
			print(min_el_list[i], end = '\t')
			print(min_uts_list[i], end = '\t')
			print()
		print()
		for i in range(len(solution)):
			for j in range(len(solution[i])):
				print('{:10.5g}'.format(solution[i][j]), end='\t')
			print('{:10.5g}'.format(homfrac[i]), end='\t')	
			print()
		print()
		print()
				
	if maxp != maxc:
		maxc = maxp
		count = 0
	else:
		count += 1

	if count > 10000 or rcount % 100 == 0:

		f = open('res.csv', 'w')
		for j in range(len(feature)):
			f.write(feature[j])
			#if j != (len(feature) - 1):
			f.write(',')
		f.write("YS(Mpa),UTS(Mpa),El(%)")
		f.write('\n')
		for i in range(len(ori_solution)):
			for j in range(len(ori_solution[i])):
				print('{:10.5g}'.format(ori_solution[i][j]), end='\t')
				f.write('{:.5g}'.format(ori_solution[i][j]))
				#if j != (len(solution[i]) - 1):
				f.write(',')
			for j in range(len(prel[i])):
				print('{:10.5g}'.format(prel[i][j]), end='\t')
				f.write('{:.5g}'.format(prel[i][j]))
				#if j != (len(prel[i]) - 1):
				f.write(',')
			print('{:10.5g}'.format(min_ys_list[i]), end='\t')
			f.write('{:.5g}'.format(min_ys_list[i]))
			f.write(',')
			print('{:10.5g}'.format(min_el_list[i]), end='\t')
			f.write('{:.5g}'.format(min_el_list[i]))
			f.write(',')
			print('{:10.5g}'.format(min_uts_list[i]), end='\t')
			f.write('{:.5g}'.format(min_uts_list[i]))
			
			f.write('\n')
			print()
		f.close()
		if count > 10000:
			break

	for i in range(int(prel.shape[0])):
		ckover = True
		while ckover:
			tmp = copy.deepcopy(ori_solution[i])
			for j in range(len(ori_solution[0])):
				if fixa[j] == 'n':
					if j <= 9:
						tmp[j] = (round(random.uniform(minl[j],maxl[j]),2))
						if tmp[j] < 0.02:
							tmp[j] = 0
					elif j == 10:
						tmp[j] = random.randrange(1173,int(maxl[j]) + 50, 50)
					elif j == 11:
						tmp[j] = random.randrange(int(minl[j]),int(maxl[j]) + 5, 5)
					elif j == 12:
						tmp[j] = random.randrange(1173,int(maxl[j]) + 50, 50)
					elif j == 13:
						tmp[j] = random.randrange(minl[j]*60,maxl[j]*60,5)/60
					elif j == 14 or j == 15 or j == 16:
						tmp[j] = 1
					else:
						tmp[j] = round(random.uniform(minl[j],maxl[j]),2)

			randlist = random.choices(list(range(0,10)), k=random.randint(0,5))
			for j in randlist:
				tmp[j] = 0

			tmp2 = 0
			tmp3 = 0
			for k in range(10):
				if fixa[k] != 'n':
					tmp[k] = float(fixa[k])
					tmp3 += tmp[k]
				else:
					tmp2 += tmp[k]
			tmp4 = 0
			maxv = float("-inf")
			for k in range(10):
				if fixa[k] == 'n':
					tmp[k] = tmp[k] / tmp2 * (100 - tmp3)
				else:
					tmp4 += tmp[k]
					continue
				if tmp[k] > 10:
					tmp[k] = round(tmp[k])
				elif tmp[k] > 1:
					tmp[k] = round(tmp[k], 1)
				else:
					tmp[k] = round(tmp[k], 2)
				tmp4 += tmp[k]
				if maxv < tmp[k]:
					maxk = k
					maxv = tmp[k]
			tmp[maxk] += 100 - tmp4
			ckover = False
			for j in range(len(tmp)):
				if fixa[j] == 'n':
					if tmp[j] < minl[j] or tmp[j] > maxl[j]:
						ckover = True
						break
		ori_solution = np.append(ori_solution, np.array([tmp]), axis=0)
