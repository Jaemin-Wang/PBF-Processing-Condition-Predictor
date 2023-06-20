import csv
import io
import sys
import pandas as pd
import numpy as np
import os
from PyQt5.QtWidgets import QItemDelegate, QTableWidget, QMainWindow, QApplication, QHBoxLayout, QVBoxLayout, QAction, QWidget, QLabel, QLineEdit, QComboBox, QPushButton, QDoubleSpinBox, QFrame, QTableWidgetItem, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import xgboost
import random
import math

class RandomSearch(QThread):
	stop_task = pyqtSignal()
	result_emit = pyqtSignal(np.ndarray, np.ndarray)
	current_emit = pyqtSignal(int, float)

	def __init__(self):
		self.fixa = None
		self.candidate = None
		self.iteration = None
		super().__init__()

	def run(self):
		f = open('train_mean_std', 'r')
		rdr = csv.reader(f)
		rdr = list(rdr)
		mean = rdr[0]
		std = rdr[1]
		mean = np.array(mean)
		std = np.array(std)
		mean = mean.astype(float)
		std = std.astype(float)
		f.close()

		fixa = self.fixa

		model = xgboost.XGBRegressor()
		model.load_model('model.bin')

		data = pd.read_csv("data.csv")

		feature = ['reflectivity (%)','thermal conductivity (W/mK)','specific heat capacity (J/gK)','density(g/cm3)','melting point (C)','laser power (watt)','scan speed (mm/s)','layer thickness (mm)','hatch spacing (mm)','energy density (J/mm3)']
		minl = []
		maxl = []
		for i in feature:
			if i == 'laser power (watt)':
				minl.append(50)
				maxl.append(400)
			elif i == 'scan speed (mm/s)':
				minl.append(100)
				maxl.append(4500)
			elif i == 'layer thickness (mm)':
				minl.append(0.025)
				maxl.append(0.12)
			elif i == 'hatch spacing (mm)':
				minl.append(0.08)
				maxl.append(0.2)
			elif i == 'energy density (J/mm3)':
				minl.append(float("-inf"))
				maxl.append(float("inf"))
			else:
				minl.append(data[i].min())
				maxl.append(data[i].max())
		
		solution = []
		
		additer = self.candidate * 2
		
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
								if abs(solution[i][k] - solution[l][k]) < 0.005:
									sumbred += 2
								elif abs(solution[i][k] - solution[l][k]) < 0.015:
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
			if rcount % 100 == 0:
				self.current_emit.emit(count, (1/(1+math.exp(-(maxc - impstd) * multstd))-0.5)*200)
						
			if maxp != maxc:
				maxc = maxp
				count = 0
			else:
				count += 1
		
			if count > self.iteration or self.isInterruptionRequested():
				pre = model.predict(solution).flatten()
				pre = pre.astype('float64')
				pre[pre <= 0] = 1e-10
				pre[pre >= 1] = 1-1e-10
				pre = (pre - 0.5) * 2 * 100
				for i in range(len(solution)):
					for j in range(len(solution[i])):
						solution[i][j] = solution[i][j] * std[j] + mean[j]
				self.result_emit.emit(solution, pre)
				break
		
			additer = self.candidate * 2 - len(pre)
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

class FloatDelegate(QItemDelegate):
	def __init__(self, parent=None):
		QItemDelegate.__init__(self, parent=parent)

	def createEditor(self, parent, option, index):
		editor = QLineEdit(parent)
		#editor.setValidator(QDoubleValidator())
		return editor


class TableWidget(QTableWidget):
	def __init__(self, parent=None):
		QTableWidget.__init__(self, parent)

	def changedf(self, df):
		self.df = df
		self.updatedf()

	def updatedf(self):
		nRows = len(self.df.index)
		nColumns = len(self.df.columns)
		self.setRowCount(nRows)
		self.setColumnCount(nColumns)
		self.setItemDelegate(FloatDelegate())
		tmp = []
		for i in self.df.columns:
			if type(i) == type((1,2)):
				tmp.append(' '.join(i))
			else:
				tmp.append(str(i))
		self.setHorizontalHeaderLabels(tmp)

		for i in range(self.rowCount()):
			for j in range(self.columnCount()):
				if type(self.df.iloc[i, j]) == type(None):
					x = ''
				elif type(self.df.iloc[i, j]) != type('str'):
					try:
						x = '{:.3f}'.format(self.df.iloc[i, j])
					except:
						x = self.df.iloc[i, j]
				else:
					x = self.df.iloc[i, j]
				self.setItem(i, j, QTableWidgetItem(x))

	def keyPressEvent(self, ev):
		if (ev.key() == Qt.Key_C) and (ev.modifiers() & Qt.ControlModifier): 
			self.copySelection()

	def copySelection(self):
		selection = self.selectedIndexes()
		if selection:
			rows = sorted(index.row() for index in selection)
			columns = sorted(index.column() for index in selection)
			rowcount = rows[-1] - rows[0] + 1
			colcount = columns[-1] - columns[0] + 1
			table = [[''] * colcount for _ in range(rowcount)]
			for index in selection:
				row = index.row() - rows[0]
				column = index.column() - columns[0]
				table[row][column] = index.data()
			stream = io.StringIO()
			csv.writer(stream, delimiter='\t').writerows(table)
			QApplication.clipboard().setText(stream.getvalue())

class PBFRD(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
		self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
		self.elements = ['Fe','Ni','Cr','Mo','Ta','Nb','Ti','Al','V','Si','Mg','Y','Nd','Zr','C','Co','Cu','Zn','Mn','Li','Ag','W','Sn','Ga']
		self.reflectivity = [65,72,63,58,78,81,55,71,61,28,74,63,58,66,27,67,90,80,44,94,97,62,54,70]
		self.atomic_list = [element + ' (at%)' for element in self.elements]
		self.input_comp = {}
		self.initUI()

	def initUI(self):
		self.randomsearch = RandomSearch()
		self.randomsearch.result_emit.connect(self.update_table)
		self.randomsearch.current_emit.connect(self.update_text)

		input_vbox = QVBoxLayout()
		input_title = QLabel('Material Properties of Powder and Processing Condition')

		composition_frame = QFrame()
		#composition_frame.setStyleSheet("border: 1px solid black;")
		composition_vbox = QVBoxLayout(composition_frame)
		composition_hbox = QHBoxLayout()
		self.composition_label = QLabel('Please insert the composition of powder.')
		self.composition_label.setStyleSheet("background-color: white;")
		self.composition_cbox = QComboBox()
		self.composition_cbox.addItems(self.atomic_list)
		self.composition_line = QLineEdit()
		self.composition_add_btn = QPushButton('Add')
		self.composition_add_btn.pressed.connect(self.composition_add)
		self.composition_remove_btn = QPushButton('Remove')
		self.composition_remove_btn.pressed.connect(self.composition_remove)
		composition_hbox.addWidget(self.composition_cbox)
		composition_hbox.addWidget(self.composition_line)
		composition_hbox.addWidget(self.composition_add_btn)
		composition_hbox.addWidget(self.composition_remove_btn)
		composition_vbox.addWidget(self.composition_label)
		composition_vbox.addLayout(composition_hbox)

		conductivity_hbox = QHBoxLayout()
		conductivity_label = QLabel('Thermal Conductivity (5-385 W/mK)')
		self.conductivity_sbox = QDoubleSpinBox()
		self.conductivity_sbox.setRange(5,385)
		self.conductivity_sbox.setValue(76.2)
		self.conductivity_sbox.setSingleStep(0.1)
		conductivity_hbox.addWidget(conductivity_label)
		conductivity_hbox.addWidget(self.conductivity_sbox)

		capacity_hbox = QHBoxLayout()
		capacity_label = QLabel('Specific Heat Capacity (0.13-1.06 J/gK)')
		self.capacity_sbox = QDoubleSpinBox()
		self.capacity_sbox.setRange(0.13, 1.06)
		self.capacity_sbox.setValue(0.44)
		self.capacity_sbox.setSingleStep(0.01)
		self.capacity_sbox.setDecimals(3)
		capacity_hbox.addWidget(capacity_label)
		capacity_hbox.addWidget(self.capacity_sbox)

		density_hbox = QHBoxLayout()
		density_label = QLabel('Density (1.74-19.3 g/cm<sup>3</sup>)')
		self.density_sbox = QDoubleSpinBox()
		self.density_sbox.setRange(1.74, 19.3)
		self.density_sbox.setValue(7.87)
		self.density_sbox.setSingleStep(0.01)
		self.density_sbox.setDecimals(3)
		density_hbox.addWidget(density_label)
		density_hbox.addWidget(self.density_sbox)

		melting_hbox = QHBoxLayout()
		melting_label = QLabel('Melting Point (441-3410 °C)')
		self.melting_sbox = QDoubleSpinBox()
		self.melting_sbox.setRange(441, 3410)
		self.melting_sbox.setValue(1535)
		self.melting_sbox.setSingleStep(0.1)
		melting_hbox.addWidget(melting_label)
		melting_hbox.addWidget(self.melting_sbox)

		power_hbox = QHBoxLayout()
		power_label = QLabel('Laser Power (50-400 W)')
		self.power_sbox = QDoubleSpinBox()
		self.power_sbox.setRange(50, 400)
		self.power_sbox.setSingleStep(10)
		self.power_btn = QPushButton('Set Free')
		self.power_btn.setFixedSize(self.power_btn.minimumSizeHint())
		self.power_btn.pressed.connect(self.power)
		power_hbox.addWidget(power_label)
		power_hbox.addWidget(self.power_sbox)
		power_hbox.addWidget(self.power_btn)

		speed_hbox = QHBoxLayout()
		speed_label = QLabel('Scan Speed (100-4500 mm/s)')
		self.speed_sbox = QDoubleSpinBox()
		self.speed_sbox.setRange(100, 4500)
		self.speed_sbox.setSingleStep(100)
		self.speed_btn = QPushButton('Set Free')
		self.speed_btn.setFixedSize(self.speed_btn.minimumSizeHint())
		self.speed_btn.pressed.connect(self.speed)
		speed_hbox.addWidget(speed_label)
		speed_hbox.addWidget(self.speed_sbox)
		speed_hbox.addWidget(self.speed_btn)

		thickness_hbox = QHBoxLayout()
		thickness_label = QLabel('Layer Thickness (0.025-0.12 mm)')
		self.thickness_sbox = QDoubleSpinBox()
		self.thickness_sbox.setDecimals(3)
		self.thickness_sbox.setRange(0.025, 0.12)
		self.thickness_sbox.setSingleStep(0.01)
		self.thickness_btn = QPushButton('Set Free')
		self.thickness_btn.setFixedSize(self.thickness_btn.minimumSizeHint())
		self.thickness_btn.pressed.connect(self.thickness)
		thickness_hbox.addWidget(thickness_label)
		thickness_hbox.addWidget(self.thickness_sbox)
		thickness_hbox.addWidget(self.thickness_btn)

		spacing_hbox = QHBoxLayout()
		spacing_label = QLabel('Hatch Spacing (0.08-0.2 mm)')
		self.spacing_sbox = QDoubleSpinBox()
		self.spacing_sbox.setRange(0.08, 0.2)
		self.spacing_sbox.setSingleStep(0.01)
		self.spacing_sbox.setDecimals(3)
		self.spacing_btn = QPushButton('Set Free')
		self.spacing_btn.setFixedSize(self.spacing_btn.minimumSizeHint())
		self.spacing_btn.pressed.connect(self.spacing)
		spacing_hbox.addWidget(spacing_label)
		spacing_hbox.addWidget(self.spacing_sbox)
		spacing_hbox.addWidget(self.spacing_btn)

		option_title = QLabel('\nPrediction Options')

		candidate_hbox = QHBoxLayout()
		candidate_label = QLabel('Maximum Candidates (5-50)')
		self.candidate_sbox = QDoubleSpinBox()
		self.candidate_sbox.setRange(5, 50)
		self.candidate_sbox.setValue(5)
		self.candidate_sbox.setSingleStep(1)
		self.candidate_sbox.setDecimals(0)
		candidate_hbox.addWidget(candidate_label)
		candidate_hbox.addWidget(self.candidate_sbox)

		iteration_hbox = QHBoxLayout()
		iteration_label = QLabel('Maximum Iterations (1-1000000)')
		self.iteration_sbox = QDoubleSpinBox()
		self.iteration_sbox.setRange(1, 1000000)
		self.iteration_sbox.setValue(100000)
		self.iteration_sbox.setSingleStep(1)
		self.iteration_sbox.setDecimals(0)
		iteration_hbox.addWidget(iteration_label)
		iteration_hbox.addWidget(self.iteration_sbox)

		input_vbox.addWidget(input_title)
		input_vbox.addWidget(composition_frame)
		input_vbox.addLayout(conductivity_hbox)
		input_vbox.addLayout(capacity_hbox)
		input_vbox.addLayout(density_hbox)
		input_vbox.addLayout(melting_hbox)
		input_vbox.addLayout(power_hbox)
		input_vbox.addLayout(speed_hbox)
		input_vbox.addLayout(thickness_hbox)
		input_vbox.addLayout(spacing_hbox)
		input_vbox.addWidget(option_title)
		input_vbox.addLayout(candidate_hbox)
		input_vbox.addLayout(iteration_hbox)
		input_vbox.setAlignment(Qt.AlignCenter)

		current_vbox = QVBoxLayout()
		current_label = QLabel("\n\nThe Current Status of the Prediction")
		self.current_text = QTextEdit()
		self.current_text.append('The number of the iterations since the candidates were changed: \nThe prediction ends when the number reaches the maximum iterations.')
		current_vbox.addWidget(current_label)
		current_vbox.addWidget(self.current_text)

		vbox = QVBoxLayout()
		self.pred_btn = QPushButton('Start Prediction')
		self.pred_btn.pressed.connect(self.pred)
		self.stop_btn = QPushButton('Stop Prediction')
		self.stop_btn.pressed.connect(self.stop)
		vbox.addLayout(input_vbox)
		vbox.addLayout(current_vbox)
		vbox.addWidget(self.pred_btn)
		vbox.addWidget(self.stop_btn)

		table_vbox = QVBoxLayout()
		table_label = QLabel('Result Table')
		self.table = TableWidget()
		self.table.changedf(pd.DataFrame())
		table_vbox.addWidget(table_label)
		table_vbox.addWidget(self.table)

		hbox = QHBoxLayout()
		hbox.addLayout(vbox)
		hbox.addLayout(table_vbox)

		wid = QWidget(self)
		self.setCentralWidget(wid)
		wid.setLayout(hbox)
		self.setWindowTitle('PBF Processing Condition Predictor')
		self.setGeometry(0, 0, 1024, 768)
		self.statusBar().showMessage('Ready.')
		self.show()

	def power(self):
		is_visible = self.power_sbox.isVisible()
		self.power_sbox.setVisible(not is_visible)
		if is_visible:
			self.power_btn.setText('Fix')
		else:
			self.power_btn.setText('Set Free')

	def speed(self):
		is_visible = self.speed_sbox.isVisible()
		self.speed_sbox.setVisible(not is_visible)
		if is_visible:
			self.speed_btn.setText('Fix')
		else:
			self.speed_btn.setText('Set Free')

	def thickness(self):
		is_visible = self.thickness_sbox.isVisible()
		self.thickness_sbox.setVisible(not is_visible)
		if is_visible:
			self.thickness_btn.setText('Fix')
		else:
			self.thickness_btn.setText('Set Free')

	def spacing(self):
		is_visible = self.spacing_sbox.isVisible()
		self.spacing_sbox.setVisible(not is_visible)
		if is_visible:
			self.spacing_btn.setText('Fix')
		else:
			self.spacing_btn.setText('Set Free')

	def composition_add(self):
		try:
			comp = float(self.composition_line.text())
			ele = self.composition_cbox.currentText()
			self.input_comp[ele] = comp
			self.change_comp_label()
		except Exception as e:
			print(e)
			self.statusBar().showMessage('Please insert float type input.')
	
	def composition_remove(self):
		ele = self.composition_cbox.currentText()
		try:
			del self.input_comp[ele]
		except:
			return
		self.change_comp_label()
	
	def change_comp_label(self):
		self.statusBar().showMessage('Ready.')
		if len(self.input_comp) == 0:
			self.composition_label.setText('Please insert the composition of powder.')
		else:
			result = []
			subdict = {}
			count = 0
			for k,v in self.input_comp.items():
				subdict[k] = v
				count += 1
				if count == 5:
					result.append(subdict)
					subdict = {}
					count = 0
			if subdict:
				result.append(subdict)
			result_text = ''
			for subdict in result:
				result_text += '  '.join([f'{key}: {value},' for key, value in subdict.items()]) + '\n'
			self.composition_label.setText(result_text[:-2])
	
	def pred(self):
		if self.randomsearch.isRunning():
			return
		
		if sum(self.input_comp.values()) != 100:
			self.statusBar().showMessage('The summation of the composition of powder should be 100 at%.')
			return
		if self.power_sbox.isVisible() and self.speed_sbox.isVisible() and self.thickness_sbox.isVisible() and self.spacing_sbox.isVisible():
			self.statusBar().showMessage('At least one processing condition should be set free.')
			return

		if not os.path.exists('train_mean_std'):
			self.statusBar().showMessage('Please provide the \"train_mean_std\" file.')
			return
		
		if not os.path.exists('model.bin'):
			self.statusBar().showMessage('Please provide the \"model.bin\" file.')
			return
		
		if not os.path.exists('data.csv'):
			self.statusBar().showMessage('Please provide the \"data.csv\" file.')
			return
		self.statusBar().showMessage('Please wait. It may take several days to predict the results.')

		fixa = []
		reflect = 0
		for k,v in self.input_comp.items():
			reflect += self.reflectivity[self.atomic_list.index(k)] * v / 100
		fixa.append(reflect)
		fixa.append(self.conductivity_sbox.value())
		fixa.append(self.capacity_sbox.value())
		fixa.append(self.density_sbox.value())
		fixa.append(self.melting_sbox.value())
		if self.power_sbox.isVisible():
			fixa.append(self.power_sbox.value())
		else:
			fixa.append('n')
		if self.speed_sbox.isVisible():
			fixa.append(self.speed_sbox.value())
		else:
			fixa.append('n')
		if self.thickness_sbox.isVisible():
			fixa.append(self.thickness_sbox.value())
		else:
			fixa.append('n')
		if self.spacing_sbox.isVisible():
			fixa.append(self.spacing_sbox.value())
		else:
			fixa.append('n')
		fixa.append('n')
		fixa = np.array(fixa)

		self.randomsearch.fixa = fixa
		self.randomsearch.candidate = int(self.candidate_sbox.value())
		self.randomsearch.iteration = int(self.iteration_sbox.value())

		self.randomsearch.start()
	
	def stop(self):
		if self.randomsearch.isRunning():
			self.randomsearch.requestInterruption()

	@pyqtSlot(np.ndarray, np.ndarray)
	def update_table(self, solution, pre):
		self.statusBar().showMessage('Ready.')
		self.table.changedf(pd.DataFrame())
		solution = solution[:,5:9]
		columns = ['Laser Power (W)','Scan Speed (mm/s)','Layer Thickness (mm)','Hatch Spacing (mm)']
		self.table.changedf(pd.DataFrame(solution, columns = columns))
		self.table.df['Certainty (%)'] = pre
		self.table.df = self.table.df.drop(self.table.df[self.table.df['Certainty (%)'] <= 0].index)
		self.table.updatedf()

		if self.table.df.empty:
			self.statusBar().showMessage('There is no candidate for the given input.')

	@pyqtSlot(int, float)
	def update_text(self, count, least):
		self.current_text.clear()
		self.current_text.append('The number of the iterations since the candidates were changed: ' + str(count) + '\nThe prediction ends when the number reaches the maximum iterations.')

	def keyPressEvent(self, e):
		return
		if e.modifiers() & Qt.ShiftModifier and e.key() == Qt.Key_C:
			if self.figure.select_mode == 0 and self.figure.mousemode == 0:
				self.copy_points()

if __name__ == '__main__':

	app = QApplication(sys.argv)
	ex = PBFRD()
	sys.exit(app.exec_())
