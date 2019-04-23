import sklearn as sk
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage
import numpy as np
import os
import threading
from pomegranate import *
from time import sleep
import copy
import json
from scipy.misc import toimage
from catboost import CatBoostClassifier, Pool

class Converter:
	def __init__(self, model,  threaded=False):
		self.geo_ref_points = [[30.69611111, -98.47833333],[29.41027778, -81.69722222], [33.55388889, -113.08194444]]
		self.client = storage.Client()
		self.bucket = self.client.get_bucket('481_project')
		###   load the classificatino model
		#model = open(model)
		#model = json.load(model)
		#self.classifier = NaiveBayes.from_json(model)
		self.slices = []
		self.processed_slices = {}
		self.data = None
		self.transform_matrix = None
		self.threaded = threaded

	
	def save2cloud(self, filename):

		#Save Class data
		np.save(filename, self.data)
		blob = self.bucket.blob("LCLU/Bayes/" + filename)
		blob.upload_from_filename(filename)
		cmd = "rm {}".format(filename)
		os.system(cmd)

		#Save LCLU map
		png_filename = filename.replace(".npy", ".png")
		return_arr = np.zeros(shape=(self.data.shape[0], self.data.shape[1], 3))
		for i in range(self.data.shape[0]):
			for j in range(self.data.shape[1]):
				k = int(self.data[i, j] - 1)
				return_arr[i, j, k] = 250
		im = toimage(return_arr)
		im.save(png_filename)
		blob = self.bucket.blob("LCLU/Bayes/images/" + png_filename)
		blob.upload_from_filename(png_filename)
		cmd = "rm {}".format(png_filename)
		os.system(cmd)

	def load_data(self, data):
		if self.threaded:
			for i in range(2):
				start = i*1250
				stop = (i+1)*1250
				if i <1:
					self.slices.append(data[start:stop, :, :])
				else:
					self.slices.append(data[start:, :, :])
		else:
			self.slices.append(data)

	def convert(self):
		if self.threaded:
			threads_list = []
			i = 0
			still_active = True
			for slice in self.slices:
				thread = threading.Thread(target=self.classify_slice, args=(slice,i))
				print("\n\nWWWWWWWWWWWWWWWWW\nStarting slice{}".format(i))
				thread.start()
				threads_list.append(thread)
				i = i+1
			#loop until threads have terminated
			while(still_active):
				i = 0
				for thread in threads_list:
					if thread.is_alive():
						i = i+1
				if i == 0:
					still_active = False
				else:
					print("\n\n----------{} Threads Still Active---------\n\n".format(i))
					sleep(20)
			print("\n\n00000 Fniished Converting Slices 00000\n\n")
			self.slices = None
			threads_list = None
		else:
			print("converting")
			self.classify_slice(self.slices[0], 0)

	def stack_proc_slices(self):
		if self.threaded:
			temp_data = self.processed_slices["0"]
			for i in range(1,2):
				temp_data = np.vstack((temp_data, self.processed_slices["{}".format(i)]))
			self.data = temp_data
			self.processed_slices = None
		else:
			self.data = self.processed_slices["0"]

	def classify_slice(self, slice, index):
		points = []
		for i in range(slice.shape[0]):
			for j in range(slice.shape[1]):
				points.append(slice[i,j,:])
		print("\n\n\n_______Processing_________\nSlice{}".format(index))
		proc_slice = self.classifier.predict(points)
		proc_slice.shape = (slice.shape[0],slice.shape[1])
		self.processed_slices["{}".format(index)] = proc_slice
		print("\n\nTTTTTTTTTTT\nslice {} processed".format(index))
		
	def geo_ref(self, geo_layer):
		y_pixels = int(geo_layer.shape[0])
		x_pixels = int(geo_layer.shape[1])
		b = np.array(([1, 1, geo_layer[0][0], geo_layer[1][1]],
					  [1, 1, geo_layer[0][-1], geo_layer[1][-2]],
					  [1, 1, geo_layer[-1][0], geo_layer[-2][1]],
					  [1, 1, geo_layer[-1][-1], geo_layer[-2][-2]]))
		a = np.array([[1, 1, 0, 0],  # coefficient matrix
					  [1, 1, 0, x_pixels],
					  [1, 1, y_pixels, 0],
					  [1, 1, y_pixels, x_pixels]])
		x = np.linalg.lstsq(a, b, rcond=-1)[0]
		x = np.linalg.pinv(x)
		self.transform_matrix = x

	'''
		This method uses the previously computed transformation matrix to find pixels corresponding to lat/long coordinates
	'''

	def get_pixel_values(self, lat, long):
		b = np.array([1, 1, lat, long])
		a = np.dot(b, self.transform_matrix)
		y = int(a[-2])
		x = int(a[-1])
		return y, x
