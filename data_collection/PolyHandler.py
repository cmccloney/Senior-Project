import os
import numpy as np
import sklearn as sk
import csv
from google.cloud import storage
import re
from matplotlib.path import Path


class PolyHandler:
	def __init__(self):
		samples_dir = "/home/lizard/483_Landsat_project/samples/"
		self.client = storage.Client()
		self.bucket = self.client.get_bucket('481_project')
		# points_lists are the csv files with hand collected points and classes
		self.points_lists = ["{}A_polys.csv".format(samples_dir)]
		# files_list are text files containing the names of the files for each image set
		self.files_list = ["{}A_poly_names.txt".format(samples_dir)]
		self.dst_files = ["{}A_poly_samples.npy".format(samples_dir)]
		self.data = None
		self.transform_matrix = None

	"""
	This method compiles the hand collected points into a set of sample observations
	There are around (50 * #images * 3) samples.
	Each sample is the vector: [band_1 reflectance, band_2 reflectance, ..., class]

	For each geographic site, the method compiles the hand collected data points into a list:
		Then for each image data object corresponding to each site:
			for each ground-truth point in the list:
				samples the spectral reflectance values at the ground-truth lat/long
	"""

	def load_data(self):
		samples = []
		for i in range(1):
			data_entries = []
			pt_list = self.points_lists[i]
			nm_list = self.files_list[i]
			with open(pt_list, newline='', encoding='utf-8') as csv_file:
				csv_reader = csv.reader(csv_file)
				for row in csv_reader:
					point = [float(re.sub("[^0-9.\-]", "", row[0])), float(re.sub("[^0-9.\-]", "", row[1])), float(re.sub("[^0-9.\-]", "", row[2])),
							 float(re.sub("[^0-9.\-]", "", row[3])), float(re.sub("[^0-9.\-]", "", row[4])), float(re.sub("[^0-9.\-]", "", row[5])),
							 float(re.sub("[^0-9.\-]", "", row[6])), float(re.sub("[^0-9.\-]", "", row[7])),
							 int(re.sub("[^0-9.\-]", "", row[8]))]
					data_entries.append(point)
				csv_file.close()
			with open(nm_list, "r") as names_file:
				for line in names_file:
					line = line.strip()
					print("\nprocessing: \t{}\n".format(line))
					blob = self.bucket.blob(line)
					blob.download_to_filename(line)
					raster = np.load(line)
					y_max  = raster.shape[0]
					geo = raster[:, :, -1]
					spectral_layers = np.abs(raster[:, :, :-1])
					if (spectral_layers.shape[2] > 7):
						vis, therm_1, therm_2 = spectral_layers[:, :, :-2], spectral_layers[:, :, -2], spectral_layers[:, :, -1]
						spectral_layers = None
						therm = np.add(therm_1, therm_2)
						therm = np.divide(therm, 2)
						spectral_layers = np.dstack((vis, therm))
						vis = None
						therm_1 = None
						therm_2 = None
					self.geo_ref(geo)
					raster = None
					cmd = "rm {}".format(line)
					os.system(cmd)
					samples_as_array = None
					for point in data_entries:
						y1, x1 = self.get_pixel_values(point[0], point[1])
						y2, x2 = self.get_pixel_values(point[2], point[3])
						y3, x3 = self.get_pixel_values(point[4], point[5])
						y4, x4 = self.get_pixel_values(point[6], point[7])
						label = point[-1]
						verts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
						codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
						path = Path(verts, codes)
						extents = path.get_extents().get_points()
						print(extents)
						for y in range(int(extents[0, 1]), int(extents[1, 1])):
							for x in range(int(extents[0, 0]), int(extents[1, 0])):
								if path.contains_point((x,y)):
									sample = np.ravel(spectral_layers[y, x, :])
									sample = np.append(sample, label)
									samples.append(sample)
						if samples_as_array is None:
							samples_as_array = np.ravel(np.array(samples[0])).astype("float32")
						for j in range(1, len(samples) - 1):
							samp = np.ravel(np.array(samples[j]))
							samples_as_array = np.dstack((samp, samples_as_array))
						samples = []
						print("\n{}\n".format(samples_as_array.shape))
				samples = []
				np.save(self.dst_files[i], samples_as_array.T)
				names_file.close()

	"""
	This method uses linear algebra to find the transformation matrix
	to convert points from lat/long coordinates to x/y pixel values.
	The Upper/Lower--Left/Right latitude and longitude values for the corner pixels of each image were obtained from landsat metadata.
	The latitude and longitude values for the corners are stored in the geospatial layer of the numpy data cube 

	The method sets the instance variable: "self.transform_matrix"
	This variable is used in the get_pixel_values method to convert from lat/long to pixel coordinates.

	solves x^-1 for Ax=b where A=[corner pixel coordinates] and b = [corner lat/long values]
	"""

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

	def shuffle(self):
		self.data = np.random.shuffle(self.data)

	def cleave(self):
		sample_reflectance_values = (self.data[:, :-1])
		sample_class_label = ((self.data[:, -1])).astype(int)
		return sample_reflectance_values, sample_class_label

