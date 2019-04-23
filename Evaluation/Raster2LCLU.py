import sklearn as sk
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage
import numpy as np
import os
import threading
from time import sleep
from Converter import Converter
samples_dir = "/home/lizard/483_Landsat_project/samples/"

if __name__ == "__main__":
	model_names = ['A_Bayes_json','PC_Bayes_json', 'M_Bayes_json']
	name_files_list = ["{}A_names.txt".format(samples_dir),"{}PC_names.txt".format(samples_dir),"{}M_names.txt".format(samples_dir)]
	client = storage.Client()
	bucket = client.get_bucket('481_project')
	print("XXXXXXXXX________START_______XXXXXXXXXXX\n\n")
	for i in range(3):
		site = name_files_list[i]
		model = model_names[i]
		with open(site, "r") as file_names:
			for line in file_names:
				line = line.strip()
				blob = bucket.blob(line)
				blob.download_to_filename(line)
				cnv = Converter(model)
				raster = np.load(line)
				geo = raster[:,:,-1]
				cnv.geo_ref(geo)
				raster = np.abs(raster[:,:,:-1])
				y_pixel, x_pixel = cnv.get_pixel_values(cnv.geo_ref_points[i][0], cnv.geo_ref_points[i][1])
				raster = raster[y_pixel:(y_pixel+2000), x_pixel:(x_pixel+2000), :]
				print(raster.shape)
				if (raster.shape[2] > 7):
					vis, therm_1, therm_2 = raster[:, :, :-2], raster[:, :, -2], raster[:, :,-1]
					raster = None
					therm = np.add(therm_1, therm_2)
					therm = np.divide(therm, 2)
					raster = np.dstack((vis, therm))
					vis = None
					therm_1 = None
					therm_2 = None
				print("\n\n\n")
				print("{}\n\n".format(raster.shape))
				cnv.load_data(raster)
				raster = None
				cmd = "rm {}".format(line)
				os.system(cmd)
				cnv.convert()
				cnv.stack_proc_slices()
				cnv.save2cloud("LCLU_{}".format(line))
				cnv = None
			file_names.close()
	cmd = "sudo shutdown now"
	os.system(cmd)








