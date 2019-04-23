import gdal
import numpy as np
import Preprocessor as prep
from osgeo import gdal_array
base_url = "https://storage.googleapis.com/gcp-public-data-landsat/"
scene_names  = "addended.txt"
from google.cloud import storage
import wget
import os



client = storage.Client()
bucket = client.get_bucket('481_project')

'''
Parses the Landsat scene ID-string for [path#, row#, sensor#]
uses the data to construct the download urls for each band.
Landsat scenes are indexed by [path, row] numbers
'''
def string2data(scene_name):
    # Landsat satellite designation
    sat = scene_name[:4]
    # path number
    path_num = scene_name[10:13]
    # row number
    row_num = scene_name[13:16]
    # url for cloud storage location
    cons_url = "{}{}/01/{}/{}/{}/{}".format(base_url, sat, path_num, row_num, scene_name, scene_name)
    bands = None
    # For landsat 4,5,7 (Landsat 6 was inoperable
    if int(sat[-1]) == 4 or int(sat[-1]) == 5:
        bands = ["{}_B1.TIF".format(cons_url), "{}_B2.TIF".format(cons_url), "{}_B3.TIF".format(cons_url), "{}_B4.TIF".format(cons_url), "{}_B5.TIF".format(cons_url), "{}_B7.TIF".format(cons_url), "{}_B6.TIF".format(cons_url)]
    elif int(sat[-1]) == 7:
        bands = ["{}_B1.TIF".format(cons_url), "{}_B2.TIF".format(cons_url), "{}_B3.TIF".format(cons_url),
                 "{}_B4.TIF".format(cons_url), "{}_B5.TIF".format(cons_url), "{}_B7.TIF".format(cons_url),
                 "{}_B6_VCID_1.TIF".format(cons_url)]
    # For landsat 8
    else:
        bands = ["{}_B2.TIF".format(cons_url), "{}_B3.TIF".format(cons_url), "{}_B4.TIF".format(cons_url),
                 "{}_B5.TIF".format(cons_url), "{}_B6.TIF".format(cons_url), "{}_B7.TIF".format(cons_url),
                 "{}_B10.TIF".format(cons_url), "{}_B11.TIF".format(cons_url)]
    return bands, int(sat[-1]), "{}_MTL.txt".format(cons_url)


def save2cloud(filename):
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)
    cmd = "rm {}".format(filename)
    os.system(cmd)

with open(scene_names ,"r") as src_file:
    for line in src_file:
        line = line.strip().split(",")
        out_file = line[1]
        bands, sat, meta_url = string2data(line[0])
        prepper = None
        print(sat)
        print(type(sat))
        if(sat==8):
            prepper = prep.Landsat8(sat, meta_url, bands)
        elif(sat==7):
            prepper = prep.Landsat7(sat, meta_url, bands)
        else:
            prepper = prep.Landsat45(sat, meta_url, bands)
        prepper.extract_metadata()
        prepper.collect()
        prepper.transform_data()
        prepper.geo_reference()
        np.save(out_file, prepper.data)
        save2cloud("{}.npy".format(line[1]))
        print("{}.npy saved to cloud\n\n".format(line[1]))
    src_file.close()




