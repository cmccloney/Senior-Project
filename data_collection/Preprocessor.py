import re
import math
import numpy as np
import urllib.request as req
from osgeo import gdal_array, gdal
from scipy import misc
import os
import wget
from numba import jit

solar_spectral_irradiances = {"B1": 1970,
                              "B2": 1842,
                              "B3": 1547,
                              "B4": 1044,
                              "B5": 225.7,
                              "B7": 82.06}

class AbstractPreprocessor(object):
    def __init__(self, satnum, meta_url, bands):
        self.satnum = satnum
        self.meta_url = meta_url
        self.bands = bands
        self.metadata = None
        self.data = None
        self.lats = {}
        self.longs = {}

    def collect(self, dtype='float32'):
        first = True
        arr = None
        j = 0
        for band in self.bands:
                j = j+1
                if first == True:
                    filename = wget.download(band)
                    arr = gdal_array.LoadFile(filename).astype(dtype)
                    cmd = "rm {}".format(filename)
                    os.system(cmd)
                    filename = None
                    print("\n\ndownloading {}\n\n\n".format(band))
                    first = False
                else:
                    print("\n\ndownloading {}\n\n\n".format(band))
                    filename = wget.download(band)
                    temp = gdal_array.LoadFile(filename).astype(dtype)
                    cmd = "rm {}".format(filename)
                    os.system(cmd)
                    filename = None
                    arr = np.dstack((arr, temp))

        self.data = arr
        arr = None

    def extract_metadata(self):
        pass


    def geo_reference(self):
        y_pixels = self.data.shape[0]
        x_pixels = self.data.shape[1]
        geo_layer = np.zeros((y_pixels,x_pixels), dtype="float32")
        geo_layer[0][0] = self.lats["UL"]
        geo_layer[1][1] = self.longs["UL"]
        geo_layer[0][-1] = self.lats["UR"]
        geo_layer[1][-2] = self.longs["UR"]
        geo_layer[-1][0] = self.lats["LL"]
        geo_layer[-2][1] = self.longs["LL"]
        geo_layer[-1][-1] = self.lats["LR"]
        geo_layer[-2][-2] = self.longs["LR"]
        self.data = np.dstack((self.data, geo_layer))

class Landsat8(AbstractPreprocessor):
    def __init__(self, satnum, meta_url, bands):
        AbstractPreprocessor.__init__(self, satnum, meta_url, bands)
        self.mult0 = {}
        self.add0 = {}
        self.mult1 = {}
        self.add1 = {}
        self.k1 = {}
        self.k2 = {}
        self.elv = None
    def extract_metadata(self, printing = False):
        print(self.meta_url)
        with req.urlopen(self.meta_url) as src:
            i = 0
            string = ""
            for line in src.readlines():
                if printing == True:
                    string = string + "{}:\n{}\n".format(i, line.decode().strip())
                temp = line.decode().split()
                if(i==25):
                    self.lats['UL'] = float(temp[-1])
                elif(i==26):
                    self.longs['UL'] = float(temp[-1])
                elif (i == 27):
                    self.lats['UR'] = float(temp[-1])
                elif (i == 28):
                    self.longs['UR'] = float(temp[-1])
                elif (i == 29):
                    self.lats['LL'] = float(temp[-1])
                elif (i == 30):
                    self.longs['LL'] = float(temp[-1])
                elif (i == 31):
                    self.lats['LR'] = float(temp[-1])
                elif (i == 32):
                    self.longs['LR'] = float(temp[-1])
                elif(i==75):
                    self.elv  = float(temp[-1])
                elif(i==165):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B1'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B1"] = float(temp[0])*math.pow(10, (float(temp[-1])))
                elif (i == 166):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B2'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B2"] = float(temp[0]) *math.pow(10, (float(temp[-1])))
                elif (i == 167):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B3'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B3"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 168):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B4'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B4"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 169):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B5'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B5"] = float(temp[0]) *math.pow(10, (float(temp[-1])))
                elif (i == 170):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B6'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B6"] = float(temp[0]) *math.pow(10, (float(temp[-1])))
                elif (i == 171):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B7'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B7"] = float(temp[0]) *math.pow(10, (float(temp[-1])))
                elif (i == 174):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B10'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B10"] = float(temp[0]) *math.pow(10, (float(temp[-1])))
                elif (i == 175):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult0['B11'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult0["B11"] = float(temp[0]) *math.pow(10, (float(temp[-1])))
                elif (i == 176):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B1'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B1"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 177):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B2'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B2"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 178):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B3'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B3"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 179):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B4'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B4"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 180):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B5'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B5"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 181):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B6'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B6"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 182):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B7'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B7"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 185):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B10'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B10"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 186):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add0['B11'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add0["B11"] = float(temp[0]) * math.pow(10, (float(temp[-1])))

                elif (i == 187):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B1'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B1"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 188):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B2'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B2"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 189):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B3'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B3"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 190):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B4'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B4"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 191):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B5'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B5"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 192):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B6'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B6"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 193):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.mult1['B7'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.mult1["B7"] = float(temp[0]) * math.pow(10, (float(temp[-1])))

                elif (i == 196):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B1'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B1"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 197):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B2'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B2"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 198):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B3'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B3"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 199):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B4'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B4"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 200):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B5'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B5"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 201):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B6'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B6"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif (i == 202):
                    temp = temp[-1]
                    if (re.match(".+E.+", temp) == None):
                        self.add1['B7'] = float(temp)
                    else:
                        temp = temp.split("E")
                        self.add1["B7"] = float(temp[0]) * math.pow(10, (float(temp[-1])))
                elif(i==207):
                    self.k1["B10"] = float(temp[-1])
                elif (i == 208):
                    self.k2["B10"] = float(temp[-1])

                elif (i == 209):
                    self.k1["B11"] = float(temp[-1])
                elif (i == 210):
                    self.k2["B11"] = float(temp[-1])
                i = i+1
                if(i>211):
                    break
        src.close()
        if printing==True:
            return string


    def transform_data(self):
        vis_mults0 = np.array([self.mult0["B2"], self.mult0["B3"], self.mult0["B4"], self.mult0["B5"], self.mult0["B7"], self.mult0["B6"]])
        vis_mults0.shape = (1,1,6)
        vis_adds0 = np.array([self.add0["B2"], self.add0["B3"], self.add0["B4"], self.add0["B5"], self.add0["B7"],self.add0["B6"]])
        vis_adds0.shape = (1, 1, 6)
        vis_mults1 = np.array([self.mult1["B2"], self.mult1["B3"], self.mult1["B4"], self.mult1["B5"], self.mult1["B7"],
                               self.mult1["B6"]])
        vis_mults1.shape = (1, 1, 6)
        vis_adds1 = np.array(
            [self.add1["B2"], self.add1["B3"], self.add1["B4"], self.add1["B5"], self.add1["B7"], self.add1["B6"]])
        vis_adds1.shape = (1, 1, 6)
        therm_mults = np.array([self.mult0["B10"], self.mult0["B11"]])
        therm_mults.shape = (1, 1, 2)
        therm_adds = np.array([self.add0["B10"], self.add0["B11"]])
        therm_adds.shape = (1, 1, 2)
        k1s  = np.array([self.k1["B10"], self.k1["B11"]])
        k1s.shape = (1, 1, 2)
        k2s = np.array([self.k2["B10"], self.k2["B11"]])
        k2s.shape = (1, 1, 2)
        vis = self.data[:,:,:6]
        therm = self.data[:,:,6:]
        self.data = None
        vis = np.multiply(vis, vis_mults0)
        vis = np.add(vis, vis_adds0)
        vis = np.multiply(vis, vis_mults1)
        vis = np.add(vis, vis_adds1)
        vis = np.abs(np.multiply(vis, (1/math.sin(self.elv))))
        therm = np.multiply(therm, therm_mults)
        therm = np.add(therm, therm_adds)
        therm = np.reciprocal(therm)
        therm = np.multiply(k1s, therm)
        therm = np.add(1, therm)
        therm = np.multiply(k2s, np.reciprocal(np.log(therm)))
        self.data = np.dstack((vis, therm))



class Landsat7(AbstractPreprocessor):

    def __init__(self, satnum, meta_url, bands):
        AbstractPreprocessor.__init__(self, satnum, meta_url, bands)
        self.earth2sun_distance = None
        self.gains_table = {}
        self.k1 = 666.09
        self.k2  = 1282.71
        self.solar_zenith = None
        self.lmax_lmin_table = {}
        irradiance =np.array([1970,1842,1547,1044,225.7,82.06])
        irradiance.shape = (1,1,6)
        self.solar_spectral_irradiances = irradiance

    def extract_metadata(self, printing = False):
        with req.urlopen(self.meta_url) as src:
            i = 0
            string = ""
            for line in src.readlines():
                if printing == True:
                    string = string + "{}:\n{}\n".format(i, line.decode().strip())
                temp = line.decode().split()
                if(i==25):
                    self.lats['UL'] = float(temp[-1])
                elif(i==26):
                    self.longs['UL'] = float(temp[-1])
                elif (i == 27):
                    self.lats['UR'] = float(temp[-1])
                elif (i == 28):
                    self.longs['UR'] = float(temp[-1])
                elif (i == 29):
                    self.lats['LL'] = float(temp[-1])
                elif (i == 30):
                    self.longs['LL'] = float(temp[-1])
                elif (i == 31):
                    self.lats['LR'] = float(temp[-1])
                elif (i == 32):
                    self.longs['LR'] = float(temp[-1])
                elif (i == 67):
                    self.solar_zenith = 90-float(temp[-1])
                elif(i==68):
                    self.earth2sun_distance  = float(temp[-1])
                elif(i==85):
                    self.lmax_lmin_table['B1'] = {'max':float(temp[-1])}
                elif (i == 86):
                    self.lmax_lmin_table['B1']['min'] = float(temp[-1].strip('-'))
                elif (i == 87):
                    self.lmax_lmin_table['B2'] = {'max': float(temp[-1])}
                elif (i == 88):
                    self.lmax_lmin_table['B2']['min'] = float(temp[-1].strip('-'))
                elif (i == 89):
                    self.lmax_lmin_table['B3'] = {'max': float(temp[-1])}
                elif (i == 90):
                    self.lmax_lmin_table['B3']['min'] = float(temp[-1].strip('-'))
                elif (i == 91):
                    self.lmax_lmin_table['B4'] = {'max': float(temp[-1])}
                elif (i == 92):
                    self.lmax_lmin_table['B4']['min'] = float(temp[-1].strip('-'))
                elif (i == 93):
                    self.lmax_lmin_table['B5'] = {'max': float(temp[-1])}
                elif (i == 94):
                    self.lmax_lmin_table['B5']['min'] = float(temp[-1].strip('-'))
                elif (i == 97):
                    self.lmax_lmin_table['B6'] = {'max': float(temp[-1])}
                elif (i == 98):
                    self.lmax_lmin_table['B6']['min'] = float(temp[-1].strip('-'))
                elif (i == 99):
                    self.lmax_lmin_table['B7'] = {'max': float(temp[-1])}
                elif (i == 100):
                    self.lmax_lmin_table['B7']['min'] = float(temp[-1].strip('-'))
                i = i+1
                if(i>101):
                    break
        src.close()
        if printing == True:
            return string


    def transform_data(self):
        self.data = np.subtract(self.data, 1.00001)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6']
        mults = []
        adds = []
        for band in bands:
            mults.append(((self.lmax_lmin_table[band]['max'] - self.lmax_lmin_table[band]['min']) / 254))
            adds.append(self.lmax_lmin_table[band]['min'])
        mults = np.array(mults)
        mults.shape = (1, 1, 7)
        adds = np.array(adds)
        adds.shape = (1, 1, 7)
        self.data = np.multiply(self.data, mults)
        self.data = np.add(self.data, adds)
        vis, therm = self.data[:, :, 0:6], self.data[:, :, 6:]
        therm = np.add(therm, 1.00001)
        self.data = None
        solar_scalar = (math.pi * self.earth2sun_distance) / math.cos(self.solar_zenith)
        vis = np.multiply(np.true_divide(vis, self.solar_spectral_irradiances), solar_scalar)
        therm = np.reciprocal(therm)
        therm = np.multiply(self.k1, therm)
        therm = np.add(therm, 1)
        therm = np.log(therm)
        therm = np.reciprocal(therm)
        therm = np.multiply(therm, self.k2)
        self.data = np.dstack((vis, therm))
        vis = None
        therm = None



class Landsat45(AbstractPreprocessor):
    def __init__(self, satnum, meta_url, bands):
        AbstractPreprocessor.__init__(self, satnum, meta_url, bands)
        self.earth2sun_distance = None
        self.gains_table = {}
        self.solar_zenith = None
        self.lmax_lmin_table = {}
        self.irradiance = None
        if(satnum==4):
            self.irradiance  =[1958,1826,1554,1033,214.7,80.70]
            self.k1 = 671.62
            self.k2 = 1284.30
        else:
            self.irradiance = [1958, 1827, 1551, 1036, 214.9, 80.65]
            self.k1 = 607.76
            self.k2 = 1260.56
        self.irradiance = np.array(self.irradiance)
        self.irradiance.shape = (1, 1, 6)
        self.solar_spectral_irradiances = self.irradiance

    def extract_metadata(self, printing = False):
        with req.urlopen(self.meta_url) as src:
            i = 0
            string = ""
            for line in src.readlines():
                if printing == True:
                    string = string + "{}:\n{}\n".format(i, line.decode().strip())
                temp = line.decode().split()
                if(i==26):
                    self.lats['UL'] = float(temp[-1])
                elif(i==27):
                    self.longs['UL'] = float(temp[-1])
                elif (i == 28):
                    self.lats['UR'] = float(temp[-1])
                elif (i == 29):
                    self.longs['UR'] = float(temp[-1])
                elif (i == 30):
                    self.lats['LL'] = float(temp[-1])
                elif (i == 31):
                    self.longs['LL'] = float(temp[-1])
                elif (i == 32):
                    self.lats['LR'] = float(temp[-1])
                elif (i == 33):
                    self.longs['LR'] = float(temp[-1])
                elif (i == 66):
                    self.solar_zenith = 90-float(temp[-1])
                elif(i==67):
                    self.earth2sun_distance  = float(temp[-1])
                elif(i==88):
                    self.lmax_lmin_table['B1'] = {'max':float(temp[-1])}
                elif (i == 89):
                    self.lmax_lmin_table['B1']['min'] = float(temp[-1].strip('-'))
                elif (i ==90):
                    self.lmax_lmin_table['B2'] = {'max': float(temp[-1])}
                elif (i == 91):
                    self.lmax_lmin_table['B2']['min'] = float(temp[-1].strip('-'))
                elif (i == 92):
                    self.lmax_lmin_table['B3'] = {'max': float(temp[-1])}
                elif (i == 93):
                    self.lmax_lmin_table['B3']['min'] = float(temp[-1].strip('-'))
                elif (i == 94):
                    self.lmax_lmin_table['B4'] = {'max': float(temp[-1])}
                elif (i == 95):
                    self.lmax_lmin_table['B4']['min'] = float(temp[-1].strip('-'))
                elif (i == 96):
                    self.lmax_lmin_table['B5'] = {'max': float(temp[-1])}
                elif (i == 97):
                    self.lmax_lmin_table['B5']['min'] = float(temp[-1].strip('-'))
                elif (i == 98):
                    self.lmax_lmin_table['B6'] = {'max': float(temp[-1])}
                elif (i == 99):
                    self.lmax_lmin_table['B6']['min'] = float(temp[-1].strip('-'))
                elif (i == 100):
                    self.lmax_lmin_table['B7'] = {'max': float(temp[-1])}
                elif (i == 101):
                    self.lmax_lmin_table['B7']['min'] = float(temp[-1].strip('-'))
                i = i+1
                if(i>101):
                    src.close()
                    if printing == True:
                        return string
                    break


    def transform_data(self):
        self.data = np.subtract(self.data, 1.00001)
        bands = ['B1','B2','B3','B4','B5','B7', 'B6']
        mults  = []
        adds = []
        for band in bands:
            mults.append(((self.lmax_lmin_table[band]['max'] - self.lmax_lmin_table[band]['min'])/254))
            adds.append(self.lmax_lmin_table[band]['min'])
        mults = np.array(mults)
        mults.shape = (1,1,7)
        adds = np.array(adds)
        adds.shape = (1,1,7)
        self.data = np.multiply(self.data, mults)
        self.data = np.add(self.data ,adds)
        vis, therm = self.data[:, :, 0:6], self.data[:, :, 6:]
        therm = np.add(therm, 1.00001)
        self.data = None
        solar_scalar = (math.pi*self.earth2sun_distance)/math.cos(self.solar_zenith)
        vis = np.multiply(np.true_divide(vis, self.solar_spectral_irradiances), solar_scalar)
        therm  = np.reciprocal(therm)
        therm = np.multiply(self.k1, therm)
        therm = np.add(therm, 1)
        therm = np.log(therm)
        therm = np.reciprocal(therm)
        therm = np.multiply(therm, self.k2)
        self.data = np.dstack((vis, therm))
        vis = None
        therm = None