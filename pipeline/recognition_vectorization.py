import cv2
import numpy as np
import os
from osgeo import ogr
import gdal
from gdalconst import *


MAP_DIR = 'C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_450547_1957_24000_bag\data'
MAP_NAME = 'CO_Louisville_450547_1957_24000_geo'
SHP_PATH = 'C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_450543_1965_24000_bag\data\\waterlines\\tl_2017_08013_linearwater_clip_proj.shp'
# SHP_PATH = 'C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_450543_1965_24000_bag\data\\roads\\tl_2017_08013_roads_clip_proj.shp'
OUTPUT_NAME = 'waterlines.txt'
ALIGN_FOLDER_NAME = 'louisville_railroads_1957_aligned'
output_path = os.path.join(MAP_DIR, OUTPUT_NAME)
outputfile = open(output_path,"w")
ALIGN_PATH = os.path.join(MAP_DIR, ALIGN_FOLDER_NAME)
map_path = os.path.join(MAP_DIR, MAP_NAME+'.tif')


def convert_to_image_coord(x,y,path): # convert geocoord to image coordinate
    dataset = gdal.Open( path, GA_ReadOnly )
    adfGeoTransform = dataset.GetGeoTransform()

    dfGeoX=float(x)
    dfGeoY =float(y)
    det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4]

    X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
    adfGeoTransform[3]) * adfGeoTransform[2]) / det

    Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
    adfGeoTransform[0]) * adfGeoTransform[4]) / det
    return [int(Y),int(X)]

def generator_from_shp(map_path, SHP_PATH):
    ds = ogr.Open(SHP_PATH)
    layer = ds.GetLayer(0)
    f = layer.GetNextFeature()
    qgis = open('qgis.txt', 'w')
    count = 0
    while f:
        geom = f.GetGeometryRef()
        if geom != None:
        # points = geom.GetPoints()
            points = geom.ExportToJson()
            points = eval(points)
            if points['type'] == "MultiLineString":
                for i in points["coordinates"]:
                    for j in i:
                        tmpt = j
                        p = convert_to_image_coord(tmpt[0],tmpt[1],map_path)
                        print p
                        outputfile.writelines(str(p[0])+","+str(p[1])+'\n')
                        qgis.writelines(str(p[1])+","+str(-p[0])+'\n')
            elif points['type'] ==  "LineString":
                for i in points['coordinates']:
                    tmpt = i
                    p = convert_to_image_coord(tmpt[0],tmpt[1],map_path)
                    outputfile.writelines(str(p[0])+","+str(p[1])+'\n')
                    qgis.writelines(str(p[1])+","+str(-p[0])+'\n')

        count += 1
        f = layer.GetNextFeature()
    print count

def generator_from_rand_points():
    return

def generator_from_txt_folder(ALIGN_PATH):
    for root, dirs, files in os.walk(ALIGN_PATH):
        for f in files:
            print f
            points = np.loadtxt(root+'/'+f, delimiter=',', dtype='int32', skiprows=1)
            if points.shape[0] != 0:
                print points.shape
                for p in range(0, points.shape[0]-1):
                    outputfile.writelines(str(-points[p][1])+","+str(points[p][0])+'\n')

# generator_from_txt_folder(ALIGN_PATH)
generator_from_shp(map_path, SHP_PATH)
