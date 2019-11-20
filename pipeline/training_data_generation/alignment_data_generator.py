import cv2
import os
from osgeo import ogr
import gdal
from gdalconst import *
import subprocess
import math
import sys

MAP_DIR = 'C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_450547_1957_24000_bag\data'
MAP_NAME = 'CO_Louisville_450547_1957_24000_geo'
SHP_PATH = 'C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_450543_1965_24000_bag\data\\railroad\Trans_RailFeature_Louisville_clip_Louisville_proj.shp'
OUTPUT_NAME = 'louisville_railroads_1957'
output_path = os.path.join(MAP_DIR, OUTPUT_NAME)
map_path = os.path.join(MAP_DIR, MAP_NAME+'.tif')


def tif_to_png(MAP_DIR, MAP_NAME):
    img = cv2.imread(os.path.join(MAP_DIR, MAP_NAME+'.tif'))
    print("converting tif to png")
    cv2.imwrite(os.path.join(MAP_DIR, MAP_NAME+'.png'), img)

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



ds = ogr.Open(SHP_PATH)
layer = ds.GetLayer(0)
f = layer.GetNextFeature()

if not os.path.exists(output_path):
    os.makedirs(output_path)
qgis = open('./qgis.txt', 'w')
count = 0
while f:
    geom = f.GetGeometryRef()
    if geom != None:
    # points = geom.GetPoints()
        points = geom.ExportToJson()
        points = eval(points)
        outputfile = open(os.path.join(output_path,str(count)+".txt"),"w")
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
