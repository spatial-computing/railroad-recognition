from osgeo import ogr
import os
import gdal
from gdalconst import *
import subprocess
import math
import sys

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


map_path = sys.args[1]
ds = ogr.Open(sys.args[2])
layer = ds.GetLayer(0)
f = layer.GetNextFeature()

count = 0
while f:
    geom = f.GetGeometryRef()
    if geom != None:
    # points = geom.GetPoints()
        points = geom.ExportToJson()
        points = eval(points)
        outputfile = open(sys.args[3]+str(count)+".txt","w")
        if points['type'] == "MultiLineString":
            for i in points["coordinates"]:
                for j in i:
                    tmpt = j
                    p = convert_to_image_coord(tmpt[0],tmpt[1],map_path)
                    print p
                    outputfile.writelines(str(p[0])+","+str(p[1])+'\n')
        elif points['type'] ==  "LineString":
            for i in points['coordinates']:
                tmpt = i
                p = convert_to_image_coord(tmpt[0],tmpt[1],map_path)
                outputfile.writelines(str(p[0])+","+str(p[1])+'\n')

    count += 1
    f = layer.GetNextFeature()
print count
