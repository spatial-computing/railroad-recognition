import subprocess
import os
import cv2
import numpy as np
from osgeo import gdal, gdalconst, ogr, osr

#------------------------------------------------------------------------------
def create_filtered_shapefile(value, filter_field, in_shapefile, out_shapefile):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(in_shapefile, 0)
    input_layer = dataSource.GetLayer()
    query_str = '{} = {}'.format(filter_field, value)
    print(query_str)
    err = input_layer.SetAttributeFilter(query_str)
    print(err)
    # Copy Filtered Layer and Output File
    out_ds = driver.CreateDataSource(out_shapefile)
    out_layer = out_ds.CopyLayer(input_layer, str(value))
    del input_layer, out_layer, out_ds
    return out_shapefile

#parameters:
vector_orig = r"C:/Users/weiweiduan/Documents/Map_proj_data/CA/Shape/NHD_H_California_State_Shape/Shape/NHDWaterbody.shp"
map_tiff_geo = r"D:\maps\zips\CA_Big_Swamp_288420_1990_24000_bag\data\CA_Big_Swamp_288420_1990_24000_geo.tif"

#full path to gdal executables>
gdalsrsinfo = r'C:\OSGeo4W64\bin\gdalsrsinfo.exe'
ogr2ogr = r'C:\OSGeo4W64\bin\ogr2ogr.exe'
gdal_translate = r'C:\OSGeo4W64\bin\gdal_translate.exe'

#the USGS quadrangle shapefile:
quadrangles = r'D:\QUADRANGLES\Quadr_US\USGS_24k_Topo_Map_Boundaries_NAD83inNAD27.shp'

#the name and state of current map quadrangle:
quadrangle_name = "Big Swamp"
quadrangle_state = "California"

#------------------------------------------------------------------------------

call = gdalsrsinfo+' -o proj4 "'+vector_orig+'"'
crs_vector=subprocess.check_output(call, shell=True).strip().replace("'","")
print (crs_vector)
call = gdalsrsinfo+' -o proj4 "'+map_tiff_geo+'"'
crs_raster=subprocess.check_output(call, shell=True).strip().replace("'","")
print (crs_raster)

# use USGS quadrangle geometry to clip vector exactly to map area
# first select quadrangle
workdir,quads = os.path.split(quadrangles)
quad_select=workdir+os.sep+'quadr_'+quadrangle_name.replace(' ','_')+'_'+quadrangle_state+'.shp'
call = """%s -where "QUAD_NAME='%s' AND ST_NAME1='%s'" %s %s""" % (ogr2ogr, quadrangle_name, quadrangle_state, quad_select, quadrangles)
print (call)
response=subprocess.check_output(call, shell=True)
print response

# Reproject vector geometry to same projection as raster
vector_proj = vector_orig.replace('.shp','_proj.shp')
call = ogr2ogr+' -t_srs "'+crs_raster+'" -s_srs "'+crs_vector+'" "'+vector_proj+'" "'+vector_orig+'"'
print (call)
response=subprocess.check_output(call, shell=True)
print response

#clip
vector_clip=vector_orig.replace('.shp','_clip.shp')
call = '%s -dim 2 -clipsrc %s %s %s ' % (ogr2ogr, quad_select, vector_clip, vector_orig)
print (call)
response=subprocess.check_output(call, shell=True)
print response

# filter wetlands from water body
vector_filter = vector_clip.replace('.shp','_filter.shp')
create_filtered_shapefile(466, 'FType', vector_clip, vector_filter)

# rasterize
# open data
raster_fn = vector_filter.replace('.shp', '.tif')
raster = gdal.Open(map_tiff_geo)
shp = ogr.Open(vector_filter)
lyr = shp.GetLayer()
# Get raster georeference info
transform = raster.GetGeoTransform()
# Create memory target raster
x_res, y_res = raster.RasterXSize, raster.RasterYSize
target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(transform)
raster_srs = osr.SpatialReference()
raster_srs.ImportFromWkt(raster.GetProjectionRef())
target_ds.SetProjection(raster_srs.ExportToWkt())
# Rasterize
err = gdal.RasterizeLayer(target_ds, [1], lyr,burn_values=[255])
if err != 0:
    print(err)
del target_ds

# convert tif to png
png_fn = raster_fn.replace('.tif', '.png')
call = 'gdal_translate -of PNG -ot Byte "'+raster_fn+'" "'+png_fn+'"'
print(call)
response=subprocess.check_output(call, shell=True)
print (response)
