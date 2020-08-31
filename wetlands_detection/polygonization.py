import subprocess
import os
import cv2
import numpy as np
from osgeo import gdal, gdalconst, ogr, osr
import argparse

def png2tif(png_fn, tif_fn):
    tifDriver = gdal.GetDriverByName("GTiff")
    png = gdal.Open(png_fn)
    tifDriver.CreateCopy(tif_fn, png, 0, [])
    del tifDriver, png
    return

def set_proj(tif_fn, map_fn):
    raster = gdal.Open(map_fn)
    transform = raster.GetGeoTransform()
    target_ds = gdal.Open(tif_fn, gdal.GA_Update)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    print(raster.GetProjectionRef())
    target_ds.SetGeoTransform(transform)
    target_ds.SetProjection(raster_srs.ExportToWkt())
    del target_ds
    return

def polynize(img, shp_path):
        # mapping between gdal type and ogr field type
    type_mapping = {gdal.GDT_Byte: ogr.OFTInteger,
                    gdal.GDT_UInt16: ogr.OFTInteger,
                    gdal.GDT_Int16: ogr.OFTInteger,
                    gdal.GDT_UInt32: ogr.OFTInteger,
                    gdal.GDT_Int32: ogr.OFTInteger,
                    gdal.GDT_Float32: ogr.OFTReal,
                    gdal.GDT_Float64: ogr.OFTReal,
                    gdal.GDT_CInt16: ogr.OFTInteger,
                    gdal.GDT_CInt32: ogr.OFTInteger,
                    gdal.GDT_CFloat32: ogr.OFTReal,
                    gdal.GDT_CFloat64: ogr.OFTReal}
    ds = gdal.Open(img)
    prj = ds.GetProjection()
    srcband = ds.GetRasterBand(1)
    dst_layername = "Shape"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(shp_path)
    srs = osr.SpatialReference(wkt=prj)

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    raster_field = ogr.FieldDefn('id', type_mapping[srcband.DataType])
    dst_layer.CreateField(raster_field)
    err = gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
    if err != 0:
        print(err)
    del img, ds, srcband, dst_ds, dst_layer
    return 0

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

def filtered_shapefile_poly(in_shp, out_shp):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(in_shp, 0)
    input_layer = dataSource.GetLayer()
    out_ds = driver.CreateDataSource(out_shp)
    proj = input_layer.GetSpatialRef()
    out_lyr = out_ds.CreateLayer(out_shp.split(".")[0], proj, ogr.wkbPolygon )
    #    copy the schema of the original shapefile to the destination shapefile
    lyr_def = input_layer.GetLayerDefn ()
    for i in range(lyr_def.GetFieldCount()):
        out_lyr.CreateField ( lyr_def.GetFieldDefn(i) )
    for feature in input_layer:
        geom = feature.GetGeometryRef()
        if geom != None:
            points = geom.ExportToJson()
            points = eval(points)
            if len(points['coordinates'][0]) <=5:
                continue
        out_lyr.CreateFeature(feature)
    return 0

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str)
parser.add_argument("--tif_map_path", type=str)
png_path = args.pred_path
map_path = args.tif_map_path

tif_path = png_path.replace('.png', '.tif')
vector = tif_path.replace('.tif', '.shp')
# convert png to tif
png2tif(png_path, tif_path)
# pass the project of geo_tif to detection tif
set_proj(tif_path, map_path)
# convert raster to vector
polynize(tif_path, vector)

# vector_fg = vector.replace('.shp', '_fg.shp')
# create_filtered_shapefile(255, 'id', vector, vector_fg)
vector_poly = vector.replace('.shp', '_poly.shp')
# filter out individual bounding box
filtered_shapefile_poly(vector, vector_poly)
