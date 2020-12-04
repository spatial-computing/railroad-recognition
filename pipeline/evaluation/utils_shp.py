import subprocess
import os, cv2
import xml.etree.ElementTree as ET
from osgeo import gdal, gdalconst, ogr, osr
import zipfile
import numpy as np
from gdalconst import *

#full path to gdal executables>
gdalsrsinfo = r'C:\OSGeo4W64\bin\gdalsrsinfo.exe'
ogr2ogr = r'C:\OSGeo4W64\bin\ogr2ogr.exe'
gdal_translate = r'C:\OSGeo4W64\bin\gdal_translate.exe'
#------------------------------------------------------------------------------

def unzip(zip_path, unzip_path):
    with zipfile.ZipFile(zip_path ,"r") as zip_ref:
        zip_ref.extractall(unzip_path)
    temp = unzip_path.split('\\')
    map_name = temp[-1].replace('_tif', '.tif')
    return os.path.join(unzip_path,map_name)


def reproject(input_vec, in_crs, out_crs, ogr2ogr):
    vector_proj = input_vec.replace('.shp','_proj.shp')
    call = ogr2ogr+' -t_srs "'+out_crs+'" -s_srs "'+in_crs+'" "'+vector_proj+'" "'+input_vec+'"'
    print (call)
    response=subprocess.check_output(call, shell=False)
    print(response)
    return vector_proj

def get_bbox(xml_file):
    bbox = [] #west, east, north, south
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root:
        if child.tag == 'idinfo':
            for cchild in child:
                if cchild.tag =='spdom':
                    for ccchild in cchild: # ccchild is bounding
                        for coor in ccchild:
                            # print coor.text
                            bbox.append(float(coor.text))
    # xmin,xmax,ymin,ymax = bbox
    return bbox

# convert bbox points to goecoordinate of maps
def point_convertor(x, y, input_epsg, out_srs):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    # print(point)
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(input_epsg)
    coordTransform = osr.CoordinateTransformation(inSpatialRef, out_srs)
    # transform point
    point.Transform(coordTransform)
    return [point.GetX(), point.GetY()]

def clip(input_vec,xmin,ymin,xmax,ymax,ogr2ogr):
    vector_clip = input_vec.replace('.shp','_clip.shp')
    # call = '%s -dim 2 -clipsrc %f %f %f %f -nlt POLYGON %s %s ' % (ogr2ogr, xmin,ymin,xmax,ymax, vector_clip, input_vec)
    call = '%s -dim 2 -clipsrc %f %f %f %f %s %s ' % (ogr2ogr, xmin,ymin,xmax,ymax, vector_clip, input_vec)
    print (call)
    response=subprocess.check_output(call, shell=False)
    print(response)
    return vector_clip


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

def vector2raster(input_vec, map_tiff_geo):
# open data
    raster_fn = input_vec.replace('.shp', '.tif')
    raster = gdal.Open(map_tiff_geo)
    shp = ogr.Open(input_vec)
    lyr = shp.GetLayer()
# # Get raster georeference info
    transform = raster.GetGeoTransform()
    # print(transform)
# # Create memory target raster
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
    return raster_fn

def tif2png(in_tif):
    png_fn = in_tif.replace('.tif', '.png')
    call = 'gdal_translate -of PNG -ot Byte "'+in_tif+'" "'+png_fn+'"'
    print(call)
    response=subprocess.check_output(call, shell=True)
    print(response)
    xml_fn = png_fn + '.aux.xml'
    os.remove(xml_fn)
    return png_fn

def dilated(png_fn):
    dilated_fn = png_fn.replace('.png', '_dilated.png')
    img = cv2.imread(png_fn,0)
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    cv2.imwrite(dilated_fn, dilation)
    return dilated_fn

def geo2img_coor(x, y, path):
    dataset = gdal.Open( path, GA_ReadOnly )
    adfGeoTransform = dataset.GetGeoTransform()
    dfGeoX=float(x)
    dfGeoY =float(y)
    det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4]
    # X = (int)(((dfGeoX - adfGeoTransform[0]) / adfGeoTransform[1]))
    # Y = (int)(((dfGeoY - adfGeoTransform[3]) / adfGeoTransform[5]))
    X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
    adfGeoTransform[3]) * adfGeoTransform[2]) / det
    Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
    adfGeoTransform[0]) * adfGeoTransform[4]) / det
    return [int(Y),int(X)]

def points_generator(raster, save_path, n_points=400):
    img = cv2.imread(raster, 0)
    index_x, index_y = np.where(img==255)
    print(index_x.shape)
    rand = np.random.randint(0, index_x.shape[0], n_points)
    points = np.array([index_x[rand], index_y[rand]])
    points = np.swapaxes(points,0,1)
    print(points.shape)
    np.savetxt(save_path, points, fmt='%d', delimiter=',')
    return save_path
