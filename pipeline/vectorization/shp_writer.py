import os
import sys
from osgeo import ogr, gdal, osr

def pixel2coord(coor_path, x, y):
    raster = gdal.Open(coor_path)
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def coor2epsg(coor_path, epsg, x, y):
    # get CRS from dataset
    ds = gdal.Open(coor_path)
    crs = osr.SpatialReference()
    # crs.ImportFromProj4(ds.GetProjection())
    crs.ImportFromWkt(ds.GetProjectionRef())
# create lat/long crs with WGS84 datum
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(epsg)
    t = osr.CoordinateTransformation(crs, crsGeo)
    (lat, long, z) = t.TransformPoint(x, y)
    return (lat, long)

def createShapefile(shapefileName, line, coor_path, epsg=None):
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Creating a new data source and layer
    if os.path.exists(shapefileName):
        driver.DeleteDataSource(shapefileName)
    if epsg == None:
        srs = gdal.Open(coor_path).GetProjection()
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4(srs)
    else:
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(epsg)

    ds = driver.CreateDataSource(shapefileName)
    if ds is None:
        print 'Could not create file'
        sys.exit(1)

    layer = ds.CreateLayer('layerName', spatial_reference, geom_type = ogr.wkbLineString)
    # add a field to the output
    fieldDefn = ogr.FieldDefn('fieldName', ogr.OFTReal)
    layer.CreateField(fieldDefn)
    cnt = 0
    for v in line:
        cnt += 1
        lineString = ogr.Geometry(ogr.wkbLineString)
        for m in v:
            p = m.split(' ')
            x, y = float(p[0]), float(p[1])
            xp, yp = pixel2coord(coor_path,x,y)
            if epsg != None:
                xp, yp = coor2epsg(coor_path, epsg, xp, yp)
            lineString.AddPoint(xp,yp)

        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(lineString)
        feature.SetField('fieldName', 'LineString')
        layer.CreateFeature(feature)
        lineString.Destroy()
        feature.Destroy()
    ds.Destroy()
    print "Shapefile created"


