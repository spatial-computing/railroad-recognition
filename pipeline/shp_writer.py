import os
import sys
from osgeo import ogr, gdal, osr

def pixel2coord(coor_path, x, y):
    raster = gdal.Open(coor_path)
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def createShapefile(shapefileName, line, coor_path):
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Creating a new data source and layer
    if os.path.exists(shapefileName):
        driver.DeleteDataSource(shapefileName)

    ds = driver.CreateDataSource(shapefileName)
    if ds is None:
        print 'Could not create file'
        sys.exit(1)

    layer = ds.CreateLayer('layerName', geom_type = ogr.wkbLineString)
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


