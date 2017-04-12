import os
import sys
import subprocess


gdalsrsinfo = r'C:\OSGeo4W64\bin\gdalsrsinfo.exe'
ogr2ogr = r'C:\OSGeo4W64\bin\ogr2ogr.exe'
gdal_translate = r'C:\OSGeo4W64\bin\gdal_translate.exe'
gdal_rasterize=r'C:\OSGeo4W64\bin\gdal_rasterize.exe'

vector="correct_vector"

list=[]

roads=r"C:\Users\Vinil\Desktop\vinil\TRAN_6_California_GU_STATEORTERRITORY\rods.shp"
water=r"C:\Users\Vinil\Downloads\tl_2016_06093_linearwater_new\water.shp"

mountain_peak="C:\Users\Vinil\Downloads\USA_Mountain_Peaks\mountain_peaks_clip_proj.shp"
roads_color=200
water_color=199
mountain_color=100

# roads_color=0
# water_color=0
# mountain_color=0

#list.append(["perfect_vector/"+vector+".shp",0])
for i in range(0,21,1):
    vector_diss="perfect_vector/"+vector+str(i)+".shp"
    call = """%s -f "ESRI Shapefile" %s %s.shp -dialect sqlite -sql "SELECT ST_Union(ST_buffer(Geometry,%f)) FROM '%s'" """ % (ogr2ogr, vector_diss, vector,i+0.5, vector)
    list.append([vector_diss,i])
    print call
    response=subprocess.check_output(call, shell=True)
    print response

# for i in range(25,51,25):
#     vector_diss="perfect_vector/"+vector+str(i)+".shp"
#     call = """%s -f "ESRI Shapefile" %s %s.shp -dialect sqlite -sql "SELECT ST_Union(ST_buffer(Geometry,%f)) FROM '%s'" """ % (ogr2ogr, vector_diss, vector,i-0.5, vector)
#     list.append([vector_diss,i])
#     print call
#     response=subprocess.check_output(call, shell=True)
#     print response
m=r'C:\Users\Vinil\Desktop\vinil\temp\quality_cal\matched_ref'
map=m+'.tif'
map_compressed=m+'.png'

call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(roads_color,"rods",roads,map)
print call
response=subprocess.check_output(call, shell=True)
print response
call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(water_color,"water",water,map)
print call
response=subprocess.check_output(call, shell=True)
print response
call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(mountain_color,"mountain_peaks_clip_proj",mountain_peak,map)
print call
response=subprocess.check_output(call, shell=True)
print response
for entry in reversed(list):
    layer=entry[0].replace("perfect_vector/","")
    layer=layer.replace(".shp","")
    print layer
    call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(255-entry[1],layer,entry[0],map)
    # call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(0,layer,entry[0],map)
    print call
    response=subprocess.check_output(call, shell=True)
    print response

call=gdal_translate+" -of PNG %s %s"%(map,map_compressed)
print call
response=subprocess.check_output(call, shell=True)
print response