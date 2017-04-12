import cv2
import random

def is_safe(x,y):
    if y>24 and  y <height-24 and x>24 and x<width-24 and img_perfect[y,x]<235:
        return True
    return False

map_geo=cv2.imread(r"D:\vinil\CA_Bray_100414_2001_24000_geo.tif")

m=r'C:\Users\Vinil\Desktop\vinil\temp\quality_cal\matched_ref'
map=m+'.tif'
map_compressed=m+'.png'

img_perfect=cv2.imread(map_compressed,0)
img_orig=cv2.imread("original0.png",0)
width=img_perfect.shape[1]
height=img_perfect.shape[0]
step=1

outfile_pos0=open("testing_positive_coodinates_buffer0.txt","w")
outfile_pos1=open("testing_positive_coordinates_buffer1.txt","w")
outfile_pos2=open("testing_positive_coordinates_buffer2.txt","w")
outfile_pos3=open("testing_positive_coordinates_buffer3.txt","w")

for y in range(24,height-24,step):
    for x in range(24,width-24,step):
        if img_orig[y,x]==255:
            outfile_pos0.writelines(str(y)+","+str(x)+"\n");
        if img_perfect[y,x]>=254:
            outfile_pos1.writelines(str(y)+","+str(x)+"\n");
        if img_perfect[y,x]>=253:
            outfile_pos2.writelines(str(y)+","+str(x)+"\n");
        if img_perfect[y,x]>=252:
            outfile_pos3.writelines(str(y)+","+str(x)+"\n");


outfile_pos0.close()
outfile_pos1.close()
outfile_pos2.close()
outfile_pos3.close()
#
# import os
# import sys
# import subprocess
#
#
# gdalsrsinfo = r'C:\OSGeo4W64\bin\gdalsrsinfo.exe'
# ogr2ogr = r'C:\OSGeo4W64\bin\ogr2ogr.exe'
# gdal_translate = r'C:\OSGeo4W64\bin\gdal_translate.exe'
# gdal_rasterize=r'C:\OSGeo4W64\bin\gdal_rasterize.exe'
#
# vector="correct_vector"
#
# list=[]
#
# #list.append(["perfect_vector/"+vector+".shp",0])
# for i in range(0,4,1):
#     vector_diss="perfect_vector/"+vector+str(i)+".shp"
#     call = """%s -f "ESRI Shapefile" %s %s.shp -dialect sqlite -sql "SELECT ST_Union(ST_buffer(Geometry,%f)) FROM '%s'" """ % (ogr2ogr, vector_diss, vector,i+0.5, vector)
#     list.append([vector_diss,i])
#     print call
#     response=subprocess.check_output(call, shell=True)
#     print response
#
# # for i in range(25,51,25):
# #     vector_diss="perfect_vector/"+vector+str(i)+".shp"
# #     call = """%s -f "ESRI Shapefile" %s %s.shp -dialect sqlite -sql "SELECT ST_Union(ST_buffer(Geometry,%f)) FROM '%s'" """ % (ogr2ogr, vector_diss, vector,i-0.5, vector)
# #     list.append([vector_diss,i])
# #     print call
# #     response=subprocess.check_output(call, shell=True)
# #     print response
# m=r'C:\Users\Vinil\Desktop\vinil\temp\quality_cal\matched_ref'
# map=m+'.tif'
# # map_compressed=m+'.png'
# i=0
# for entry in (list):
#     layer=entry[0].replace("perfect_vector/","")
#     layer=layer.replace(".shp","")
#     print layer
#     call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(255-entry[1],layer,entry[0],map)
#     # call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(0,layer,entry[0],map)
#     print call
#     response=subprocess.check_output(call, shell=True)
#     print response
#     map_compressed=m+str(i)+"_test_.png"
#     call=gdal_translate+" -of PNG %s %s"%(map,map_compressed)
#     print call
#     response=subprocess.check_output(call, shell=True)
#     print response
#     i+=1