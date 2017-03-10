
import cv2
import time
import gdal
from gdalconst import *

gdal_translate = r'C:\OSGeo4W64\bin\gdal_translate.exe'
dataset = gdal.Open( "CA_Bray_100414_2001_24000_bag\data\CA_Bray_100414_2001_24000_geo.tif", GA_ReadOnly )
padfTransform = dataset.GetGeoTransform()

img=cv2.imread("test.jpg",0) #open rasterised image in grayscale
map_geo=cv2.imread("CA_Bray_100414_2001_24000_bag\data\CA_Bray_100414_2001_24000_geo.tif")
width=img.shape[1]
height=img.shape[0]

pos=0
neg=0
step=4
outfile_pos=open("positive_image_cordinates_step"+str(step)+"_.txt","w")
outfile_neg=open("negative_image_cordinates_step"+str(step)+"_.txt","w")
start_time=time.clock()
for y in range(24,height-24,step):
    for x in range(24,width-24,step):
        if img[y,x]==255:

            pos+=1
            outfile_pos.writelines(str(y)+","+str(x)+"\n")
            # new_img=map_geo[y-24:y+24,x-24:x+24]
            # cv2.imwrite("temp_data/pos_new_img"+str(pos)+".tif",new_img)

            #if you want to crop using gdal-co-ordinates, use the following lines
            # xp = padfTransform[0] + x*padfTransform[1] + y*padfTransform[2]
            # yp = padfTransform[3] + x*padfTransform[4] + y*padfTransform[5]
            #outfile_pos.writelines(str(xp)+","+str(yp)+"\n")
            #gdal_cmd = gdal_translate+' -of GTiff -projwin %s %s %s %s %s %s' % (xp-24,yp+24,xp+24,yp-24,map_tiff_geo,"temp_data/pos_new_img"+str(pos)+".tif")
        else :
            outfile_neg.writelines(str(y)+","+str(x)+"\n")
            # new_img=map_geo[y-24:y+24,x-24:x+24]
            # cv2.imwrite("temp_data/neg_new_img"+str(pos)+".tif",new_img)

            #if you want to crop using gdal-co-ordinates, use the following lines
            # xp = padfTransform[0] + x*padfTransform[1] + y*padfTransform[2]
            # yp = padfTransform[3] + x*padfTransform[4] + y*padfTransform[5]
            #outfile_pos.writelines(str(xp)+","+str(yp)+"\n")
            #gdal_cmd = gdal_translate+' -of GTiff -projwin %s %s %s %s %s %s' % (xp-24,yp+24,xp+24,yp-24,map_tiff_geo,"temp_data/neg_new_img"+str(pos)+".tif")
            neg+=1

print time.clock() - start_time, "seconds"
print pos,neg
outfile_neg.close()
outfile_pos.close()


