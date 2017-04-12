import cv2
import sys
import subprocess
gdal_translate = r'C:\OSGeo4W64\bin\gdal_translate.exe'
gdal_rasterize=r'C:\OSGeo4W64\bin\gdal_rasterize.exe'

m=r'C:\Users\Vinil\Desktop\vinil\temp\quality_cal\matched_ref'
map=m+'.tif'
map_compressed=m+'_cropped.png'

bounding_shapefile=r"C:\Users\Vinil\Desktop\vinil\temp\quality_cal\boundingbox.shp"

call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(1,"boundingbox",bounding_shapefile,map)
print call
response=subprocess.check_output(call, shell=True)
print response

call=gdal_translate+" -of PNG %s %s"%(map,map_compressed)
print call
response=subprocess.check_output(call, shell=True)
print response

call=gdal_rasterize+' -b 1 -burn %d -l %s %s %s'%(0,"boundingbox",bounding_shapefile,map)
print call
response=subprocess.check_output(call, shell=True)
print response




output_file=open("bounding_coordinates.txt","w")
start_x=0
start_y=0
end_x=0
end_y=0

img=cv2.imread(map_compressed,0)
width=img.shape[1]
height=img.shape[0]
flag=0
for y in range(0,height,1):
    for x in range(0,width,1):
        if img[y,x]==1:
            start_x=x
            start_y=y
            flag=1
            break
    if flag:
        break

for x in range(start_x,width,1):
    if img[start_y,x]==0:
        end_x=x-1
        break

for y in range(start_y,height,1):
    if img[y,end_x]==0:
        end_y=y-1
        break


output_file.writelines(str(start_x)+"\n")
output_file.writelines(str(start_y)+"\n")
output_file.writelines(str(end_x)+"\n")
output_file.writelines(str(end_y)+"\n")
print start_x
print start_y
print end_x
print end_y