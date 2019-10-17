import cv2
import numpy as np
import os

# map_path = "C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_bag\data\CA_Bray_100414_2001_24000_geo.tif"
# map_path = "C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_450543_1965_24000_bag\data\CO_Louisville_450543_1965_24000_geo.tif"
map_path = "C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_296903_1950_62500_geo\CA_Bray_296903_1950_62500_geo.png"
map_img = cv2.imread(map_path)
print map_img.shape
count = 0
aligned_shp = np.zeros((map_img.shape[0],map_img.shape[1],map_img.shape[2]))
for root, dirs, files in os.walk("C:\Users\weiweiduan\Documents\Alignment_RL\Bray_railroads_1950_alignment"):
    print files
    for f in files:
        print f
        points = np.loadtxt(root+'/'+f, delimiter=',', dtype='int32', skiprows=1)
        if points.shape[0] != 0:
            print points.shape
            for p in range(0, points.shape[0]-1):
                 # cv2.line(aligned_shp,(points[p][1],points[p][0]),(points[p+1][1],points[p+1][0]),(255-count,255-count,255-count),1)
                cv2.line(aligned_shp,(points[p][0],-points[p][1]),(points[p+1][0],-points[p+1][1]),(255,255,255),2)
        count += 1
# points = []
# count = 0
# for line in seg2:
#     if count == 0:
#         count += 1
#     else:
#         tmpt = line.strip().split(",")
#         # print tmpt
#         points.append([-int(tmpt[1]), int(tmpt[0])])
#
# for p in range(len(points)-1):
#     cv2.line(aligned_shp,(points[p][1],points[p][0]),(points[p+1][1],points[p+1][0]),(255,255,255),1)
# print aligned_shp.shape
cv2.imwrite('CA_Bray_railroads_aligned_1950.png', aligned_shp)


