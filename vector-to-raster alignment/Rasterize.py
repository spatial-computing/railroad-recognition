import cv2
import numpy as np
import os
import sys

map_path = sys.args[2]
map_img = cv2.imread(map_path)
print map_img.shape
count = 0
aligned_shp = np.zeros((map_img.shape[0],map_img.shape[1],map_img.shape[1]))
for root, dirs, files in os.walk(sys.args[2]):
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

cv2.imwrite(sys.args[3], aligned_shp)


