
import cv2
import numpy as np
import os
import sys

MAP_DIR = 'C:\Users\weiweiduan\Documents\Map_proj_data\CO\CO_Louisville_233659_1942_31680_bag\data'
MAP_NAME = 'CO_Louisville_233659_1942_31680_geo'
ALIGN_VECTOR_NAME = 'louisville_railroads_1942_aligned'
align_vector_path = os.path.join(MAP_DIR, ALIGN_VECTOR_NAME)
output_path = os.path.join(MAP_DIR, ALIGN_VECTOR_NAME+'.png')
map_path = os.path.join(MAP_DIR, MAP_NAME+'.png')

map_img = cv2.imread(map_path)
print map_img.shape
count = 0
aligned_shp = np.zeros((map_img.shape[0],map_img.shape[1]))
for root, dirs, files in os.walk(align_vector_path):
    print files
    for f in files:
        print f
        points = np.loadtxt(root+'/'+f, delimiter=',', dtype='int32', skiprows=1)
        if points.shape[0] != 0:
            print points.shape
            for p in range(0, points.shape[0]-1):
                 # cv2.line(aligned_shp,(points[p][1],points[p][0]),(points[p+1][1],points[p+1][0]),(255-count,255-count,255-count),1)
                cv2.line(aligned_shp,(points[p][0],-points[p][1]),(points[p+1][0],-points[p+1][1]),(255,255,255),3)
        count += 1

cv2.imwrite(output_path, aligned_shp)
