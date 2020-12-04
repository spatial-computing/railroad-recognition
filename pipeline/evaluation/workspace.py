import numpy as np
import os, cv2
import copy
from skeletonize import make_skeleton
from buffer import buffer
import utils_shp

'''
inputs for the evaluation are 
ground truth shape file
segmentation results
'''

data_dir = 'C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_geo_tif'
location_name = 'CA_Bray_100414_2001_24000_geo'
gt_name = 'CA_Bray_railroads_gt_buffer0.shp'
pred_name = 'CA_Bray_100414_2001_24000_geo_pointrend_unet_pred.png'
buffer_size = 2

pred_path = os.path.join(data_dir, 'res', pred_name)
gt_shp_path = os.path.join(data_dir, 'ground_truth', gt_name)
map_tif_path = os.path.join(data_dir, location_name+'.tif')

# Step 1: rasterize the ground truth shapefile
gt_tif_path = utils_shp.vector2raster(gt_path, map_tif_path)
gt_png_path = utils_shp.tif2png(gt_tif_path)

# Step 2: load the bounding box coordinates to remove the white borders of the map
bbox_file = os.path.join(data_dir, 'bbox.txt')
points = np.loadtxt(bbox_file, dtype='int32', delimiter=',')
start_point, end_point = points[0], points[1]
print(start_point, end_point)

# Step 3: skeletonize the segmentation results
img_copy, pred_ske = make_skeleton(pred_path)
pred_ske = pred_ske.astype('uint')

# Step 4: buffer the segmentation results and ground truth
pred_buffer = buffer(pred_ske, buffer_size=buffer_size)
gt_map = cv2.imread(gt_png_path,0) / 255
gt_buffer = buffer(gt_map, buffer_size=buffer_size)

# Step 5 calculate correctness and completeness
overlap_map = gt_buffer * pred_ske
fp_map = pred_ske - overlap_map


tp = np.count_nonzero(overlap_map)
fp = np.count_nonzero(fp_map)

correctness = tp / (tp+fp)
print('correctness = ', correctness)

overlap_comp_map = gt_map * pred_buffer

fn_map = gt_map - overlap_comp_map

tp_comp = np.count_nonzero(overlap_comp_map)
fn = np.count_nonzero(fn_map)

completeness = tp_comp / (tp_comp+fn)
print('completeness = ', completeness)

