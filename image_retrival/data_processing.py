from utils import load_data, data_augmentation
import numpy as np
import os
import cv2

   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

TARGET_SAMPLE_DIR = "./data/bray_2001/target_samples"
MAP_PATH = './data/bray_2001/CA_Bray_100414_2001_24000_geo.png'
MASK_PATH = './data/bray_2001/wetlands.png'
SHIFT_LIST = [-20,-15,-10,-5,0,5,10,15,20] 
ROTATION_ANGLE = []
for i in range(0, 360, 90):
	ROTATION_ANGLE.append(i)
IMG_SIZE = 80
STRIDE = 30

x_u, img_names = load_data.load_all_data(MAP_PATH, MASK_PATH, IMG_SIZE, STRIDE)

x_l, target_name = load_data.load_wetland_samples(TARGET_SAMPLE_DIR)

x_l_aug = data_augmentation.data_aug(x_l, SHIFT_LIST, ROTATION_ANGLE)

