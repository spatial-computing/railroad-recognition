import os
import numpy as np
import cv2
from keras.backend import cast_to_floatx

    
def load_all_data(map_path, mask_path, img_size, stride, flip=False):
    x_test, img_name = [], []
    map_img = cv2.imread(map_path)
    #map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path,0)
    #print(map_img.shape, mask.shape)
    if flip == True:
        mask = 255 - mask
    for i in range(0, map_img.shape[0]-img_size, stride):
        for j in range(0, map_img.shape[1]-img_size, stride):
            if mask_path != '':
                cropped_img = mask[i:i+img_size, j:j+img_size]
                nums = np.where(cropped_img==255)[0].shape[0]
                if nums == img_size*img_size:
                    img = map_img[i:i+img_size, j:j+img_size]
                    if img_size < 48:
                        img = cv2.resize(img, (48,48))
                    x_test.append(img)
                    img_name.append(str(i)+'_'+str(j))
            else:
                img = map_img[i:i+img_size, j:j+img_size]
                if img_size < 48:
                    img = cv2.resize(img, (48,48))
                x_test.append(img)
                img_name.append(str(i)+'_'+str(j))
    x_test = np.array(x_test) 
    return x_test, img_name

