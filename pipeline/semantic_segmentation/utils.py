from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
from keras import initializers
import tensorflow as tf
import cv2
import os
import numpy as np
from itertools import product
import itertools
import keras.backend.tensorflow_backend as tfbe
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

#DATA_PATH = './data/CO_Louisville_1950'
#MAP_NAME = 'CO_Louisville_450550_1950_24000_geo.png'
#LABEL_NAME = 'louisville_railroads_1950_aligned.png'
#MAP_PATH = os.path.join(DATA_PATH, MAP_NAME)
#LABEL_PATH = os.path.join(DATA_PATH, LABEL_NAME)
#OBJECT_LIST = ['railroads', 'roads', 'waterlines']
#OBJECT_NUMS = [400, 800, 400, 400]
#WIN_SIZE = 320
#NB_CLASSES = 2
#NUM_POS_AUG = 4

############ Data generation ##########

def square_from_center(image, center_y, center_x, window_size):
    origin_y = int(center_y - (window_size - 1) / 2)
    origin_x = int(center_x - (window_size - 1) / 2)
    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size]).astype(np.float64)

def square_from_center_label(image, center_y, center_x, window_size):
    origin_y = int(center_y - (window_size - 1) / 2)
    origin_x = int(center_x - (window_size - 1) / 2)
    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size]).astype(np.float64) / 255.

def generate_data_from_center_coords(image, coordinates, window_size):
    data = []

    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center(image, y_coord, x_coord, window_size)
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            data.append(cropped_image) 
    return np.array(data)

def generate_data_from_center_coords_label(image, coordinates, window_size):
    data = []

    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center_label(image, y_coord, x_coord, window_size)
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            data.append(cropped_image)
    return np.array(data)

def label_generation(image, window_size, NB_CLASSES):
    tmpt = np.array(image[:,:,0]).astype(np.float32)
    label = np.zeros((window_size, window_size, 2))
    for row in range(tmpt.shape[0]):
        label[row, :] = np_utils.to_categorical(np.array(tmpt[row, :]), num_classes=NB_CLASSES)
    return label


def points_generator(DATA_PATH,OBJECT_LIST,OBJECT_NUMS):
    obj_list = []
    for obj_index in range(len(OBJECT_LIST)):
        obj_name = OBJECT_LIST[obj_index]
        print(os.path.join(DATA_PATH, obj_name+'.txt'))
        obj_points = np.loadtxt(os.path.join(DATA_PATH, obj_name+'.txt'), dtype=np.int32, delimiter=",")
        print(obj_points.shape)
        np.random.shuffle(obj_points)
        obj_list.append(obj_points)

    x_train_coor_pos = obj_list[0][:OBJECT_NUMS[0]]
    for i in range(1, len(OBJECT_LIST)):
        if i == 1:
            x_train_coor_neg = obj_list[i][:OBJECT_NUMS[i]]
        else:
            x_train_coor_neg = np.concatenate((x_train_coor_neg, obj_list[i][:OBJECT_NUMS[i]]), axis=0)
   
    return x_train_coor_pos, x_train_coor_neg

def data_generator(DATA_PATH, MAP_PATH,LABEL_PATH,OBJECT_LIST,OBJECT_NUMS,WIN_SIZE,NUM_POS_AUG,NB_CLASSES,augment=True, check_data=False):
    img  = cv2.imread(MAP_PATH)
    label = cv2.imread(LABEL_PATH)
    print('map, label shape: ', img.shape, label.shape)
    pos_coor, neg_coor =  points_generator(DATA_PATH,OBJECT_LIST,OBJECT_NUMS)
    if augment == False:
        #coor = np.vstack((pos_coor, neg_coor))
        coor = pos_coor
        x_train = generate_data_from_center_coords(img, coor, WIN_SIZE)
        y_img = generate_data_from_center_coords_label(label, coor, WIN_SIZE)
        y_train = []
        for i in range(y_img.shape[0]):
            y_train.append(label_generation(y_img[i,:,:,:], WIN_SIZE, NB_CLASSES))
      
        y_train = np.array(y_train)
        x_train = preprocess_input(x_train)

    else:
        x_train, y_train = data_augmentation(img, label, pos_coor, neg_coor, WIN_SIZE, NUM_POS_AUG, NB_CLASSES)
    if check_data==True:
        for i in range(x_train.shape[0]):
            img = image.array_to_img(x_train[i])
            img.save('./check_data/'+str(i)+'.png') 
            cv2.imwrite('./check_data/'+str(i)+'_l.png', y_train[i][:,:,1]*255)
    return x_train, y_train

def data_augmentation(img, label, pos_coor, neg_coor, WIN_SIZE, NUM_POS_AUG, NB_CLASSES):
    x_train_pos = generate_data_from_center_coords(img, pos_coor, WIN_SIZE)
    y_train_pos = generate_data_from_center_coords_label(label, pos_coor, WIN_SIZE)

    x_train_neg = generate_data_from_center_coords(img, neg_coor, WIN_SIZE)
    y_train_neg = generate_data_from_center_coords_label(label,neg_coor, WIN_SIZE)
    x_train_expanded, y_train_expanded = np.array([]), np.array([])

    datagen = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                horizontal_flip=True, 
                vertical_flip=True,
                rotation_range=180)

    datagen_orig = ImageDataGenerator()
    datagen_orig.fit(x_train_pos, seed=1)
    #### rotation augmentation for x_pos
    for num in range(NUM_POS_AUG): 
        batches = 0
        for x_batch_orig in datagen_orig.flow(x_train_pos, batch_size=10, seed=num+1):
            datagen.fit(x_batch_orig, seed=num+1)
            b = 0
            for x_batch in datagen.flow(x_batch_orig, batch_size=10, seed=num+1):
                if x_train_expanded.shape[0] == 0:
                    x_train_expanded = np.concatenate((x_batch_orig, x_batch), axis=0)
                elif num == 0:
                    x_train_expanded = np.concatenate((x_train_expanded, x_batch_orig, x_batch), axis=0)
                else:
                    x_train_expanded = np.concatenate((x_train_expanded, x_batch), axis=0)
                b += 1
                if b >= len(x_batch_orig) / 10:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
            batches += 1
            if batches >= len(x_train_pos)  / 10:
                break
    #### rotation augmentation for y_pos
    for num in range(NUM_POS_AUG):
        batches = 0
        for y_batch_orig in datagen_orig.flow(y_train_pos, batch_size=10, seed=num+1):
            datagen.fit(y_batch_orig, seed=num+1)
            b = 0
            for y_batch in datagen.flow(y_batch_orig, batch_size=10, seed=num+1):
                if y_train_expanded.shape[0] == 0:
                    y_train_expanded = np.concatenate((y_batch_orig, y_batch), axis=0)
                elif num == 0:
                    y_train_expanded = np.concatenate((y_train_expanded, y_batch_orig, y_batch), axis=0)
                else:
                    y_train_expanded = np.concatenate((y_train_expanded, y_batch), axis=0)
                b += 1
                if b >= len(y_batch_orig) / 10:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
            batches += 1
            if batches >= len(y_train_pos)  / 10:
                break
    
    batches = 0
    for x_batch_orig in datagen_orig.flow(x_train_neg, batch_size=10, seed=1):
        datagen.fit(x_batch_orig, seed=1)
        b = 0
        for x_batch in datagen.flow(x_batch_orig, batch_size=10, seed=1):
            if x_train_expanded.shape[0] != 0:
                x_train_expanded = np.concatenate((x_train_expanded, x_batch_orig, x_batch), axis=0)
            else:
                x_train_expanded = np.concatenate((x_batch_orig, x_batch), axis=0)
            b += 1
            if b >= len(x_batch_orig) / 10:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        batches += 1
        if batches >= len(x_train_neg)  / 10:
            break

    batches = 0
    for y_batch_orig in datagen_orig.flow(y_train_neg, batch_size=10, seed=1):
        #        print "Original data shape = ", x_batch_orig.shape
        datagen.fit(y_batch_orig, seed=1)
        b = 0
        for y_batch in datagen.flow(y_batch_orig, batch_size=10, seed=1):
            if y_train_expanded.shape[0] != 0:
                y_train_expanded = np.concatenate((y_train_expanded, y_batch_orig, y_batch), axis=0)
            else:
                y_train_expanded = np.concatenate((y_batch_orig, y_batch), axis=0)
            b += 1
            if b >= len(y_batch_orig) / 10:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        batches += 1
        if batches >= len(y_train_neg)  / 10:
            break
        

    x_train_expanded = preprocess_input(x_train_expanded)
    annotation_train = []
    for i in range(y_train_expanded.shape[0]):
        annotation_train.append(label_generation(y_train_expanded[i,:,:,:], WIN_SIZE, NB_CLASSES))
    annotation_train = np.array(annotation_train)
    return x_train_expanded, annotation_train

