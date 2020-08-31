import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import argparse
import os
import cv2
from models.vgg import vgg16
import argparse
from utils import load_data

WIN_SIZE = 40
NUM_CLS = 2
parser = argparse.ArgumentParser()
parser.add_argument("--test_map_path", type=str)
parser.add_argument("--pred_path", type=str)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

TEST_MAP_PATH = args.test_map_path
PRED_PATH = args.pred_path
STRIDE = 20

model = vgg16()
model.load_weights(args.model_path)

test_data, img_names = load_data.load_all_data(MAP_PATH, '', WIN_SIZE, 20, flip=False)
test_data  = preprocess_input(test_data.astype('float64'))

pred = model.predict(test_data)
map_img = cv2.imread(MAP_PATH)
res = np.zeros((map_img.shape[0], map_img.shape[1]))
for i in range(pred.shape[0]): 
    if pred[i, 0] > 0.99:
        idx = img_names[i].split('_')
        x, y = int(idx[0]), int(idx[1])
        res[x:x+WIN_SIZE, y:y+WIN_SIZE] = 255
cv2.imwrite(PRED_PATH, res)
