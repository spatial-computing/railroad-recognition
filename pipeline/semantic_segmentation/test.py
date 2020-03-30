import cv2
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from models.pspnet_modified import modified_pspnet
#from models.pspnet_deconv import modified_pspnet
#from models.pspnet_drn import create_pspnet_drn

os.environ["CUDA_VISIBLE_DEVICES"]="0"

TRAINED_MODEL_PATH = ''
DATA_PATH = './data/CO_Louisville_1965'
MAP_NAME = 'CO_Louisville_1965_degeo.png'
MAP_PATH = os.path.join(DATA_PATH, MAP_NAME)
WIN_SIZE = 320
NB_CLASSES = 2
STRIDE = 50
row_start, row_end = 595, 12145#535, 10915#742， 12310#705, 12277
col_start, col_end = 1305, 10195#588, 8550#1175， 10078#1340, 10245
SAVE_MODEL_DIR = './trained_models/'#'./logs'#
TRAINED_MODEL_NAME = 'louisville_railroads_1965_pspnetdeconv.hdf5'#'louisville_railroads_1965_pspnetdrn.hdf5'
SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR, TRAINED_MODEL_NAME)
PREDICTION_NAME = 'louisville_railroads_1965_pspnetdeconv.png'
PREDICTION_PATH = os.path.join(DATA_PATH, PREDICTION_NAME)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model = modified_pspnet(WIN_SIZE, NB_CLASSES)
#model = create_pspnet_drn((WIN_SIZE,WIN_SIZE,3), NB_CLASSES)
model.summary()
model.load_weights(SAVE_MODEL_PATH)

map_img = cv2.imread(MAP_PATH)
pred_map = np.zeros((map_img.shape[0], map_img.shape[1]))

count = 0
for row in range(row_start, row_end-WIN_SIZE, STRIDE):
    for col in range(col_start, col_end-WIN_SIZE, STRIDE):
        sub_img = map_img[row:row+WIN_SIZE, col:col+WIN_SIZE].astype('float64')
        x_test = preprocess_input(np.array([sub_img]))
        res = model.predict(x_test)[0]
        pred = np.argmax(np.squeeze(res), axis=-1).astype(np.uint8)
            #pred_map[row:row+WIN_SIZE, col:col+WIN_SIZE] = pred*255.0
        pred_map[row:row+WIN_SIZE, col:col+WIN_SIZE] = np.logical_or(pred_map[row:row+WIN_SIZE, col:col+WIN_SIZE], pred)
        count += 1
        if count % 1000 == 0:
            print("************************"+str(count)+"************************")
                
cv2.imwrite(PREDICTION_PATH, pred_map*255.0)
