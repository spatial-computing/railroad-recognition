from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.models import Model
from keras import backend as K
import cv2

def vgg16(IMG_SIZE=48):
  base_model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMG_SIZE,IMG_SIZE,3), pooling=None, classes=2)
  flatten = Flatten()(base_model.output)
  fc1 = Dense(1024, activation='relu')(flatten)
  cls = Dense(NUM_CLS, activation='linear')(fc1)
  cls = Activation(tf.nn.softmax)(cls)
  model = Model(base_model.input, cls)
  return model
