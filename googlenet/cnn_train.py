import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import Adam
from keras.utils import np_utils

from datetime import datetime

import cv2
import os
import numpy as np

parser = argparse.ArgumentParser(description='Fit VGG on training data')
parser.add_argument("img", help='path to raster map')
parser.add_argument("pos_coords", help='path to file with positive coordinates')
parser.add_argument("neg_coords", help='path to file with negative coordinates')
parser.add_argument("output_dir", help='directory where output will be saved to')
parser.add_argument("model_name", help='what to call saved trained model')
parser.add_argument("--window_size", help='size, in pixels, of input data', default=47, type=int)

args = parser.parse_args()
img_path = args.img
pos_coords_path = args.pos_coords
neg_coords_path = args.neg_coords
output_dir = args.output_dir
model_name = args.model_name
window_size = args.window_size


img = cv2.imread(img_path)

# Check that the image has been properly loaded
assert img is not None

pos_coords = np.loadtxt(pos_coords_path, dtype=np.uint16, delimiter=",")
neg_coords = np.loadtxt(neg_coords_path, dtype=np.uint16, delimiter=",")


def square_from_center(image, center_y, center_x, window_size):
    """
    Crop a square region from :image centered on :center_y, :center_x with edge length of :window_size

    :param image:  numpy representation of image to crop
    :param center_y: y coordinate of center
    :param center_x: x coordinate of center
    :param window_size: size of cropped region

    :return: numpy representation of cropped square region
    """

    origin_y = center_y - (window_size - 1) / 2
    origin_x = center_x - (window_size - 1) / 2

    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size])


def generate_data_from_center_coords(image, coordinates, window_size):
    """
    Generate an array of cropped square region from :image where each region is centered on coordinates given in
    :coordinates array of centers and has size of :window_size

    :param image: image to be cropped
    :param coordinates: an array of coordinates that represent centers of cropped regions
    :param window_size: size of cropped region

    :return: array of generated cropped regions
    """

    data = []

    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center(image, y_coord, x_coord, window_size).astype(np.float32) / 255.
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            data.append(cropped_image)

    return np.array(data)


pos_train = generate_data_from_center_coords(img, pos_coords, window_size)
neg_train = generate_data_from_center_coords(img, neg_coords, window_size)

print " ----------- "
print "positive shape: " + str(pos_train.shape)
print "negative shape: " + str(neg_train.shape)
print " ----------- "

X_train = np.vstack((pos_train, neg_train))
y_train = np.concatenate((np.ones(len(pos_train)), np.zeros(len(neg_train))))
y_train = np_utils.to_categorical(np.array(y_train), nb_classes=2)


print "X_train shape: " + str(X_train.shape)
print "y_train shape: " + str(y_train.shape)



#### DEFINE MODEL ####
# TODO: This should probably be in its own class
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(window_size, window_size, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

adam = Adam(lr=5e-5)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])


#### FIT MODEL ####

class_weight = {1: len(X_train)/float(len(pos_train)) - 1, 0: 1.}
print "Using the following class weights: " + str(class_weight)
time_start = datetime.now()
print str(time_start) + ": Fitting"

history = model.fit(X_train, y_train, nb_epoch=50, verbose=1, class_weight=class_weight)

time_end = datetime.now()

print str(time_end) + ": Saving model"
model_output_path = os.path.join(output_dir, model_name + ".h5")
model.save(model_output_path)

history_output_path = os.path.join(output_dir, model_name + "_history.txt")

with open(history_output_path, "w") as f:
    f.write("Training time (s): " + str((time_end - time_start).total_seconds()))
    f.write(str(history.params))
    f.write(str(history.history))
