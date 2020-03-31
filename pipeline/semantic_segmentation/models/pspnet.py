# Base model as VGG16 

from __future__ import print_function
from math import ceil
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda, Reshape
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras import initializers
from keras.regularizers import l2
import keras

learning_rate = 1e-4  # Layer specific learning rate
# Weight decay not implemented
def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


def VGG16(input_shape, weights=None):
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights=weights, input_tensor=None, input_shape=input_shape, pooling=None, classes=20)
    x = Conv2DTranspose(64, (3,3), strides=(4,4), padding='same', name='convTrans_32', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(base_model.layers[-1].output)    #0.0000001
    x = Activation('relu')(x)
    model = Model(base_model.input, x)
    return model

def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width, channel = shape
    resized = ktf.image.resize_images(x, [new_height, new_width],
                                      align_corners=True)
    return resized

def interp_block(prev_layer, level, feature_map_shape ):
    if level == 1:
        kernel = (feature_map_shape[0], feature_map_shape[0])
    else:
        kernel = (30/level, 30/level)
    strides = (int(feature_map_shape[0]/level), int(feature_map_shape[0]/level))
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name="level"+str(level),
                        use_bias=False,kernel_regularizer=l2(1e-4))(prev_layer)
    #prev_layer = BN()(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Lambda(Interp, arguments={'shape': feature_map_shape})(prev_layer)
    return prev_layer

def build_pyramid_pooling_module(vgg, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    #feature_map_size = (40,40,64)
    print("PSP module will interpolate to a final feature map size of %s" % (feature_map_size, ))

    interp_block1 = interp_block(vgg, 6, feature_map_size)
    interp_block2 = interp_block(vgg, 3, feature_map_size)
    interp_block3 = interp_block(vgg, 2, feature_map_size)
    interp_block6 = interp_block(vgg, 1, feature_map_size)

    # concat all these layers. resulted shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([vgg,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res

def PSPNet(input_shape=(320,320, 3), nb_classes=2, activation='softmax', learning_rate=0.0001):
    res = VGG16(input_shape)
    psp = build_pyramid_pooling_module(res.get_layer('convTrans_32').output, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False,kernel_regularizer=l2(1e-4))(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6",kernel_regularizer=l2(1e-4))(x)
    x = Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1], nb_classes)})(x)
   # x = Reshape((320*320, 4))(x)
    x = Activation(activation)(x)

    model = Model(inputs=res.input, outputs=x)

    # Solver
#    loss = ""
#    if activation == 'softmax':
#        loss = 'categorical_crossentropy'
#    elif activation == 'sigmoid':
#        loss = 'binary_crossentropy'

#    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd,
#                  loss=loss,
#                  metrics=['accuracy'])
    return model

