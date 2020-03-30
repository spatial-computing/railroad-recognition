from .pspnet import PSPNet
from keras.layers import *
from keras.models import Model

def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width, channel = shape
    resized = ktf.image.resize_images(x, [new_height, new_width],
                                      align_corners=True)
    return resized

def modified_pspnet(input_size, nb_classes, pretrained=False, resume=False, weights_path='/home/weiweidu/Keras-FCN/Models/PSPNet/checkpoint_weights.hdf5'):
    base_model = PSPNet(input_shape=(input_size, input_size, 3),nb_classes=21)
    if pretrained == True:
        base_model.load_weights(weights_path)
        
    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(base_model.layers[-4].output)
    x = Lambda(Interp, arguments={'shape': (input_size, input_size, nb_classes)})(x)
    x = Activation('softmax')(x)
    model = Model(base_model.input, x)
    if resume == True:
        model.load_weights(weights_path)
    return model
