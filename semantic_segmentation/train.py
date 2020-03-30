import utils
import os
from keras.optimizers import SGD, Adam
from models.pspnet_modified import modified_pspnet
#from models.pspnet_drn import create_pspnet_drn
from models.pspnet_deconv import modified_pspnet
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]="0"
TRAINED_MODEL_PATH = ''
DATA_PATH = './data/CO_Louisville_1965'
MAP_NAME = 'CO_Louisville_1965_degeo.png'
LABEL_NAME = 'Louisville_railroads_aligned_1965_buffer1.png'
MAP_PATH = os.path.join(DATA_PATH, MAP_NAME)
LABEL_PATH = os.path.join(DATA_PATH, LABEL_NAME)
OBJECT_LIST = ['railroads', 'roads', 'waterlines']
OBJECT_NUMS = [100, 400, 400]
WIN_SIZE = 320
NB_CLASSES = 2
NUM_POS_AUG = 4
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 20
SAVE_MODEL_DIR = './trained_models/'
TRAINED_MODEL_NAME = 'louisville_railroads_1965_pspnetdeconv.hdf5'
SAVE_MODEL_PATH = os.path.join(SAVE_MODEL_DIR, TRAINED_MODEL_NAME)
PRETRAINED = True
RESUME = False
WIGHTS_PATH = '../Keras-FCN/Models/PSPNet/checkpoint_weights.hdf5'
CHECK_DATA = False
#'../Keras-FCN/Models/PSPNet/checkpoint_weights.hdf5'
#'./trained_models/louisville_railroads_1942_pspnetdrn_e600.hdf5'

x_train, y_train = utils.data_generator(DATA_PATH,MAP_PATH,LABEL_PATH,OBJECT_LIST,OBJECT_NUMS,\
                                        WIN_SIZE,NUM_POS_AUG,NB_CLASSES,augment=True, check_data=CHECK_DATA)
print(x_train.shape, y_train.shape)

model = modified_pspnet(WIN_SIZE, NB_CLASSES, PRETRAINED, RESUME, WIGHTS_PATH)
#model = create_pspnet_drn((WIN_SIZE,WIN_SIZE,3), NB_CLASSES, PRETRAINED, RESUME, WIGHTS_PATH)
model.summary()
optimizer = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9)
#optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['binary_accuracy'])
checkpoint = ModelCheckpoint('./logs/weights{epoch:08d}.h5',save_weights_only=True, period=5)

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])

model.save_weights(SAVE_MODEL_PATH)
